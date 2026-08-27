#!/usr/bin/env python
"""
Run the OVTAS pipeline and save per-frame predictions.

Two modes:

1. **Fast mode** (recommended for iterating on Stage 2 hyperparameters):
   reuse embeddings already extracted by ``extract_embeddings.py`` and
   only re-run Stage 1 similarity + Stage 2 decoding.

       python scripts/run_pipeline.py \\
           --config configs/gtea.yaml \\
           --embeddings-dir outputs/embeddings \\
           --out-dir outputs/predictions

2. **Full mode**: run the encoder from scratch (slower, needed the
   first time or when nothing has been cached yet).

       python scripts/run_pipeline.py \\
           --config configs/gtea.yaml \\
           --out-dir outputs/predictions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from ovtas.config import load_config
from ovtas.data import DATASETS, load_frames
from ovtas.pipeline import OVTASPipeline
from ovtas.stage1_faes import compute_faes, compute_faes_plus
from ovtas.stage2_smts import decode_asot, shuffle_action_order
from ovtas.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Directory of cached .npz embeddings from extract_embeddings.py. "
        "If omitted, the encoder is run from scratch on raw video frames.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory to write per-video prediction .npz files."
    )
    return parser.parse_args()


def _run_from_cache(embeddings_dir: str, out_dir: str, config, rng: np.random.Generator) -> None:
    files = sorted(f for f in os.listdir(embeddings_dir) if f.endswith(".npz"))
    for fname in tqdm(files, desc="Decoding (Stage 2/3) from cached embeddings"):
        video_id = fname[:-4]
        data = np.load(os.path.join(embeddings_dir, fname), allow_pickle=True)
        if getattr(config, "faes_plus", None) and config.faes_plus.enabled:
            faes_out = compute_faes_plus(
                data["frame_embeddings"],
                data["action_embeddings"],
                config=config.faes_plus,
                already_normalized=True,
            )
        else:
            faes_out = compute_faes(
                data["frame_embeddings"], data["action_embeddings"], already_normalized=True
            )
        num_actions = faes_out.num_actions
        action_order = (
            shuffle_action_order(num_actions, rng=rng)
            if config.shuffle_actions
            else np.arange(num_actions)
        )
        shuffled = faes_out.similarity[:, action_order]
        smts_out = decode_asot(shuffled, config=config.asot)
        predicted_labels = action_order[smts_out.labels]

        save_dict = {
            "predicted_labels": predicted_labels,
            "frame_labels": data["frame_labels"],
            "label_names": data["label_names"],
        }
        if getattr(faes_out, "raw_similarity", None) is not None:
            save_dict["similarity"] = faes_out.similarity
            save_dict["raw_similarity"] = faes_out.raw_similarity

        np.savez_compressed(
            os.path.join(out_dir, fname),
            **save_dict,
        )


def _run_from_scratch(out_dir: str, config, rng: np.random.Generator) -> None:
    dataset_kwargs = {"root": config.data.root} if config.data.root else {}
    dataset = DATASETS.build(config.data.name, **dataset_kwargs)
    if config.encoder.name == "mock":
        encoder_kwargs = {"embed_dim": config.encoder.embed_dim}
    else:
        encoder_kwargs = {
            k: v
            for k, v in {
                "model_name": config.encoder.model_name,
                "pretrained": config.encoder.pretrained,
                "device": config.encoder.device,
                "batch_size": config.encoder.batch_size,
            }.items()
            if v is not None
        }
    pipeline = OVTASPipeline(
        encoder_name=config.encoder.name,
        encoder_kwargs=encoder_kwargs,
        faes_plus_config=getattr(config, "faes_plus", None),
        asot_config=config.asot,
        prompt_dataset=config.data.prompt_dataset,
        shuffle_actions=config.shuffle_actions,
        seed=config.seed,
    )

    label_names = dataset.label_names()
    for video_id in tqdm(dataset.list_videos(), desc="Running full pipeline"):
        annotation = dataset.load_annotation(video_id)
        if hasattr(dataset, "get_frames"):
            # In-memory datasets (e.g. SyntheticTASDataset) have no
            # real file on disk to read frames from.
            frames = dataset.get_frames(video_id)
            frames = frames[:: config.data.frame_stride]
            if config.data.max_frames is not None:
                frames = frames[: config.data.max_frames]
        else:
            frames = load_frames(
                annotation.frames_dir_or_video_path,
                stride=config.data.frame_stride,
                max_frames=config.data.max_frames,
            )
        gt_labels = annotation.frame_labels[:: config.data.frame_stride][: len(frames)]

        result = pipeline.run(frames, label_names, video_id=video_id)
        np.savez_compressed(
            os.path.join(out_dir, f"{video_id}.npz"),
            predicted_labels=result.predicted_labels,
            frame_labels=gt_labels,
            label_names=np.array(label_names, dtype=object),
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    if args.embeddings_dir:
        logger.info("Running in fast mode from cached embeddings: %s", args.embeddings_dir)
        _run_from_cache(args.embeddings_dir, args.out_dir, config, rng)
    else:
        logger.info("No --embeddings-dir given; running the encoder from scratch.")
        _run_from_scratch(args.out_dir, config, rng)

    logger.info("Done. Predictions written to %s", args.out_dir)


if __name__ == "__main__":
    main()
