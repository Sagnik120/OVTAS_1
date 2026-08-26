#!/usr/bin/env python
"""
Extract and cache VLM embeddings for every video in a dataset.

Splitting embedding extraction from decoding (``run_pipeline.py``)
lets you re-run Stage 2 (SMTS) with different hyperparameters (epsilon,
rho, ...) many times without re-running the (comparatively expensive)
VLM forward passes each time -- useful for the hyperparameter sweep
described in the paper's Sec. IV-A.4.

Usage
-----
    python scripts/extract_embeddings.py \\
        --config configs/gtea.yaml \\
        --out-dir outputs/embeddings

Each video's frame embeddings, action embeddings, and label names are
saved to ``<out-dir>/<video_id>.npz``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from ovtas.config import load_config
from ovtas.data import DATASETS, load_frames
from ovtas.encoders import ENCODERS
from ovtas.stage1_faes import build_prompts
from ovtas.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--out-dir", required=True, help="Directory to write per-video .npz embedding files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Building encoder '%s'...", config.encoder.name)
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
    encoder = ENCODERS.build(config.encoder.name, **encoder_kwargs)
    logger.info("Encoder ready: %r", encoder)

    logger.info("Loading dataset '%s'...", config.data.name)
    dataset_kwargs = {"root": config.data.root} if config.data.root else {}
    dataset = DATASETS.build(config.data.name, **dataset_kwargs)
    video_ids = dataset.list_videos()
    logger.info("Found %d videos.", len(video_ids))

    label_names = dataset.label_names()
    prompts = build_prompts(label_names, dataset=config.data.prompt_dataset)
    logger.info("Encoding %d candidate action prompts...", len(prompts))
    action_embeddings = encoder.encode_text(prompts)

    for video_id in tqdm(video_ids, desc="Extracting frame embeddings"):
        out_path = os.path.join(args.out_dir, f"{video_id}.npz")
        if os.path.exists(out_path):
            logger.info("Skipping %s (already extracted).", video_id)
            continue

        annotation = dataset.load_annotation(video_id)
        if hasattr(dataset, "get_frames"):
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
        frame_embeddings = encoder.encode_images(frames)

        # Ground-truth labels are per raw frame; subsample them the
        # same way frames were subsampled (stride + max_frames) so
        # `frame_labels[i]` still lines up with `frame_embeddings[i]`.
        strided_labels = annotation.frame_labels[:: config.data.frame_stride]
        if config.data.max_frames is not None:
            strided_labels = strided_labels[: config.data.max_frames]
        if len(strided_labels) != len(frames):
            logger.warning(
                "Frame/label count mismatch for %s (%d frames vs %d labels); "
                "truncating to the shorter length.",
                video_id,
                len(frames),
                len(strided_labels),
            )
            n = min(len(frames), len(strided_labels))
            frames, frame_embeddings, strided_labels = (
                frames[:n],
                frame_embeddings[:n],
                strided_labels[:n],
            )

        np.savez_compressed(
            out_path,
            frame_embeddings=frame_embeddings,
            action_embeddings=action_embeddings,
            label_names=np.array(label_names, dtype=object),
            frame_labels=strided_labels,
            frame_stride=config.data.frame_stride,
        )

    logger.info("Done. Embeddings written to %s", args.out_dir)


if __name__ == "__main__":
    main()
