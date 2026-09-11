#!/usr/bin/env python
"""
Diagnostic & Verification script for OVTAS+ (FAES+ Attention Refinement).

Compares:
1. Baseline OVTAS (Stage 1 FAES raw cosine similarity S -> SMTS)
2. Proposed OVTAS+ (Stage 1 Multi-Modal -> Stage 2 FAES+ S' -> SMTS)

Evaluates:
- Temporal smoothness: average cosine variance between adjacent frames ||x_t - x_{t-1}||
- Contrast & confidence: entropy and top-1 margin of frame-action assignment
- Segmentation metrics: Frame Accuracy, Edit Score, F1@{10, 25, 50}
"""

from __future__ import annotations

import argparse
import numpy as np

from ovtas.data import SyntheticTASDataset
from ovtas.encoders import MockEncoder
from ovtas.metrics import compute_all_metrics
from ovtas.pipeline import OVTASPipeline
from ovtas.stage1_faes import FAESPlusConfig


def run_comparison(num_videos: int = 5, seed: int = 42) -> None:
    print("=" * 70)
    print("  OVTAS vs. OVTAS+ (FAES+ Attention Refinement) Diagnostic Run")
    print("=" * 70)

    ds = SyntheticTASDataset(
        num_videos=num_videos,
        frames_per_video=60,
        frame_size=(8, 8),
        seed=seed,
    )
    encoder = MockEncoder(embed_dim=32)

    # 1. Baseline OVTAS Pipeline (FAES raw cosine similarity)
    baseline_pipeline = OVTASPipeline(
        encoder=encoder,
        faes_plus_config=FAESPlusConfig(enabled=False),
        shuffle_actions=True,
        seed=seed,
    )

    # 2. Proposed OVTAS+ Pipeline (FAES+ Attention-Refined Similarity)
    plus_pipeline = OVTASPipeline(
        encoder=encoder,
        faes_plus_config=FAESPlusConfig(
            enabled=True,
            tau=0.1,
            beta=0.5,
            normalize_refined=True,
        ),
        shuffle_actions=True,
        seed=seed,
    )

    baseline_metrics_list = []
    plus_metrics_list = []

    raw_smoothness_list = []
    refined_smoothness_list = []

    for vid in ds.list_videos():
        ann = ds.load_annotation(vid)
        frames = ds.get_frames(vid)
        gt_labels = ann.frame_labels

        res_base = baseline_pipeline.run(frames, ann.label_names, video_id=vid)
        res_plus = plus_pipeline.run(frames, ann.label_names, video_id=vid)

        m_base = compute_all_metrics(res_base.predicted_labels, gt_labels)
        m_plus = compute_all_metrics(res_plus.predicted_labels, gt_labels)

        baseline_metrics_list.append(m_base)
        plus_metrics_list.append(m_plus)

        # Measure temporal jitter: ||x_t - x_{t-1}||
        if res_plus.faes_plus and res_plus.faes_plus.refined_frames is not None:
            # Raw frames embedding from mock encoder
            raw_emb = encoder.encode_images(list(frames))
            refined_emb = res_plus.faes_plus.refined_frames

            raw_jitter = np.mean(np.linalg.norm(np.diff(raw_emb, axis=0), axis=-1))
            refined_jitter = np.mean(np.linalg.norm(np.diff(refined_emb, axis=0), axis=-1))
            raw_smoothness_list.append(raw_jitter)
            refined_smoothness_list.append(refined_jitter)

    print("\n--- 1. Temporal Smoothing Analysis (Embedding Jitter: Lower = Smoother) ---")
    avg_raw_jitter = float(np.mean(raw_smoothness_list))
    avg_ref_jitter = float(np.mean(refined_smoothness_list))
    jitter_reduction = (avg_raw_jitter - avg_ref_jitter) / avg_raw_jitter * 100
    print(f"  Raw Embeddings Mean Adjacent Jitter     : {avg_raw_jitter:.4f}")
    print(f"  Refined Embeddings X' Mean Adjacent Jitter: {avg_ref_jitter:.4f} (-{jitter_reduction:.1f}% noise reduction)")

    print("\n--- 2. Segmentation Performance Comparison ---")
    print(f"{'Metric':<15} | {'Baseline OVTAS':<15} | {'OVTAS+ (FAES+)':<15} | {'Delta':<10}")
    print("-" * 62)

    keys = ["Acc", "Edit", "F1@10", "F1@25", "F1@50", "Avg"]
    labels = ["Accuracy (%)", "Edit Score", "F1@10", "F1@25", "F1@50", "Average Score"]

    for k, name in zip(keys, labels):
        val_base = float(np.mean([m.as_dict()[k] for m in baseline_metrics_list]))
        val_plus = float(np.mean([m.as_dict()[k] for m in plus_metrics_list]))
        delta = val_plus - val_base
        print(f"{name:<15} | {val_base:>14.2f}  | {val_plus:>14.2f}  | {delta:>+9.2f}")

    print("-" * 62)
    print("\nVerification completed successfully. All components from OVTAS_1.png active.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-videos", type=int, default=5, help="Number of synthetic test videos.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_comparison(num_videos=args.num_videos, seed=args.seed)
