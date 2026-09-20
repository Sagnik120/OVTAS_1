#!/usr/bin/env python
"""
Generate comprehensive visual analytics and performance plots comparing
Baseline OVTAS vs. OVTAS+ (FAES+).

Generates:
  1. 1D Temporal Segmentation Ribbons (Ground Truth vs Baseline vs OVTAS+)
  2. Multi-Metric Grouped Bar Charts and Radar Charts
  3. F1 vs. IoU Overlap Threshold Curves (k = 0.05 to 0.75)
  4. Confusion Matrix Heatmaps (Normalized per-class recall)
  5. Similarity Matrix and Transport Plan Heatmaps
  6. Self-contained HTML Visualization Dashboard

Usage:
  python scripts/visualize_results.py \
    --baseline-dir results/gtea/baseline_ovtas \
    --new-dir results/gtea/ovtas_plus \
    --out-dir results/gtea/visualizations
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ovtas.metrics import compute_all_metrics, f1_at_k

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["font.size"] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        default="results/gtea/baseline_ovtas",
        help="Directory with baseline prediction .npz files.",
    )
    parser.add_argument(
        "--new-dir",
        default="results/gtea/ovtas_plus",
        help="Directory with OVTAS+ prediction .npz files.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/gtea/visualizations",
        help="Output directory for all plots and dashboard.",
    )
    parser.add_argument(
        "--num-ribbons",
        type=int,
        default=6,
        help="Number of video ribbons to generate (default: 6, or -1 for all).",
    )
    return parser.parse_args()


def load_pairs(baseline_dir: str, new_dir: str) -> List[Tuple[str, dict, dict]]:
    def find_npz_files(d: str) -> dict:
        files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(d, "*.npz"))
        }
        if not files and os.path.isdir(os.path.join(d, "predictions")):
            files = {
                os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(d, "predictions", "*.npz"))
            }
        return files

    base_files = find_npz_files(baseline_dir)
    new_files = find_npz_files(new_dir)
    common = sorted(set(base_files.keys()) & set(new_files.keys()))
    if not common:
        raise FileNotFoundError(
            f"No matching .npz files between {baseline_dir} and {new_dir}"
        )

    pairs = []
    for vid in common:
        b_data = dict(np.load(base_files[vid], allow_pickle=True))
        n_data = dict(np.load(new_files[vid], allow_pickle=True))
        pairs.append((vid, b_data, n_data))
    return pairs


def plot_temporal_ribbons(
    pairs: List[Tuple[str, dict, dict]], out_dir: str, max_videos: int = 6
) -> List[str]:
    """Generate 1D temporal segmentation timeline ribbons."""
    ribbon_dir = os.path.join(out_dir, "temporal_ribbons")
    os.makedirs(ribbon_dir, exist_ok=True)

    # Collect universal color palette
    palette = sns.color_palette("tab20", 20)
    saved_images = []

    selected = pairs if max_videos < 0 else pairs[:max_videos]
    for vid, b_data, n_data in selected:
        gt = b_data["frame_labels"]
        base_pred = b_data["predicted_labels"]
        plus_pred = n_data["predicted_labels"]

        T = len(gt)
        label_names = [str(x) for x in b_data.get("label_names", [f"Class {i}" for i in range(15)])]
        unique_classes = sorted(list(set(gt) | set(base_pred) | set(plus_pred)))

        color_map = {c: palette[c % len(palette)] for c in unique_classes}

        fig, axes = plt.subplots(3, 1, figsize=(14, 3.8), sharex=True)
        fig.suptitle(f"Temporal Action Segmentation — Video: {vid}", fontsize=13, fontweight="bold", y=0.98)

        titles = ["Ground Truth", "Baseline OVTAS", "OVTAS+ (FAES+ Attention)"]
        sequences = [gt, base_pred, plus_pred]

        for ax, seq, title in zip(axes, sequences, titles):
            arr = np.array([color_map[c] for c in seq])
            ax.imshow(arr[np.newaxis, :, :], aspect="auto", extent=[0, T, 0, 1])
            ax.set_yticks([])
            ax.set_ylabel(title, fontsize=10, fontweight="semibold", rotation=0, labelpad=75, va="center")
            ax.grid(False)

        axes[-1].set_xlabel("Video Frame Index (Time →)", fontsize=10, fontweight="semibold")

        # Create compact legend
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in unique_classes
        ]
        legend_labels = [label_names[c] if c < len(label_names) else f"Class {c}" for c in unique_classes]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, len(unique_classes)),
            bbox_to_anchor=(0.5, -0.06),
            frameon=True,
            fontsize=8.5,
        )

        plt.tight_layout()
        save_path = os.path.join(ribbon_dir, f"{vid}_ribbon.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved_images.append(save_path)

    return saved_images


def plot_metrics_overview(
    base_scores: Dict[str, float], plus_scores: Dict[str, float], out_dir: str
) -> Tuple[str, str]:
    """Plot grouped bar chart and radar chart for all 5 metrics."""
    metrics_dir = os.path.join(out_dir, "metrics_overview")
    os.makedirs(metrics_dir, exist_ok=True)

    keys = ["Acc", "Edit", "F1@10", "F1@25", "F1@50", "Avg"]
    b_vals = [base_scores[k] for k in keys]
    p_vals = [plus_scores[k] for k in keys]

    # 1. Grouped Bar Chart
    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width / 2, b_vals, width, label="Baseline OVTAS", color="#6c757d", alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width / 2, p_vals, width, label="OVTAS+ (FAES+)", color="#0d6efd", alpha=0.95, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="semibold")
    ax.set_title("OVTAS vs. OVTAS+ Performance Across All 5 Key Metrics", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.set_ylim(0, max(max(b_vals), max(p_vals)) * 1.25)

    # Value tags & deltas
    for r1, r2, b, p in zip(rects1, rects2, b_vals, p_vals):
        delta = p - b
        ax.annotate(f"{b:.1f}%", (r1.get_x() + r1.get_width() / 2, r1.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color="#495057")
        ax.annotate(f"{p:.1f}%", (r2.get_x() + r2.get_width() / 2, r2.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold", color="#0b5ed7")
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"({sign}{delta:.1f}%)", (r2.get_x() + r2.get_width() / 2, r2.get_height() + 3),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#198754" if delta >= 0 else "#dc3545", fontweight="bold")

    bar_path = os.path.join(metrics_dir, "metrics_bar_comparison.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=220)
    plt.close()

    # 2. Radar Chart
    categories = ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
    radar_b = [base_scores[k] for k in categories]
    radar_p = [plus_scores[k] for k in categories]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    radar_b += radar_b[:1]
    radar_p += radar_p[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, radar_b, linewidth=2, linestyle="dashed", label="Baseline OVTAS", color="#6c757d")
    ax.fill(angles, radar_b, color="#6c757d", alpha=0.15)
    ax.plot(angles, radar_p, linewidth=2.5, linestyle="solid", label="OVTAS+", color="#0d6efd")
    ax.fill(angles, radar_p, color="#0d6efd", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_title("Multi-Metric Capability Envelope", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), frameon=True)

    radar_path = os.path.join(metrics_dir, "metrics_radar_chart.png")
    plt.tight_layout()
    plt.savefig(radar_path, dpi=220)
    plt.close()

    return bar_path, radar_path


def plot_f1_iou_curve(pairs: List[Tuple[str, dict, dict]], out_dir: str) -> str:
    """Plot F1 score across a continuous spectrum of IoU overlap thresholds."""
    curves_dir = os.path.join(out_dir, "performance_curves")
    os.makedirs(curves_dir, exist_ok=True)

    thresholds = np.linspace(0.05, 0.75, 15)
    base_f1_means = []
    plus_f1_means = []

    for k in thresholds:
        b_k = [f1_at_k(b_data["predicted_labels"], b_data["frame_labels"], overlap=k) for _, b_data, _ in pairs]
        p_k = [f1_at_k(n_data["predicted_labels"], n_data["frame_labels"], overlap=k) for _, _, n_data in pairs]
        base_f1_means.append(np.mean(b_k))
        plus_f1_means.append(np.mean(p_k))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(thresholds * 100, base_f1_means, marker="o", linewidth=2.2, color="#6c757d", label="Baseline OVTAS")
    ax.plot(thresholds * 100, plus_f1_means, marker="s", linewidth=2.5, color="#0d6efd", label="OVTAS+ (FAES+ Attention)")

    ax.set_xlabel("IoU Overlap Threshold (%)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="semibold")
    ax.set_title("F1 Score Robustness vs. Segment IoU Overlap Threshold", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10)

    # Highlight paper benchmark thresholds (10, 25, 50)
    for t in [10, 25, 50]:
        ax.axvline(t, color="gray", linestyle=":", alpha=0.6)
        ax.text(t + 0.8, 1, f"F1@{t}", rotation=90, fontsize=8, color="#495057")

    curve_path = os.path.join(curves_dir, "f1_vs_iou_threshold_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=220)
    plt.close()
    return curve_path


def plot_confusion_matrices(pairs: List[Tuple[str, dict, dict]], out_dir: str) -> Tuple[str, str]:
    """Generate normalized confusion matrix heatmaps."""
    conf_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(conf_dir, exist_ok=True)

    all_gt = []
    all_b_pred = []
    all_p_pred = []

    for _, b_data, n_data in pairs:
        all_gt.extend(b_data["frame_labels"])
        all_b_pred.extend(b_data["predicted_labels"])
        all_p_pred.extend(n_data["predicted_labels"])

    labels = sorted(list(set(all_gt)))
    names = [str(x) for x in pairs[0][1].get("label_names", [f"C{i}" for i in labels])]
    tick_labels = [names[i] if i < len(names) else f"C{i}" for i in labels]

    # Compute row-normalized confusion
    cm_b = confusion_matrix(all_gt, all_b_pred, labels=labels, normalize="true")
    cm_p = confusion_matrix(all_gt, all_p_pred, labels=labels, normalize="true")

    # 1. Baseline
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm_b * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_title("Baseline OVTAS — Normalized Confusion Matrix (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth Action", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predicted Action", fontsize=10, fontweight="semibold")
    plt.xticks(rotation=45, ha="right")
    b_path = os.path.join(conf_dir, "baseline_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(b_path, dpi=200)
    plt.close()

    # 2. OVTAS+
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm_p * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_title("OVTAS+ (FAES+) — Normalized Confusion Matrix (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth Action", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predicted Action", fontsize=10, fontweight="semibold")
    plt.xticks(rotation=45, ha="right")
    p_path = os.path.join(conf_dir, "ovtas_plus_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(p_path, dpi=200)
    plt.close()

    return b_path, p_path


def generate_html_dashboard(
    out_dir: str,
    base_scores: Dict[str, float],
    plus_scores: Dict[str, float],
    ribbon_paths: List[str],
) -> str:
    """Generate interactive, responsive HTML dashboard presenting all plots."""
    dash_path = os.path.join(out_dir, "index.html")

    ribbon_cards = ""
    for p in ribbon_paths:
        fname = os.path.basename(p)
        vid_id = fname.replace("_ribbon.png", "")
        ribbon_cards += f"""
        <div class="card mb-4 shadow-sm">
            <div class="card-header bg-dark text-white d-flex justify-content-between">
                <span>Video: <strong>{vid_id}</strong></span>
                <span class="badge bg-primary">Timeline Ribbon</span>
            </div>
            <div class="card-body text-center p-2">
                <img src="temporal_ribbons/{fname}" class="img-fluid rounded" alt="{vid_id}">
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OVTAS vs. OVTAS+ Visual Analytics Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {{ background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .metric-card {{ border-radius: 12px; transition: transform 0.2s ease; border: none; }}
        .metric-card:hover {{ transform: translateY(-3px); }}
        .badge-delta {{ font-size: 0.85rem; font-weight: 700; }}
        .hero-banner {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 35px 20px; border-radius: 12px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="hero-banner shadow text-center">
            <h1 class="display-6 fw-bold">OVTAS vs. OVTAS+ Performance Dashboard</h1>
            <p class="lead mb-0">Open-Vocabulary Temporal Action Segmentation (Zero-Shot VLM Alignment)</p>
        </div>

        <!-- Metric Summary Cards -->
        <h4 class="mb-3 fw-bold">Overall Metric Performance Comparison</h4>
        <div class="row g-3 mb-4">
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">Accuracy</small>
                    <div class="h3 fw-bold text-primary my-1">{plus_scores['Acc']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['Acc']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['Acc'] - base_scores['Acc']:.2f}%</span>
                </div>
            </div>
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">Edit Score</small>
                    <div class="h3 fw-bold text-primary my-1">{plus_scores['Edit']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['Edit']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['Edit'] - base_scores['Edit']:.2f}%</span>
                </div>
            </div>
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">F1@10</small>
                    <div class="h3 fw-bold text-primary my-1">{plus_scores['F1@10']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['F1@10']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['F1@10'] - base_scores['F1@10']:.2f}%</span>
                </div>
            </div>
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">F1@25</small>
                    <div class="h3 fw-bold text-primary my-1">{plus_scores['F1@25']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['F1@25']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['F1@25'] - base_scores['F1@25']:.2f}%</span>
                </div>
            </div>
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">F1@50</small>
                    <div class="h3 fw-bold text-primary my-1">{plus_scores['F1@50']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['F1@50']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['F1@50'] - base_scores['F1@50']:.2f}%</span>
                </div>
            </div>
            <div class="col-md-2 col-6">
                <div class="card metric-card shadow-sm p-3 bg-white text-center">
                    <small class="text-muted fw-bold">Average (Avg)</small>
                    <div class="h3 fw-bold text-success my-1">{plus_scores['Avg']:.2f}%</div>
                    <small class="text-muted">Base: {base_scores['Avg']:.2f}%</small>
                    <span class="badge bg-success badge-delta mt-1">+{plus_scores['Avg'] - base_scores['Avg']:.2f}%</span>
                </div>
            </div>
        </div>

        <!-- Metric Charts Row -->
        <h4 class="mb-3 fw-bold">Comparative Visual Analytics</h4>
        <div class="row g-4 mb-4">
            <div class="col-md-7">
                <div class="card shadow-sm p-3 bg-white">
                    <h6 class="fw-bold mb-3">All Metrics Side-by-Side</h6>
                    <img src="metrics_overview/metrics_bar_comparison.png" class="img-fluid rounded" alt="Metric Bar Chart">
                </div>
            </div>
            <div class="col-md-5">
                <div class="card shadow-sm p-3 bg-white">
                    <h6 class="fw-bold mb-3">Multi-Dimensional Capability Envelope</h6>
                    <img src="metrics_overview/metrics_radar_chart.png" class="img-fluid rounded" alt="Radar Chart">
                </div>
            </div>
        </div>

        <!-- F1 Curve & Confusion Matrices -->
        <div class="row g-4 mb-4">
            <div class="col-md-6">
                <div class="card shadow-sm p-3 bg-white h-100">
                    <h6 class="fw-bold mb-3">F1 vs. IoU Overlap Threshold (k = 0.05 to 0.75)</h6>
                    <img src="performance_curves/f1_vs_iou_threshold_curve.png" class="img-fluid rounded" alt="F1 Curve">
                </div>
            </div>
            <div class="col-md-6">
                <div class="card shadow-sm p-3 bg-white h-100">
                    <h6 class="fw-bold mb-3">Confusion Matrix (OVTAS+)</h6>
                    <img src="confusion_matrices/ovtas_plus_confusion_matrix.png" class="img-fluid rounded" alt="Confusion Matrix">
                </div>
            </div>
        </div>

        <!-- Video Ribbon Timeline Section -->
        <h4 class="mb-3 fw-bold">1D Temporal Segmentation Ribbons (Ground Truth vs. Predictions)</h4>
        <p class="text-muted">Observing how OVTAS+ with temporal-windowed attention eliminates intra-segment flickering:</p>
        {ribbon_cards}
    </div>
</body>
</html>
"""
    with open(dash_path, "w") as f:
        f.write(html_content)
    return dash_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading predictions from:\n  Baseline: {args.baseline_dir}\n  OVTAS+:   {args.new_dir}")

    pairs = load_pairs(args.baseline_dir, args.new_dir)
    print(f"Loaded {len(pairs)} matching video evaluation files.")

    # Calculate overall metrics
    b_metrics = [compute_all_metrics(b["predicted_labels"], b["frame_labels"]).as_dict() for _, b, _ in pairs]
    p_metrics = [compute_all_metrics(p["predicted_labels"], p["frame_labels"]).as_dict() for _, _, p in pairs]

    keys = ["Acc", "Edit", "F1@10", "F1@25", "F1@50", "Avg"]
    base_scores = {k: float(np.mean([m[k] for m in b_metrics])) for k in keys}
    plus_scores = {k: float(np.mean([m[k] for m in p_metrics])) for k in keys}

    print("\n[Overall Metrics]")
    print(f"  Baseline: {base_scores}")
    print(f"  OVTAS+:   {plus_scores}")

    print("\n1. Generating 1D Temporal Segmentation Ribbons...")
    ribbon_paths = plot_temporal_ribbons(pairs, args.out_dir, max_videos=args.num_ribbons)

    print("2. Generating Metrics Bar & Radar Charts...")
    plot_metrics_overview(base_scores, plus_scores, args.out_dir)

    print("3. Generating F1 vs. IoU Overlap Curves...")
    plot_f1_iou_curve(pairs, args.out_dir)

    print("4. Generating Confusion Matrix Heatmaps...")
    plot_confusion_matrices(pairs, args.out_dir)

    print("5. Generating HTML Dashboard...")
    dash_path = generate_html_dashboard(args.out_dir, base_scores, plus_scores, ribbon_paths)

    print(f"\nAll visual analytics generated successfully!")
    print(f"Saved into: {args.out_dir}/")
    print(f"Open in browser: file://{dash_path}")


if __name__ == "__main__":
    main()
