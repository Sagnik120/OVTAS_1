#!/usr/bin/env python
"""
Compare Baseline OVTAS vs. OVTAS+ (FAES+) Evaluation Results.

Generates:
1. Video-by-video side-by-side metric comparison table (Console, CSV, Markdown)
2. Overall aggregate score differences and deltas
3. Structured summary JSONs in the results directory

Usage
-----
# Compare GTEA results:
python scripts/compare_results.py --dataset gtea

# Or provide custom CSV paths:
python scripts/compare_results.py \\
    --baseline-csv results/gtea/baseline_ovtas/metrics.csv \\
    --plus-csv results/gtea/ovtas_plus/metrics.csv \\
    --out-dir results/gtea/comparison
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def read_metrics_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Read a metrics CSV file produced by scripts/evaluate.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics CSV file not found: {path}")

    records: Dict[str, Dict[str, float]] = {}
    metric_keys: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected header: video_id, Acc, Edit, F1@10, F1@25, F1@50, Avg
        metric_keys = header[1:]

        for row in reader:
            if not row or not row[0].strip():
                continue
            video_id = row[0].strip()
            row_dict = {}
            for k, val in zip(metric_keys, row[1:]):
                try:
                    row_dict[k] = float(val)
                except ValueError:
                    row_dict[k] = 0.0
            records[video_id] = row_dict

    return metric_keys, records


def compare_metrics(
    baseline_path: str,
    plus_path: str,
    out_dir: str,
    dataset_name: str = "Dataset",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metric_keys, base_records = read_metrics_csv(baseline_path)
    _, plus_records = read_metrics_csv(plus_path)

    common_videos = [v for v in base_records.keys() if v in plus_records and v != "OVERALL"]
    if not common_videos:
        print("Warning: No matching video IDs found between baseline and plus results.")
        return

    # 1. Console display
    print("=" * 86)
    print(f"  Benchmark Comparison: Baseline OVTAS vs. OVTAS+ ({dataset_name.upper()})")
    print("=" * 86)

    # Print overall summary if present
    if "OVERALL" in base_records and "OVERALL" in plus_records:
        b_ov = base_records["OVERALL"]
        p_ov = plus_records["OVERALL"]

        print(f"\n{'Metric':<15} | {'Baseline OVTAS':<15} | {'OVTAS+ (FAES+)':<15} | {'Delta':<12}")
        print("-" * 65)
        for k in metric_keys:
            b_val = b_ov.get(k, 0.0)
            p_val = p_ov.get(k, 0.0)
            delta = p_val - b_val
            sign = "+" if delta >= 0 else ""
            print(f"{k:<15} | {b_val:>14.2f}  | {p_val:>14.2f}  | {sign}{delta:>10.2f}")
        print("-" * 65)

    # 2. Build comparison CSV table
    csv_header = ["video_id"]
    for k in metric_keys:
        csv_header.extend([f"{k}_base", f"{k}_plus", f"{k}_delta"])

    csv_rows = []
    all_videos = common_videos + (["OVERALL"] if "OVERALL" in base_records and "OVERALL" in plus_records else [])

    for vid in all_videos:
        row = [vid]
        b_dict = base_records[vid]
        p_dict = plus_records[vid]
        for k in metric_keys:
            b_val = b_dict.get(k, 0.0)
            p_val = p_dict.get(k, 0.0)
            delta = p_val - b_val
            row.extend([f"{b_val:.2f}", f"{p_val:.2f}", f"{delta:+.2f}"])
        csv_rows.append(row)

    csv_out_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    # 3. Build Markdown Summary Table
    md_out_path = os.path.join(out_dir, "comparison.md")
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Comparison: Baseline OVTAS vs. OVTAS+\n\n")
        f.write(f"**Dataset**: `{dataset_name}`  \n")
        f.write(f"**Total Videos Evaluated**: {len(common_videos)}  \n\n")

        if "OVERALL" in base_records and "OVERALL" in plus_records:
            f.write("## Overall Dataset Summary\n\n")
            f.write("| Metric | Baseline OVTAS | OVTAS+ (FAES+) | Delta |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            b_ov = base_records["OVERALL"]
            p_ov = plus_records["OVERALL"]
            for k in metric_keys:
                b_val = b_ov.get(k, 0.0)
                p_val = p_ov.get(k, 0.0)
                delta = p_val - b_val
                sign = "+" if delta >= 0 else ""
                f.write(f"| **{k}** | {b_val:.2f} | {p_val:.2f} | **{sign}{delta:.2f}** |\n")
            f.write("\n")

        f.write("## Per-Video Breakdown\n\n")
        f.write("| Video ID | Acc (Base / Plus / Δ) | Edit (Base / Plus / Δ) | F1@10 (Base / Plus / Δ) | Avg (Base / Plus / Δ) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for vid in common_videos:
            b = base_records[vid]
            p = plus_records[vid]
            acc_str = f"{b.get('Acc', 0):.1f} / {p.get('Acc', 0):.1f} / {p.get('Acc', 0) - b.get('Acc', 0):+.1f}"
            edit_str = f"{b.get('Edit', 0):.1f} / {p.get('Edit', 0):.1f} / {p.get('Edit', 0) - b.get('Edit', 0):+.1f}"
            f1_str = f"{b.get('F1@10', 0):.1f} / {p.get('F1@10', 0):.1f} / {p.get('F1@10', 0) - b.get('F1@10', 0):+.1f}"
            avg_str = f"{b.get('Avg', 0):.1f} / {p.get('Avg', 0):.1f} / {p.get('Avg', 0) - b.get('Avg', 0):+.1f}"
            f.write(f"| `{vid}` | {acc_str} | {edit_str} | {f1_str} | {avg_str} |\n")

    # 4. Save JSON summary in respective parent folders
    base_summary_path = os.path.join(os.path.dirname(baseline_path), "summary.json")
    plus_summary_path = os.path.join(os.path.dirname(plus_path), "summary.json")
    if "OVERALL" in base_records:
        with open(base_summary_path, "w", encoding="utf-8") as f:
            json.dump(base_records["OVERALL"], f, indent=2)
    if "OVERALL" in plus_records:
        with open(plus_summary_path, "w", encoding="utf-8") as f:
            json.dump(plus_records["OVERALL"], f, indent=2)

    print(f"\nDetailed comparison files generated:")
    print(f"  - CSV table : {csv_out_path}")
    print(f"  - Markdown  : {md_out_path}")


def evaluate_dir_if_needed(pred_dir: str, csv_path: str | None = None) -> str:
    """If csv_path exists and is newer than prediction npz files, return it; otherwise recompute."""
    out_csv = csv_path or os.path.join(pred_dir, "metrics.csv")

    # Search for npz files
    npz_dir = pred_dir
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files and os.path.isdir(os.path.join(pred_dir, "predictions")):
        npz_dir = os.path.join(pred_dir, "predictions")
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    if not npz_files and os.path.isfile(out_csv):
        return out_csv

    if not npz_files:
        raise FileNotFoundError(f"No prediction .npz or metrics.csv found in {pred_dir}")

    # If metrics.csv exists and is newer than all npz files, reuse it
    if os.path.isfile(out_csv):
        csv_mtime = os.path.getmtime(out_csv)
        newest_npz = max(os.path.getmtime(f) for f in npz_files)
        if csv_mtime >= newest_npz:
            return out_csv

    # Compute metrics on the fly
    from ovtas.metrics import compute_all_metrics
    out_csv = csv_path or os.path.join(pred_dir, "metrics.csv")
    metric_keys = ["Acc", "Edit", "F1@10", "F1@25", "F1@50", "Avg"]
    rows = []

    for f in npz_files:
        vid = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f, allow_pickle=True)
        scores = compute_all_metrics(data["predicted_labels"], data["frame_labels"]).as_dict()
        rows.append({"video_id": vid, **scores})

    overall = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id"] + metric_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow({"video_id": "OVERALL", **overall})

    return out_csv


def main() -> None:
    import glob
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="Name of dataset under results/ (e.g. 'gtea')")
    parser.add_argument("--dataset-name", default=None, help="Descriptive name of dataset for report headers")
    parser.add_argument("--baseline-csv", default=None, help="Path to baseline metrics.csv")
    parser.add_argument("--plus-csv", default=None, help="Path to OVTAS+ metrics.csv")
    parser.add_argument("--baseline-dir", default=None, help="Directory with baseline predictions or results")
    parser.add_argument("--new-dir", "--plus-dir", dest="new_dir", default=None, help="Directory with OVTAS+ predictions or results")
    parser.add_argument("--out-dir", default=None, help="Output directory for comparison results")

    args = parser.parse_args()

    if args.baseline_dir and args.new_dir:
        dataset_name = args.dataset_name or "GTEA (SigLIP)"
        out_dir = args.out_dir or "results/gtea/comparison"
        baseline_csv = evaluate_dir_if_needed(args.baseline_dir)
        plus_csv = evaluate_dir_if_needed(args.new_dir)
    elif args.dataset:
        dataset = args.dataset
        dataset_name = args.dataset_name or dataset
        baseline_csv = args.baseline_csv or f"results/{dataset}/baseline_ovtas/metrics.csv"
        plus_csv = args.plus_csv or f"results/{dataset}/ovtas_plus/metrics.csv"
        out_dir = args.out_dir or f"results/{dataset}/comparison"
    else:
        if not args.baseline_csv or not args.plus_csv or not args.out_dir:
            parser.error("Specify --dataset <name>, or (--baseline-dir, --new-dir), or (--baseline-csv, --plus-csv, --out-dir).")
        dataset_name = args.dataset_name or os.path.basename(os.path.dirname(args.out_dir)) or "Dataset"
        baseline_csv = args.baseline_csv
        plus_csv = args.plus_csv
        out_dir = args.out_dir

    compare_metrics(baseline_csv, plus_csv, out_dir, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
