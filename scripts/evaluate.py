#!/usr/bin/env python
"""
Evaluate saved predictions against ground truth and report Table VI
style metrics (Accuracy, Edit, F1@10/25/50, Avg), both per-video and
averaged across the whole dataset.

Usage
-----
    python scripts/evaluate.py --predictions-dir outputs/predictions
    python scripts/evaluate.py --predictions-dir outputs/predictions --csv outputs/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from ovtas.metrics import compute_all_metrics
from ovtas.utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory of per-video .npz files written by run_pipeline.py.",
    )
    parser.add_argument(
        "--csv", default=None, help="Optional path to also write results as CSV."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(f for f in os.listdir(args.predictions_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No .npz prediction files found in {args.predictions_dir}")

    rows = []
    for fname in files:
        video_id = fname[:-4]
        data = np.load(os.path.join(args.predictions_dir, fname), allow_pickle=True)
        scores = compute_all_metrics(data["predicted_labels"], data["frame_labels"])
        row = {"video_id": video_id, **scores.as_dict()}
        rows.append(row)
        logger.info(
            "%-20s Acc=%.2f Edit=%.2f F1@10=%.2f F1@25=%.2f F1@50=%.2f Avg=%.2f",
            video_id,
            row["Acc"],
            row["Edit"],
            row["F1@10"],
            row["F1@25"],
            row["F1@50"],
            row["Avg"],
        )

    metric_keys = ["Acc", "Edit", "F1@10", "F1@25", "F1@50", "Avg"]
    overall = {key: float(np.mean([r[key] for r in rows])) for key in metric_keys}
    logger.info("=" * 60)
    logger.info(
        "OVERALL (%d videos): Acc=%.2f Edit=%.2f F1@10=%.2f F1@25=%.2f F1@50=%.2f Avg=%.2f",
        len(rows),
        overall["Acc"],
        overall["Edit"],
        overall["F1@10"],
        overall["F1@25"],
        overall["F1@50"],
        overall["Avg"],
    )

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video_id"] + metric_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow({"video_id": "OVERALL", **overall})
        logger.info("Wrote CSV results to %s", args.csv)


if __name__ == "__main__":
    main()
