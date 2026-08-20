"""
Temporal Action Segmentation metrics (Sec. IV-A.2 of the paper):

    Accuracy : proportion of correctly labeled frames.
    Edit     : normalized Levenshtein distance between predicted and
               ground-truth *segment label sequences* (i.e. after
               collapsing consecutive repeats), penalizing
               over-segmentation / fragmentation.
    F1@k     : segment-level F1 score at IoU overlap threshold
               k in {10, 25, 50} (percent), the standard protocol from
               Lea et al. (2017) / Farha & Gall (2019) used across the
               TAS literature (and referenced by the ASOT paper this
               work builds on).

All functions operate on 1D integer label arrays and are dependency-
free (pure NumPy), so they can be unit tested and used to score any
baseline or pipeline output without extra plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #

def frame_accuracy(pred: Sequence[int], gt: Sequence[int]) -> float:
    """Fraction of frames where ``pred[t] == gt[t]``."""
    pred_arr, gt_arr = _as_equal_length_arrays(pred, gt)
    if len(gt_arr) == 0:
        return 0.0
    return float(np.mean(pred_arr == gt_arr) * 100.0)


# --------------------------------------------------------------------------- #
# Segment extraction (shared by Edit score and F1@k)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Segment:
    """A contiguous run of a single label: ``[start, end)`` (end exclusive)."""

    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def labels_to_segments(labels: Sequence[int]) -> List[Segment]:
    """Collapse a per-frame label sequence into contiguous segments.

    Example
    -------
    >>> [s.label for s in labels_to_segments([0, 0, 1, 1, 1, 0])]
    [0, 1, 0]
    """
    labels = np.asarray(labels)
    segments: List[Segment] = []
    if len(labels) == 0:
        return segments

    start = 0
    current = labels[0]
    for t in range(1, len(labels)):
        if labels[t] != current:
            segments.append(Segment(label=int(current), start=start, end=t))
            start = t
            current = labels[t]
    segments.append(Segment(label=int(current), start=start, end=len(labels)))
    return segments


# --------------------------------------------------------------------------- #
# Edit score
# --------------------------------------------------------------------------- #

def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Standard dynamic-programming Levenshtein (edit) distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # deletion
                dp[i, j - 1] + 1,  # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )
    return int(dp[n, m])


def edit_score(pred: Sequence[int], gt: Sequence[int]) -> float:
    """Normalized edit score between the *segment* label sequences.

    ``100 * (1 - lev_distance / max(len(pred_segments), len(gt_segments)))``

    Operating on collapsed segment sequences (rather than raw frames)
    is what makes this metric sensitive to over-segmentation: extra
    spurious short segments increase the sequence length and the edit
    distance even if most frames are still correctly labeled.
    """
    pred_seq = [s.label for s in labels_to_segments(pred)]
    gt_seq = [s.label for s in labels_to_segments(gt)]
    if len(pred_seq) == 0 and len(gt_seq) == 0:
        return 100.0
    max_len = max(len(pred_seq), len(gt_seq))
    if max_len == 0:
        return 100.0
    distance = _levenshtein(pred_seq, gt_seq)
    return float((1.0 - distance / max_len) * 100.0)


# --------------------------------------------------------------------------- #
# F1@k (segment overlap)
# --------------------------------------------------------------------------- #

def _segment_iou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union == 0:
        return 0.0
    return inter / union


def f1_at_k(pred: Sequence[int], gt: Sequence[int], overlap: float) -> float:
    """Segment-level F1 score at IoU threshold ``overlap`` (e.g. 0.10).

    Follows the standard greedy-matching protocol (Lea et al. 2017):
    each ground-truth segment can be matched to at most one predicted
    segment (and vice-versa); a predicted segment is a *true positive*
    if its best-IoU-matching, same-label, not-yet-matched ground-truth
    segment has IoU >= ``overlap``.
    """
    pred_segs = labels_to_segments(pred)
    gt_segs = labels_to_segments(gt)

    if len(pred_segs) == 0 and len(gt_segs) == 0:
        return 100.0
    if len(pred_segs) == 0 or len(gt_segs) == 0:
        return 0.0

    gt_matched = [False] * len(gt_segs)
    tp = 0
    for p in pred_segs:
        best_iou = -1.0
        best_idx = -1
        for gi, g in enumerate(gt_segs):
            if gt_matched[gi] or g.label != p.label:
                continue
            iou = _segment_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx >= 0 and best_iou >= overlap:
            tp += 1
            gt_matched[best_idx] = True

    fp = len(pred_segs) - tp
    fn = len(gt_segs) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1 * 100.0)


# --------------------------------------------------------------------------- #
# Convenience: compute all metrics at once
# --------------------------------------------------------------------------- #

@dataclass
class SegmentationScores:
    """All Table VI metrics for a single prediction/ground-truth pair."""

    accuracy: float
    edit: float
    f1_10: float
    f1_25: float
    f1_50: float

    @property
    def average(self) -> float:
        """Mean of F1@10, F1@25, F1@50, Edit and Accuracy (paper's "Avg")."""
        return float(
            np.mean([self.f1_10, self.f1_25, self.f1_50, self.edit, self.accuracy])
        )

    def as_dict(self) -> dict:
        return {
            "Acc": self.accuracy,
            "Edit": self.edit,
            "F1@10": self.f1_10,
            "F1@25": self.f1_25,
            "F1@50": self.f1_50,
            "Avg": self.average,
        }


def compute_all_metrics(pred: Sequence[int], gt: Sequence[int]) -> SegmentationScores:
    """Compute Accuracy, Edit, F1@10/25/50 in one call."""
    _as_equal_length_arrays(pred, gt)  # validates shapes early
    return SegmentationScores(
        accuracy=frame_accuracy(pred, gt),
        edit=edit_score(pred, gt),
        f1_10=f1_at_k(pred, gt, overlap=0.10),
        f1_25=f1_at_k(pred, gt, overlap=0.25),
        f1_50=f1_at_k(pred, gt, overlap=0.50),
    )


def _as_equal_length_arrays(
    pred: Sequence[int], gt: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape; got {pred_arr.shape} vs {gt_arr.shape}"
        )
    return pred_arr, gt_arr
