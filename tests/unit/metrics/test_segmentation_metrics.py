"""Unit tests for segmentation evaluation metrics (Accuracy, Edit score, F1@k)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.metrics import (
    compute_all_metrics,
    edit_score,
    f1_at_k,
    frame_accuracy,
    labels_to_segments,
)


@pytest.mark.unit
def test_frame_accuracy_perfect_match():
    labels = [0, 1, 2, 1, 0]
    assert frame_accuracy(labels, labels) == pytest.approx(100.0)


@pytest.mark.unit
def test_frame_accuracy_no_match():
    pred = [0, 0, 0]
    gt = [1, 1, 1]
    assert frame_accuracy(pred, gt) == pytest.approx(0.0)


@pytest.mark.unit
def test_frame_accuracy_partial_match():
    pred = [0, 1, 1, 0]
    gt = [0, 1, 0, 0]
    assert frame_accuracy(pred, gt) == pytest.approx(75.0)


@pytest.mark.unit
def test_frame_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        frame_accuracy([0, 1], [0, 1, 2])


@pytest.mark.unit
def test_labels_to_segments_basic():
    segs = labels_to_segments([0, 0, 1, 1, 1, 2])
    assert [(s.label, s.start, s.end) for s in segs] == [(0, 0, 2), (1, 2, 5), (2, 5, 6)]


@pytest.mark.unit
def test_labels_to_segments_empty():
    assert labels_to_segments([]) == []


@pytest.mark.unit
def test_labels_to_segments_single_label():
    segs = labels_to_segments([3, 3, 3])
    assert len(segs) == 1
    assert segs[0].length == 3


@pytest.mark.unit
def test_edit_score_identical_sequences_is_100():
    labels = [0, 0, 1, 1, 2]
    assert edit_score(labels, labels) == pytest.approx(100.0)


@pytest.mark.unit
def test_edit_score_penalizes_oversegmentation():
    gt = [0, 0, 0, 1, 1, 1]
    fragmented = [0, 0, 1, 0, 1, 1]
    score_clean = edit_score(gt, gt)
    score_fragmented = edit_score(fragmented, gt)
    assert score_fragmented < score_clean


@pytest.mark.unit
def test_edit_score_totally_different_labels():
    gt = [0, 0, 1, 1]
    pred = [2, 2, 3, 3]
    score = edit_score(pred, gt)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_f1_at_k_perfect_prediction():
    labels = [0, 0, 1, 1, 1, 2, 2]
    for k in (0.10, 0.25, 0.50):
        assert f1_at_k(labels, labels, overlap=k) == pytest.approx(100.0)


@pytest.mark.unit
def test_f1_at_k_no_overlap_gives_zero():
    gt = [0, 0, 0, 0]
    pred = [1, 1, 1, 1]
    assert f1_at_k(pred, gt, overlap=0.10) == pytest.approx(0.0)


@pytest.mark.unit
def test_f1_at_k_partial_overlap_threshold_sensitivity():
    gt = [0] * 10
    pred = [0] * 6 + [1] * 4
    expected_f1_when_matched = 2 * 0.5 * 1.0 / (0.5 + 1.0) * 100.0
    assert f1_at_k(pred, gt, overlap=0.10) == pytest.approx(expected_f1_when_matched)
    assert f1_at_k(pred, gt, overlap=0.50) == pytest.approx(expected_f1_when_matched)
    assert f1_at_k(pred, gt, overlap=0.90) == pytest.approx(0.0)


@pytest.mark.unit
def test_f1_at_k_empty_sequences():
    assert f1_at_k([], [], overlap=0.10) == pytest.approx(100.0)


@pytest.mark.unit
def test_compute_all_metrics_perfect_prediction():
    labels = [0, 0, 1, 1, 2, 2, 2]
    scores = compute_all_metrics(labels, labels)
    d = scores.as_dict()
    for key in ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]:
        assert d[key] == pytest.approx(100.0)
    assert scores.average == pytest.approx(100.0)


@pytest.mark.unit
def test_compute_all_metrics_returns_bounded_values():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 4, size=50).tolist()
    gt = rng.integers(0, 4, size=50).tolist()
    scores = compute_all_metrics(pred, gt)
    for value in scores.as_dict().values():
        assert 0.0 <= value <= 100.0
