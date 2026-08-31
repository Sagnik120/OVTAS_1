"""Unit tests for Stage 1 raw cosine similarity (FAES)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage1_faes.similarity import (
    FAESOutput,
    compute_faes,
    l2_normalize,
    softmax,
)


@pytest.mark.unit
def test_l2_normalize_unit_norm():
    x = np.random.default_rng(0).normal(size=(5, 8))
    x_norm = l2_normalize(x)
    norms = np.linalg.norm(x_norm, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-8)


@pytest.mark.unit
def test_l2_normalize_handles_zero_vector():
    x = np.zeros((1, 4))
    x_norm = l2_normalize(x)
    assert np.all(np.isfinite(x_norm))


@pytest.mark.unit
def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(1).normal(size=(4, 6))
    p = softmax(x, axis=1)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(4), atol=1e-8)


@pytest.mark.unit
def test_compute_faes_shapes_and_ranges():
    rng = np.random.default_rng(42)
    T, N, C = 10, 4, 16
    frames = rng.normal(size=(T, C))
    actions = rng.normal(size=(N, C))

    out = compute_faes(frames, actions)
    assert out.similarity.shape == (T, N)
    assert out.probabilities.shape == (T, N)
    assert out.num_frames == T
    assert out.num_actions == N
    # Cosine similarity of L2-normalized vectors is bounded in [-1, 1].
    assert out.similarity.min() >= -1.0 - 1e-6
    assert out.similarity.max() <= 1.0 + 1e-6
    # Probabilities are a valid row-wise distribution.
    np.testing.assert_allclose(out.probabilities.sum(axis=1), np.ones(T), atol=1e-8)


@pytest.mark.unit
def test_compute_faes_identical_frame_and_action_gives_similarity_one():
    C = 8
    rng = np.random.default_rng(7)
    action = rng.normal(size=(1, C))
    frames = np.vstack([action, rng.normal(size=(3, C))])
    actions = np.vstack([action, rng.normal(size=(2, C))])

    out = compute_faes(frames, actions)
    np.testing.assert_allclose(out.similarity[0, 0], 1.0, atol=1e-6)
    assert np.argmax(out.similarity[0]) == 0


@pytest.mark.unit
def test_compute_faes_dimension_mismatch_raises():
    frames = np.zeros((3, 8))
    actions = np.zeros((2, 16))
    with pytest.raises(ValueError):
        compute_faes(frames, actions)


@pytest.mark.unit
def test_compute_faes_rejects_non_2d_input():
    with pytest.raises(ValueError):
        compute_faes(np.zeros((3, 4, 5)), np.zeros((2, 5)))
