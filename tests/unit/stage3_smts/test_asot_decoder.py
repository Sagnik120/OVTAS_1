"""Unit tests for Stage 3 ASOT temporal selection and decoding."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage2_smts.asot_decoder import (
    ASOTConfig,
    decode_asot,
    frame_argmax_labels,
    shuffle_action_order,
    temporal_prior,
    visual_cost,
)


@pytest.mark.unit
def test_temporal_prior_shape_and_zero_near_diagonal():
    R = temporal_prior(num_frames=10, num_actions=5)
    assert R.shape == (10, 5)
    assert R[0, 0] < R[0, 4]
    assert R[9, 4] < R[9, 0]


@pytest.mark.unit
def test_temporal_prior_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        temporal_prior(0, 5)
    with pytest.raises(ValueError):
        temporal_prior(5, 0)


@pytest.mark.unit
def test_visual_cost_is_one_minus_similarity():
    sim = np.array([[0.2, -0.5], [1.0, 0.0]])
    cost = visual_cost(sim)
    np.testing.assert_allclose(cost, 1.0 - sim)


@pytest.mark.unit
def test_shuffle_action_order_is_a_valid_permutation():
    rng = np.random.default_rng(0)
    perm = shuffle_action_order(7, rng=rng)
    assert sorted(perm.tolist()) == list(range(7))


@pytest.mark.unit
def test_shuffle_action_order_reproducible_with_seeded_rng():
    perm_a = shuffle_action_order(10, rng=np.random.default_rng(123))
    perm_b = shuffle_action_order(10, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(perm_a, perm_b)


@pytest.mark.unit
def test_decode_asot_output_shapes():
    rng = np.random.default_rng(4)
    T, N = 30, 4
    sim = rng.uniform(-1, 1, size=(T, N))
    out = decode_asot(sim)
    assert out.coupling.shape == (T, N)
    assert out.labels.shape == (T,)
    assert out.cost.shape == (T, N)
    assert out.temporal_prior.shape == (T, N)
    assert set(out.labels.tolist()).issubset(set(range(N)))


@pytest.mark.unit
def test_decode_asot_recovers_clean_block_structure():
    T, N = 30, 3
    sim = -1.0 * np.ones((T, N))
    block = T // N
    for j in range(N):
        sim[j * block : (j + 1) * block, j] = 1.0

    out = decode_asot(sim, config=ASOTConfig(epsilon=0.03, rho=0.5))
    for j in range(N):
        segment_labels = out.labels[j * block : (j + 1) * block]
        values, counts = np.unique(segment_labels, return_counts=True)
        majority_label = values[np.argmax(counts)]
        assert majority_label == j


@pytest.mark.unit
def test_decode_asot_rejects_non_2d_similarity():
    with pytest.raises(ValueError):
        decode_asot(np.zeros((5,)))


@pytest.mark.unit
def test_frame_argmax_labels_matches_naive_argmax():
    sim = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    labels = frame_argmax_labels(sim)
    np.testing.assert_array_equal(labels, np.array([1, 0, 0]))
