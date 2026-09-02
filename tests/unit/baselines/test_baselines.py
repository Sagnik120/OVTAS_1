"""Unit tests for baseline heuristics (RU, ES-Mean, ES-Vote, ES-NRP)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.baselines import (
    BASELINES,
    EqualSplitsMeanBaseline,
    EqualSplitsNRPBaseline,
    EqualSplitsVoteBaseline,
    RandomUniformBaseline,
    bin_edges,
    run_baseline,
)


@pytest.mark.unit
def test_bin_edges_covers_full_range_contiguously():
    edges = bin_edges(10, 3)
    assert edges[0][0] == 0
    assert edges[-1][1] == 10
    for (_, end), (next_start, _) in zip(edges, edges[1:]):
        assert end == next_start


@pytest.mark.unit
def test_bin_edges_rejects_nonpositive_bins():
    with pytest.raises(ValueError):
        bin_edges(10, 0)


@pytest.mark.unit
def test_random_uniform_shape_and_range():
    sim = np.zeros((20, 4))
    baseline = RandomUniformBaseline(seed=0)
    labels = baseline.predict(sim)
    assert labels.shape == (20,)
    assert labels.min() >= 0
    assert labels.max() < 4


@pytest.mark.unit
def test_random_uniform_is_reproducible_with_seed():
    sim = np.zeros((20, 4))
    a = RandomUniformBaseline(seed=42).predict(sim)
    b = RandomUniformBaseline(seed=42).predict(sim)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_es_mean_recovers_clean_blocks():
    T, N = 12, 3
    sim = -np.ones((T, N))
    block = T // N
    for j in range(N):
        sim[j * block : (j + 1) * block, j] = 1.0

    baseline = EqualSplitsMeanBaseline(num_bins=N)
    labels = baseline.predict(sim)
    expected = np.repeat(np.arange(N), block)
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.unit
def test_es_mean_constant_within_each_bin():
    rng = np.random.default_rng(0)
    sim = rng.normal(size=(17, 5))
    baseline = EqualSplitsMeanBaseline(num_bins=4)
    labels = baseline.predict(sim)
    for start, end in bin_edges(17, 4):
        assert len(set(labels[start:end].tolist())) == 1


@pytest.mark.unit
def test_es_vote_recovers_clean_blocks():
    T, N = 12, 3
    sim = -np.ones((T, N))
    block = T // N
    for j in range(N):
        sim[j * block : (j + 1) * block, j] = 1.0

    baseline = EqualSplitsVoteBaseline(num_bins=N)
    labels = baseline.predict(sim)
    expected = np.repeat(np.arange(N), block)
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.unit
def test_es_vote_tie_break_uses_mean_score():
    sim = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )
    baseline = EqualSplitsVoteBaseline(num_bins=1)
    labels = baseline.predict(sim)
    assert set(labels.tolist()) == {0}


@pytest.mark.unit
def test_es_nrp_discourages_adjacent_repeats():
    bin_scores_like = np.array(
        [
            [1.0, 0.9],
            [1.0, 0.99],
            [1.0, 0.9],
        ]
    )
    baseline = EqualSplitsNRPBaseline(num_bins=3, penalty=0.5)
    labels = baseline.predict(bin_scores_like)
    assert not np.all(labels == 0)


@pytest.mark.unit
def test_es_nrp_zero_penalty_matches_es_mean():
    rng = np.random.default_rng(3)
    sim = rng.normal(size=(20, 4))
    nrp_labels = EqualSplitsNRPBaseline(num_bins=4, penalty=0.0).predict(sim)
    mean_labels = EqualSplitsMeanBaseline(num_bins=4).predict(sim)
    np.testing.assert_array_equal(nrp_labels, mean_labels)


@pytest.mark.unit
def test_es_nrp_output_shape():
    rng = np.random.default_rng(4)
    sim = rng.normal(size=(23, 6))
    labels = EqualSplitsNRPBaseline(num_bins=5, penalty=1.0).predict(sim)
    assert labels.shape == (23,)


@pytest.mark.unit
def test_all_four_baselines_registered():
    for name in ["random_uniform", "es_mean", "es_vote", "es_nrp"]:
        assert name in BASELINES


@pytest.mark.unit
def test_run_baseline_convenience_function():
    rng = np.random.default_rng(5)
    sim = rng.normal(size=(10, 3))
    labels = run_baseline("es_mean", sim, num_bins=3)
    assert labels.shape == (10,)
