"""Unit tests for Stage 3 log-stabilized Sinkhorn Optimal Transport solver."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage2_smts.sinkhorn import SinkhornResult, log_sinkhorn


@pytest.mark.unit
def test_sinkhorn_converges_on_random_cost():
    rng = np.random.default_rng(0)
    cost = rng.uniform(0, 1, size=(20, 5))
    result = log_sinkhorn(cost, epsilon=0.1, num_iters=200, tol=1e-8)
    assert result.converged
    assert result.final_marginal_error < 1e-6


@pytest.mark.unit
def test_sinkhorn_row_marginals_match_target():
    rng = np.random.default_rng(1)
    T, N = 15, 6
    cost = rng.uniform(0, 1, size=(T, N))
    result = log_sinkhorn(cost, epsilon=0.05, num_iters=300)
    row_sums = result.coupling.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.full(T, 1.0 / T), atol=1e-5)


@pytest.mark.unit
def test_sinkhorn_col_marginals_match_target():
    rng = np.random.default_rng(2)
    T, N = 15, 6
    cost = rng.uniform(0, 1, size=(T, N))
    result = log_sinkhorn(cost, epsilon=0.05, num_iters=300)
    col_sums = result.coupling.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.full(N, 1.0 / N), atol=1e-5)


@pytest.mark.unit
def test_sinkhorn_coupling_is_nonnegative():
    rng = np.random.default_rng(3)
    cost = rng.uniform(-1, 1, size=(10, 4))
    result = log_sinkhorn(cost, epsilon=0.2)
    assert np.all(result.coupling >= 0)


@pytest.mark.unit
def test_sinkhorn_prefers_low_cost_entries():
    cost = np.array(
        [
            [0.0, 5.0, 5.0],
            [5.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ]
    )
    result = log_sinkhorn(cost, epsilon=0.02, num_iters=500, tol=1e-10)
    assert np.argmax(result.coupling, axis=1).tolist() == [0, 1, 2]


@pytest.mark.unit
def test_sinkhorn_rejects_bad_shapes():
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((3, 3)), row_marginal=np.zeros(2))
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((3,)))  # not 2D
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((3, 3)), epsilon=0)
