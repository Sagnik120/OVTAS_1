"""Unit tests for Stage 2 parameter-free Cross-Attention Frame-to-Text Refinement."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage1_faes.faes_plus import cross_attention_refinement


@pytest.mark.unit
def test_cross_attention_shapes_and_stochasticity():
    rng = np.random.default_rng(99)
    T, K, D = 30, 8, 48
    X_prime = rng.normal(size=(T, D))
    A = rng.normal(size=(K, D))

    tau = 0.1
    A_cross = cross_attention_refinement(X_prime, A, tau=tau)

    assert A_cross.shape == (T, K)

    # Row stochasticity: sum over action dimension K must equal 1
    row_sums = np.sum(A_cross, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-6)

    assert np.all(A_cross >= 0.0)
    assert np.all(A_cross <= 1.0)


@pytest.mark.unit
def test_cross_attention_temperature_effect():
    """Lower temperature sharpens the distribution; higher temperature flattens it."""
    rng = np.random.default_rng(2024)
    T, K, D = 10, 5, 16
    X_prime = rng.normal(size=(T, D))
    A = rng.normal(size=(K, D))

    sharp = cross_attention_refinement(X_prime, A, tau=0.01)
    diffuse = cross_attention_refinement(X_prime, A, tau=10.0)

    # Maximum probability per row should be higher for sharp than diffuse
    assert np.mean(np.max(sharp, axis=1)) > np.mean(np.max(diffuse, axis=1))


@pytest.mark.unit
def test_cross_attention_invalid_tau_raises():
    X = np.ones((5, 8))
    A = np.ones((3, 8))
    with pytest.raises(ValueError, match="Temperature tau must be positive"):
        cross_attention_refinement(X, A, tau=0.0)
    with pytest.raises(ValueError, match="Temperature tau must be positive"):
        cross_attention_refinement(X, A, tau=-0.5)
