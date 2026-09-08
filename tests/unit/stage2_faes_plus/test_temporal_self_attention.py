"""Unit tests for Stage 2 parameter-free Temporal Self-Attention."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage1_faes.faes_plus import temporal_self_attention


@pytest.mark.unit
def test_temporal_self_attention_shapes_and_stochasticity():
    rng = np.random.default_rng(42)
    T, D = 25, 64
    X = rng.normal(size=(T, D))

    X_prime, A_self = temporal_self_attention(X, normalize=True)

    assert X_prime.shape == (T, D)
    assert A_self.shape == (T, T)

    # Row stochasticity: sum over key dimension must equal 1
    row_sums = np.sum(A_self, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-6)

    # Probabilities in [0, 1]
    assert np.all(A_self >= 0.0)
    assert np.all(A_self <= 1.0)

    # Output normalized rows have unit norm
    norms = np.linalg.norm(X_prime, axis=-1)
    np.testing.assert_allclose(norms, np.ones(T), atol=1e-6)


@pytest.mark.unit
def test_temporal_self_attention_smooths_noisy_embeddings():
    """Verify diagram claim: 'smooths noisy per-frame embeddings using temporally nearby frames'."""
    rng = np.random.default_rng(123)
    T, D = 50, 32

    # Clean signal: slowly varying sequence of embeddings
    base = rng.normal(size=(1, D))
    clean_frames = np.repeat(base, T, axis=0)

    # Add high-frequency Gaussian noise to each frame
    noise = rng.normal(scale=0.5, size=(T, D))
    noisy_frames = clean_frames + noise

    # Apply temporal self-attention
    refined_frames, _ = temporal_self_attention(noisy_frames, normalize=False)

    # Error to ground-truth clean frames should decrease
    err_noisy = np.linalg.norm(noisy_frames - clean_frames)
    err_refined = np.linalg.norm(refined_frames - clean_frames)

    assert err_refined < err_noisy, (
        f"Expected temporal self-attention to smooth noise; "
        f"err_refined={err_refined:.4f} >= err_noisy={err_noisy:.4f}"
    )


@pytest.mark.unit
def test_temporal_self_attention_window_masking():
    rng = np.random.default_rng(7)
    T, D = 20, 16
    X = rng.normal(size=(T, D))

    window = 2
    _, A_self = temporal_self_attention(X, normalize=True, temporal_window=window)

    for i in range(T):
        for j in range(T):
            if abs(i - j) > window:
                assert A_self[i, j] < 1e-6


@pytest.mark.unit
def test_temporal_self_attention_empty_input():
    X = np.empty((0, 32))
    X_prime, A_self = temporal_self_attention(X)
    assert X_prime.shape == (0, 32)
    assert A_self.shape == (0, 0)
