"""Unit tests for Stage 2 Similarity Fusion Layer S' = S + beta * A_cross."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.stage1_faes.faes_plus import (
    FAESPlusConfig,
    FAESPlusOutput,
    compute_faes_plus,
)
from ovtas.stage1_faes.similarity import compute_faes


@pytest.mark.unit
def test_compute_faes_plus_exact_math():
    rng = np.random.default_rng(314)
    T, K, D = 15, 6, 32
    frames = rng.normal(size=(T, D))
    actions = rng.normal(size=(K, D))

    beta = 0.75
    tau = 0.2
    config = FAESPlusConfig(enabled=True, beta=beta, tau=tau)

    out = compute_faes_plus(frames, actions, config=config)

    assert isinstance(out, FAESPlusOutput)
    assert out.similarity.shape == (T, K)
    assert out.raw_similarity.shape == (T, K)
    assert out.cross_attention.shape == (T, K)
    assert out.self_attention.shape == (T, T)
    assert out.refined_frames.shape == (T, D)

    # Check exact mathematical relation: S' = S + beta * A_cross
    expected_refined = out.raw_similarity + beta * out.cross_attention
    np.testing.assert_allclose(out.similarity, expected_refined, atol=1e-8)


@pytest.mark.unit
def test_compute_faes_plus_regression_when_beta_zero():
    """When beta=0, S' == S (exact baseline similarity values)."""
    rng = np.random.default_rng(555)
    frames = rng.normal(size=(10, 16))
    actions = rng.normal(size=(4, 16))

    config = FAESPlusConfig(enabled=True, beta=0.0, tau=0.1)
    out_plus = compute_faes_plus(frames, actions, config=config)
    out_base = compute_faes(frames, actions)

    np.testing.assert_allclose(out_plus.similarity, out_base.similarity, atol=1e-8)


@pytest.mark.unit
def test_compute_faes_plus_when_disabled():
    """When enabled=False, returns raw cosine similarity."""
    rng = np.random.default_rng(777)
    frames = rng.normal(size=(12, 16))
    actions = rng.normal(size=(5, 16))

    config = FAESPlusConfig(enabled=False)
    out_plus = compute_faes_plus(frames, actions, config=config)
    out_base = compute_faes(frames, actions)

    np.testing.assert_allclose(out_plus.similarity, out_base.similarity, atol=1e-8)
    assert out_plus.cross_attention is None
    assert out_plus.self_attention is None
