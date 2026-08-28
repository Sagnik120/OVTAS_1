"""Shared pytest fixtures for the OVTAS test suite."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.encoders import MockEncoder
from ovtas.stage1_faes import FAESPlusConfig
from ovtas.stage2_smts import ASOTConfig


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator for repeatable tests."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Default dependency-free mock encoder with embed_dim=16."""
    return MockEncoder(embed_dim=16)


@pytest.fixture
def sample_embeddings(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic (frames, actions) unit-norm embeddings (T=20, K=5, D=32)."""
    T, K, D = 20, 5, 32
    raw_frames = rng.normal(size=(T, D))
    raw_actions = rng.normal(size=(K, D))

    frames = raw_frames / np.linalg.norm(raw_frames, axis=1, keepdims=True)
    actions = raw_actions / np.linalg.norm(raw_actions, axis=1, keepdims=True)
    return frames, actions


@pytest.fixture
def default_asot_config() -> ASOTConfig:
    """Standard paper ASOT configuration (balanced optimal transport)."""
    return ASOTConfig(epsilon=0.07, rho=0.5, sinkhorn_iters=100, sinkhorn_tol=1e-6)


@pytest.fixture
def default_faes_plus_config() -> FAESPlusConfig:
    """Standard OVTAS+ FAES+ configuration."""
    return FAESPlusConfig(enabled=True, tau=0.1, beta=0.5, normalize_refined=True)
