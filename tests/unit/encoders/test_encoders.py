"""Unit tests for ovtas.encoders (BaseVLMEncoder interface + MockEncoder + registry)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.encoders import ENCODERS, BaseVLMEncoder, MockEncoder


@pytest.mark.unit
def test_mock_encoder_registered():
    assert "mock" in ENCODERS
    assert ENCODERS.get("mock") is MockEncoder


@pytest.mark.unit
def test_mock_encoder_is_a_base_vlm_encoder():
    enc = MockEncoder(embed_dim=16)
    assert isinstance(enc, BaseVLMEncoder)
    assert enc.embed_dim == 16


@pytest.mark.unit
def test_mock_encoder_text_embeddings_are_unit_norm():
    enc = MockEncoder(embed_dim=8)
    texts = ["pour coffee", "add sugar", "background"]
    embs = enc.encode_text(texts)
    assert embs.shape == (3, 8)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-8)


@pytest.mark.unit
def test_mock_encoder_image_embeddings_are_unit_norm():
    enc = MockEncoder(embed_dim=8)
    images = [np.zeros((4, 4, 3)), np.ones((4, 4, 3))]
    embs = enc.encode_images(images)
    assert embs.shape == (2, 8)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-8)


@pytest.mark.unit
def test_mock_encoder_is_deterministic_given_same_content():
    enc = MockEncoder(embed_dim=8)
    a = enc.encode_text(["pour coffee"])
    b = enc.encode_text(["pour coffee"])
    np.testing.assert_allclose(a, b)


@pytest.mark.unit
def test_mock_encoder_different_text_gives_different_embedding():
    enc = MockEncoder(embed_dim=8)
    a = enc.encode_text(["pour coffee"])[0]
    b = enc.encode_text(["stir milk"])[0]
    assert not np.allclose(a, b)


@pytest.mark.unit
def test_mock_encoder_identical_images_give_identical_embeddings():
    enc = MockEncoder(embed_dim=8)
    frame = np.arange(48).reshape(4, 4, 3).astype(np.float32)
    a = enc.encode_images([frame])[0]
    b = enc.encode_images([frame.copy()])[0]
    np.testing.assert_allclose(a, b)


@pytest.mark.unit
def test_mock_encoder_empty_batch_returns_empty_array():
    enc = MockEncoder(embed_dim=8)
    assert enc.encode_text([]).shape == (0, 8)
    assert enc.encode_images([]).shape == (0, 8)


@pytest.mark.unit
def test_build_via_registry():
    enc = ENCODERS.build("mock", embed_dim=24)
    assert isinstance(enc, MockEncoder)
    assert enc.embed_dim == 24
