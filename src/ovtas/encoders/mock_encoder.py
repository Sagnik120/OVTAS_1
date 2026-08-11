"""
Dependency-free mock VLM encoder.

Real CLIP / SigLIP backbones need PyTorch, HuggingFace weights, and a
network connection to a model hub -- none of which are guaranteed to
be available in a CI runner or on a low-resource machine. This encoder
implements the exact same :class:`~ovtas.encoders.base.BaseVLMEncoder`
interface using nothing but NumPy, by mapping each image/text to a
deterministic, seeded pseudo-random unit vector derived from a hash of
its content.

It is **not** a real VLM and produces semantically meaningless
embeddings -- it exists purely so that Stage 1 (FAES), Stage 2 (SMTS),
and the end-to-end :class:`~ovtas.pipeline.OVTASPipeline` can be
exercised, tested, and demonstrated without any heavy dependency or
internet access. Swap in ``ClipEncoder`` or ``SiglipEncoder`` (see
``clip_encoder.py`` / ``siglip_encoder.py``) for real results.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ovtas.encoders.base import BaseVLMEncoder, _l2_normalize_rows
from ovtas.encoders.registry import ENCODERS


def _seed_from_bytes(data: bytes) -> int:
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@ENCODERS.register("mock")
class MockEncoder(BaseVLMEncoder):
    """Deterministic, hash-seeded pseudo-random encoder for testing."""

    name = "mock"

    def __init__(self, embed_dim: int = 32):
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _embed_one(self, key: bytes) -> np.ndarray:
        rng = np.random.default_rng(_seed_from_bytes(key))
        return rng.normal(size=self._embed_dim)

    def encode_images(self, images: Sequence) -> np.ndarray:
        vectors = []
        for idx, image in enumerate(images):
            # Frames are hashed by their array bytes when possible
            # (so visually-identical frames get identical embeddings,
            # useful for constructing synthetic test fixtures),
            # falling back to the object's repr + index.
            arr = np.asarray(image)
            key = arr.tobytes() if arr.size else repr(image).encode()
            vectors.append(self._embed_one(b"image:" + key))
        embeddings = np.stack(vectors, axis=0) if vectors else np.zeros((0, self._embed_dim))
        return _l2_normalize_rows(embeddings)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self._embed_one(("text:" + t).encode("utf-8")) for t in texts]
        embeddings = np.stack(vectors, axis=0) if vectors else np.zeros((0, self._embed_dim))
        return _l2_normalize_rows(embeddings)
