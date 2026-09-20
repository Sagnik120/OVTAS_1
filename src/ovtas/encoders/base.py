"""
Abstract dual-encoder interface shared by every VLM backbone.

The paper's Sec. III-B frames a VLM as a pair of towers sharing an
embedding space:

    f_img: Image -> R^C   (vision encoder)
    f_txt: Text  -> R^C   (text encoder)

Every concrete encoder (CLIP, SigLIP, OpenCLIP, a future model, or the
dependency-free :class:`~ovtas.encoders.mock_encoder.MockEncoder` used
in tests) implements this same two-method contract, so Stage 1 (FAES)
never needs to know which backbone produced the embeddings.
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np


class BaseVLMEncoder(abc.ABC):
    """Common interface for all frozen Vision-Language Model encoders.

    Subclasses must implement :meth:`encode_images` and
    :meth:`encode_text`; both are expected to return **row-wise
    L2-normalized** ``float64`` arrays of shape ``(num_items,
    embed_dim)`` sharing the same ``embed_dim``, so that a plain dot
    product between them is a cosine similarity (see
    ``ovtas.stage1_faes.similarity.compute_faes``).
    """

    #: Human-readable name, e.g. "clip-vit-b32". Used in logs/configs.
    name: str = "base_vlm_encoder"

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Shared embedding dimensionality ``C`` of both towers."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode_images(self, images: Sequence) -> np.ndarray:
        """Encode a batch of frames (e.g. PIL Images or arrays).

        Returns
        -------
        np.ndarray
            Shape ``(len(images), embed_dim)``, L2-normalized rows.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of natural-language action-label phrases.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embed_dim)``, L2-normalized rows.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r}, embed_dim={self.embed_dim})"


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Shared helper: row-wise L2 normalization for encoder outputs."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, a_min=1e-8, a_max=None)


def resolve_torch_device(device: str | None = None) -> str:
    """Resolve compute device: prefers explicit device, then CUDA, then Apple Silicon MPS, then CPU."""
    if device is not None:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
