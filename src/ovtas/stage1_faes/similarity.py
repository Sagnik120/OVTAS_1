"""
Stage 1: Frame-Action Embedding Similarity (FAES).

Paper notation (Sec. III-B, III-D):
    X in R^{T x C}  : row-wise L2-normalized frame embeddings.
    A in R^{N x C}  : row-wise L2-normalized action-label embeddings.
    S = X A^T in R^{T x N} : cosine similarity matrix.
    P = softmax_N(S)       : row-wise frame classification probabilities.

T = number of frames, N = number of candidate action labels,
C = embedding dimension shared by the vision and text towers of the
VLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization, matching ``||x_t||_2 = 1`` in the paper.

    The ablation in Table IV shows removing this normalization costs
    roughly 19 points of average score on GTEA, so it is treated as a
    first-class, always-on step rather than an implicit detail.
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.clip(norm, a_min=eps, a_max=None)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x_shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass
class FAESOutput:
    """Container for Stage 1 outputs.

    Attributes
    ----------
    similarity: S in R^{T x N}, cosine similarity (frames x actions).
    probabilities: P in R^{T x N}, row-wise softmax of ``similarity``.
    """

    similarity: np.ndarray
    probabilities: np.ndarray

    @property
    def num_frames(self) -> int:
        return self.similarity.shape[0]

    @property
    def num_actions(self) -> int:
        return self.similarity.shape[1]


def compute_faes(
    frame_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    already_normalized: bool = False,
) -> FAESOutput:
    """Compute the FAES similarity matrix ``S = X A^T``.

    Parameters
    ----------
    frame_embeddings:
        Array of shape ``(T, C)`` — one embedding per video frame.
    action_embeddings:
        Array of shape ``(N, C)`` — one embedding per candidate
        action-label phrase.
    already_normalized:
        If True, skip the (redundant) L2 normalization because the
        encoder already guarantees unit-norm rows.

    Returns
    -------
    FAESOutput
        Holds both the raw cosine-similarity matrix ``S`` and the
        row-wise softmax probabilities ``P``.
    """
    frame_embeddings = np.asarray(frame_embeddings, dtype=np.float64)
    action_embeddings = np.asarray(action_embeddings, dtype=np.float64)

    if frame_embeddings.ndim != 2 or action_embeddings.ndim != 2:
        raise ValueError(
            "Expected 2D arrays (T, C) and (N, C); got shapes "
            f"{frame_embeddings.shape} and {action_embeddings.shape}."
        )
    if frame_embeddings.shape[1] != action_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch between frames "
            f"({frame_embeddings.shape[1]}) and actions "
            f"({action_embeddings.shape[1]})."
        )

    if not already_normalized:
        frame_embeddings = l2_normalize(frame_embeddings, axis=1)
        action_embeddings = l2_normalize(action_embeddings, axis=1)

    similarity = frame_embeddings @ action_embeddings.T  # (T, N)
    probabilities = softmax(similarity, axis=1)
    return FAESOutput(similarity=similarity, probabilities=probabilities)
