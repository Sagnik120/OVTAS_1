"""
Stage 2 (Proposed): FAES+ (Frame-Action Embedding Similarity with Attention Refinement).

As specified in OVTAS+ (OVTAS_1.png):
    1. Multi-Modal Encoding (Stage 1):
       X in R^{T x D} : row-wise L2-normalized frame embeddings.
       A in R^{K x D} : row-wise L2-normalized action-text embeddings.

    2. FAES+ Attention-Refined Similarity (Stage 2 - parameter-free, fully zero-shot):
       (a) Temporal Self-Attention:
           Smooths noisy per-frame embeddings using temporally nearby frames.
           A_self = softmax(X X^T / sqrt(D)) in R^{T x T}
           X' = A_self X                     in R^{T x D}

       (b) Cross-Attention Frame <-> Text Refinement:
           A_cross = softmax(X' A^T / tau)   in R^{T x K}
           tau: tunable scalar temperature (tau > 0).

       (c) Similarity Fusion Layer:
           S = X A^T                         in R^{T x K}  (raw cosine similarity)
           S' = S + beta * A_cross           in R^{T x K}  (refined similarity matrix)
           beta: tunable scalar residual weight.

T = number of frames, K = number of candidate action labels,
D = embedding dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ovtas.stage1_faes.similarity import FAESOutput, l2_normalize, softmax


@dataclass(frozen=True)
class FAESPlusConfig:
    """Hyperparameters for FAES+ attention refinement.

    Attributes
    ----------
    enabled:
        Whether FAES+ refinement is active. If False, the pipeline falls back
        to standard raw FAES (S = X A^T).
    tau:
        Cross-attention temperature hyperparameter (tau > 0). Smaller values
        sharpen frame-to-text assignment; larger values produce smoother distributions.
    beta:
        Residual fusion weight hyperparameter (beta >= 0). Scales the cross-attention
        refined similarity before adding to raw similarity: S' = S + beta * A_cross.
    normalize_refined:
        If True, re-normalizes X' to unit L2 norm after temporal self-attention,
        preserving the hyperspherical geometry before computing cross-attention.
    temporal_window:
        Optional local temporal context window radius (in frames). If provided,
        frames only attend to [t - W, t + W], masking out distant frames.
        If None (default), full dense temporal self-attention is used.
    """

    enabled: bool = True
    tau: float = 0.1
    beta: float = 0.5
    normalize_refined: bool = True
    temporal_window: Optional[int] = None


@dataclass
class FAESPlusOutput(FAESOutput):
    """Container for FAES+ outputs, subclassing FAESOutput for drop-in compatibility.

    Attributes
    ----------
    similarity:
        S' in R^{T x K}, refined similarity matrix (S + beta * A_cross).
    probabilities:
        P' in R^{T x K}, row-wise softmax of refined similarity S'.
    raw_similarity:
        S in R^{T x K}, original raw cosine similarity (X A^T).
    cross_attention:
        A_cross in R^{T x K}, cross-attention frame-to-action refinement matrix.
    self_attention:
        A_self in R^{T x T}, temporal self-attention affinity matrix.
    refined_frames:
        X' in R^{T x D}, temporally smoothed frame embeddings.
    """

    raw_similarity: Optional[np.ndarray] = None
    cross_attention: Optional[np.ndarray] = None
    self_attention: Optional[np.ndarray] = None
    refined_frames: Optional[np.ndarray] = None


def temporal_self_attention(
    frame_embeddings: np.ndarray,
    normalize: bool = True,
    temporal_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform parameter-free temporal self-attention over frame embeddings.

    Computes:
        A_self = softmax(X X^T / sqrt(D), axis=-1)
        X' = A_self X

    Parameters
    ----------
    frame_embeddings:
        Array of shape ``(T, D)`` with frame embeddings.
    normalize:
        If True, L2-normalize ``X'`` row-wise after attention.
    temporal_window:
        Optional window radius for local attention. When set, positions with
        ``|t - s| > temporal_window`` are masked with -inf before softmax.

    Returns
    -------
    refined_frames:
        Array of shape ``(T, D)`` containing smoothed embeddings X'.
    self_attention_weights:
        Array of shape ``(T, T)`` containing attention weights A_self.
    """
    X = np.asarray(frame_embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D frame embeddings (T, D); got shape {X.shape}.")

    T, D = X.shape
    if T == 0:
        return np.empty((0, D), dtype=X.dtype), np.empty((0, 0), dtype=X.dtype)

    # Scaled dot-product temporal affinity: (T, T)
    scale = np.sqrt(max(D, 1))
    logits = (X @ X.T) / scale

    # Optional local window masking
    if temporal_window is not None and temporal_window >= 0:
        time_indices = np.arange(T)
        dist = np.abs(time_indices[:, None] - time_indices[None, :])
        mask = dist > temporal_window
        logits = np.where(mask, -1e9, logits)

    # Numerically stable row-wise softmax: A_self in R^{T x T}
    A_self = softmax(logits, axis=-1)

    # Aggregate: X' = A_self @ X in R^{T x D}
    X_prime = A_self @ X

    if normalize:
        X_prime = l2_normalize(X_prime, axis=-1)

    return X_prime, A_self


def cross_attention_refinement(
    refined_frames: np.ndarray,
    action_embeddings: np.ndarray,
    tau: float = 0.1,
) -> np.ndarray:
    """Perform parameter-free cross-attention between refined frames and action labels.

    Computes:
        A_cross = softmax(X' A^T / tau, axis=-1) in R^{T x K}

    Parameters
    ----------
    refined_frames:
        Array of shape ``(T, D)`` containing frame embeddings X'.
    action_embeddings:
        Array of shape ``(K, D)`` containing action text embeddings A.
    tau:
        Temperature parameter (tau > 0).

    Returns
    -------
    cross_attention_weights:
        Array of shape ``(T, K)`` with row-stochastic attention probabilities.
    """
    if tau <= 0.0:
        raise ValueError(f"Temperature tau must be positive; got {tau}.")

    X_prime = np.asarray(refined_frames, dtype=np.float64)
    A = np.asarray(action_embeddings, dtype=np.float64)

    if X_prime.ndim != 2 or A.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays (T, D) and (K, D); got shapes {X_prime.shape} and {A.shape}."
        )
    if X_prime.shape[1] != A.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: X' has dim {X_prime.shape[1]}, A has dim {A.shape[1]}."
        )

    # Scaled frame-action affinity: (T, K)
    affinity = (X_prime @ A.T) / tau

    # Numerically stable softmax over action classes (axis=-1)
    A_cross = softmax(affinity, axis=-1)
    return A_cross


def compute_faes_plus(
    frame_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    config: Optional[FAESPlusConfig] = None,
    already_normalized: bool = False,
) -> FAESPlusOutput:
    """Compute the FAES+ attention-refined similarity matrix S' = S + beta * A_cross.

    Parameters
    ----------
    frame_embeddings:
        Array of shape ``(T, D)`` — one embedding per video frame (X).
    action_embeddings:
        Array of shape ``(K, D)`` — one embedding per candidate action text (A).
    config:
        Hyperparameters for FAES+ (tau, beta, normalization, window).
        If None, default FAESPlusConfig() is used.
    already_normalized:
        If True, skip initial L2-normalization of inputs.

    Returns
    -------
    FAESPlusOutput
        Holds refined similarity S', raw similarity S, cross-attention A_cross,
        temporal self-attention A_self, and refined frames X'.
    """
    if config is None:
        config = FAESPlusConfig()

    X = np.asarray(frame_embeddings, dtype=np.float64)
    A = np.asarray(action_embeddings, dtype=np.float64)

    if X.ndim != 2 or A.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays (T, D) and (K, D); got shapes {X.shape} and {A.shape}."
        )
    if X.shape[1] != A.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: frames {X.shape[1]} vs actions {A.shape[1]}."
        )

    if not already_normalized:
        X = l2_normalize(X, axis=-1)
        A = l2_normalize(A, axis=-1)

    # 1. Raw cosine similarity: S = X A^T in R^{T x K}
    raw_similarity = X @ A.T

    # If disabled, return raw similarity in FAESPlusOutput container
    if not config.enabled:
        probabilities = softmax(raw_similarity, axis=-1)
        return FAESPlusOutput(
            similarity=raw_similarity,
            probabilities=probabilities,
            raw_similarity=raw_similarity,
            cross_attention=None,
            self_attention=None,
            refined_frames=X,
        )

    # 2. Stage 2 (FAES+) - Temporal Self-Attention: X' = A_self X
    X_prime, A_self = temporal_self_attention(
        X,
        normalize=config.normalize_refined,
        temporal_window=config.temporal_window,
    )

    # 3. Stage 2 (FAES+) - Cross-Attention: A_cross = softmax(X' A^T / tau)
    A_cross = cross_attention_refinement(X_prime, A, tau=config.tau)

    # 4. Stage 2 (FAES+) - Similarity Fusion Layer: S' = S + beta * A_cross
    refined_similarity = raw_similarity + config.beta * A_cross

    probabilities = softmax(refined_similarity, axis=-1)

    return FAESPlusOutput(
        similarity=refined_similarity,
        probabilities=probabilities,
        raw_similarity=raw_similarity,
        cross_attention=A_cross,
        self_attention=A_self,
        refined_frames=X_prime,
    )
