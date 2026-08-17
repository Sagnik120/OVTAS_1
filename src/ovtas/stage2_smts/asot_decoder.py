"""
Stage 2: Similarity-Matrix driven Temporal Segmentation (SMTS).

Implements the "Action Segmentation Optimal Transport" (ASOT) decoder
of Xu & Gould (2024), as adopted by the paper (Sec. III-C, III-E):

    Visual cost:      C = 1 - S                     in R^{T x N}
    Temporal prior:   R_ij = | i/T - j/N |           in R^{T x N}
    Coupling:         Pi* = argmin_{Pi in U(u,v)} <Pi, C + rho*R> - eps*H(Pi)
                       u = (1/T) * 1_T,  v = (1/N) * 1_N
    Decoding:         yhat_t = argmax_j Pi*_{t,j}

``rho`` trades off the visual-similarity cost against the diagonal
temporal prior that encourages a (roughly) monotone frame-to-action
alignment; ``eps`` is the entropic regularization strength solved by
:func:`ovtas.stage2_smts.sinkhorn.log_sinkhorn`.

Action-set supervision (Sec. III-D / III-E.1): OVTAS assumes only the
*set* of candidate actions is known, not their order, so the column
ordering of ``A`` (and hence of ``R``) used to build the temporal
prior is randomized rather than taken from any ground-truth ordering.
:func:`shuffle_action_order` implements that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ovtas.stage2_smts.sinkhorn import SinkhornResult, log_sinkhorn


@dataclass(frozen=True)
class ASOTConfig:
    """Hyperparameters for the ASOT decoder.

    The paper fixes these values via a small grid search and reuses
    them across all experiments (Sec. IV-A.4):

        epsilon = 0.07   (entropic regularization strength)
        rho     = 0.5    (temporal-prior weight; paper calls it alpha)
        sinkhorn_iters, tol: solver convergence controls (not stated
            explicitly in the paper text; sane, generous defaults).

    Notes
    -----
    The paper also reports three further values from the *unbalanced*
    ASOT formulation of Xu & Gould (2024) -- ``r = 0.04``,
    ``lambda_frames = 0.11``, ``lambda_actions = 0.01`` -- but states
    that its own grid search found the **balanced** formulation (used
    here, and matching Eq. 1 of the paper) to perform best. Those three
    unbalanced-only knobs are therefore not needed for this
    implementation; they are kept as optional/inert fields so the
    config still round-trips if you later add an unbalanced solver.
    """

    epsilon: float = 0.07
    rho: float = 0.5
    sinkhorn_iters: int = 100
    sinkhorn_tol: float = 1e-6
    # Unbalanced-OT-only knobs, inert under the balanced formulation
    # used here; kept for documentation / future extension parity with
    # the original ASOT paper.
    unbalanced_r: float = 0.04
    unbalanced_lambda_frames: float = 0.11
    unbalanced_lambda_actions: float = 0.01


@dataclass
class SMTSOutput:
    """Container for Stage 2 outputs."""

    coupling: np.ndarray  # Pi* in R^{T x N}
    labels: np.ndarray  # yhat in {0, ..., N-1}^T (per-frame argmax action index)
    cost: np.ndarray  # C in R^{T x N}
    temporal_prior: np.ndarray  # R in R^{T x N}
    sinkhorn: SinkhornResult


def temporal_prior(num_frames: int, num_actions: int) -> np.ndarray:
    """Build the diagonal temporal prior ``R_ij = |i/T - j/N|``.

    This is a soft "band matrix" that is smallest near the diagonal
    (frame index proportionally close to action index) and largest at
    the corners, gently encouraging a monotone frame -> action
    alignment without hard-coding it.
    """
    if num_frames <= 0 or num_actions <= 0:
        raise ValueError("num_frames and num_actions must be positive.")
    i = np.arange(num_frames, dtype=np.float64)[:, None] / num_frames
    j = np.arange(num_actions, dtype=np.float64)[None, :] / num_actions
    return np.abs(i - j)


def visual_cost(similarity: np.ndarray) -> np.ndarray:
    """Visual cost ``C = 1 - S`` from the FAES similarity matrix."""
    return 1.0 - np.asarray(similarity, dtype=np.float64)


def shuffle_action_order(
    num_actions: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Return a random permutation of action-column indices.

    Under action-set supervision (Sec. III-E.1) the true action order
    is unknown, so the temporal prior's column ordering is randomized
    rather than assumed to match any canonical/ground-truth order.
    Apply the returned permutation to both the action embeddings (or
    equivalently the similarity matrix's columns) before decoding, and
    invert it on the decoded label indices to recover the original
    action identities.
    """
    rng = rng or np.random.default_rng()
    return rng.permutation(num_actions)


def decode_asot(
    similarity: np.ndarray,
    config: Optional[ASOTConfig] = None,
) -> SMTSOutput:
    """Run the full ASOT decoder: cost + prior -> Sinkhorn -> argmax.

    Parameters
    ----------
    similarity:
        FAES similarity matrix ``S`` of shape ``(T, N)``.
    config:
        :class:`ASOTConfig` hyperparameters. Defaults to the paper's
        fixed values.

    Returns
    -------
    SMTSOutput
        Coupling matrix, decoded per-frame label indices, and the
        intermediate cost / prior matrices for inspection or plotting.
    """
    config = config or ASOTConfig()
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2D (T, N); got shape {similarity.shape}")

    T, N = similarity.shape
    C = visual_cost(similarity)
    R = temporal_prior(T, N)
    combined_cost = C + config.rho * R

    sinkhorn_result = log_sinkhorn(
        cost=combined_cost,
        epsilon=config.epsilon,
        num_iters=config.sinkhorn_iters,
        tol=config.sinkhorn_tol,
    )
    labels = np.argmax(sinkhorn_result.coupling, axis=1)

    return SMTSOutput(
        coupling=sinkhorn_result.coupling,
        labels=labels,
        cost=C,
        temporal_prior=R,
        sinkhorn=sinkhorn_result,
    )


def frame_argmax_labels(similarity: np.ndarray) -> np.ndarray:
    """Stage-2-ablation baseline: per-frame argmax with **no** OT.

    Mirrors the paper's "Stage2 (ablated)" row in Table VII ("we
    perform frame level predictions using maximum probability label
    for each frame's action classification probabilities"), used to
    quantify SMTS's contribution over naive per-frame classification.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    return np.argmax(similarity, axis=1)
