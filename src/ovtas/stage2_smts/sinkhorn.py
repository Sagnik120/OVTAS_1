"""
Entropy-regularized Optimal Transport via the Sinkhorn-Knopp algorithm,
computed in log-space for numerical stability ("log-stabilization"),
as referenced in the paper (Sec. III-E.2):

    "In practice, Pi* is computed via Sinkhorn [26] iterations with
    log-stabilization, which scales linearly in T x N per iteration."

We solve the *balanced* entropic OT problem:

    Pi* = argmin_{Pi in U(u, v)}  <Pi, K> - eps * H(Pi)

where ``K`` is a generic cost matrix (Stage 2 combines the visual cost
and the temporal prior into K = C + rho * R before calling this
solver, see ``asot_decoder.py``), ``U(u, v)`` is the transport
polytope with row/column marginals ``u`` and ``v``, ``H`` is the
Shannon entropy of the coupling, and ``eps > 0`` is the entropic
regularization strength.

Reference: Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of
Optimal Transport." NeurIPS 2013. Knight, P. A. "The Sinkhorn-Knopp
algorithm: convergence and applications." SIAM J. Matrix Anal. 2008.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_LOG_EPS = 1e-300  # guards log(0) without materially biasing results


@dataclass
class SinkhornResult:
    """Container for the Sinkhorn solver's output."""

    coupling: np.ndarray  # Pi* in R^{T x N}
    num_iters: int
    converged: bool
    final_marginal_error: float


def log_sinkhorn(
    cost: np.ndarray,
    row_marginal: Optional[np.ndarray] = None,
    col_marginal: Optional[np.ndarray] = None,
    epsilon: float = 0.07,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> SinkhornResult:
    """Solve entropic OT in log-space (log-sum-exp stabilized updates).

    Parameters
    ----------
    cost:
        Cost matrix ``K`` of shape ``(T, N)``. Lower cost = more likely
        to be matched under the coupling.
    row_marginal:
        Target row sums ``u`` of shape ``(T,)``. Defaults to the
        uniform distribution ``u = 1/T * 1_T``, matching the paper.
    col_marginal:
        Target column sums ``v`` of shape ``(N,)``. Defaults to the
        uniform distribution ``v = 1/N * 1_N``, matching the paper.
    epsilon:
        Entropic regularization strength. The paper's fixed value is
        ``eps = 0.07``.
    num_iters:
        Maximum number of alternating row/column-scaling iterations.
    tol:
        Stop early once the row-marginal L1 error drops below ``tol``.

    Returns
    -------
    SinkhornResult
        The coupling ``Pi*`` together with convergence diagnostics.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D (T, N); got shape {cost.shape}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}")

    T, N = cost.shape
    if row_marginal is None:
        row_marginal = np.full(T, 1.0 / T, dtype=np.float64)
    if col_marginal is None:
        col_marginal = np.full(N, 1.0 / N, dtype=np.float64)

    row_marginal = np.asarray(row_marginal, dtype=np.float64)
    col_marginal = np.asarray(col_marginal, dtype=np.float64)
    if row_marginal.shape != (T,):
        raise ValueError(f"row_marginal must have shape ({T},); got {row_marginal.shape}")
    if col_marginal.shape != (N,):
        raise ValueError(f"col_marginal must have shape ({N},); got {col_marginal.shape}")

    log_u = np.log(np.clip(row_marginal, _LOG_EPS, None))
    log_v = np.log(np.clip(col_marginal, _LOG_EPS, None))

    # Log-domain kernel: log K = -cost / epsilon.
    log_K = -cost / epsilon

    # Dual (scaling) potentials in log-space.
    f = np.zeros(T, dtype=np.float64)  # log(row scaling)
    g = np.zeros(N, dtype=np.float64)  # log(col scaling)

    converged = False
    marginal_error = float("inf")
    it = 0
    for it in range(1, num_iters + 1):
        # f_t = log(u_t) - logsumexp_j (log_K[t, j] + g_j)
        f = log_u - _logsumexp(log_K + g[None, :], axis=1)
        # g_j = log(v_j) - logsumexp_t (log_K[t, j] + f_t)
        g = log_v - _logsumexp(log_K + f[:, None], axis=0)

        log_pi = f[:, None] + log_K + g[None, :]
        pi = np.exp(log_pi)
        row_sums = pi.sum(axis=1)
        marginal_error = float(np.abs(row_sums - row_marginal).sum())
        if marginal_error < tol:
            converged = True
            break

    log_pi = f[:, None] + log_K + g[None, :]
    coupling = np.exp(log_pi)

    return SinkhornResult(
        coupling=coupling,
        num_iters=it,
        converged=converged,
        final_marginal_error=marginal_error,
    )


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x)))`` along ``axis``."""
    x_max = np.max(x, axis=axis, keepdims=True)
    # Guard against an entire row/column of -inf (e.g. extreme costs).
    x_max_safe = np.where(np.isfinite(x_max), x_max, 0.0)
    summed = np.sum(np.exp(x - x_max_safe), axis=axis, keepdims=True)
    result = x_max_safe + np.log(np.clip(summed, _LOG_EPS, None))
    return np.squeeze(result, axis=axis)
