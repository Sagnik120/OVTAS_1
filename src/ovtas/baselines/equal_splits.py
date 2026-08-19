"""Equal-Splits family of training-free baselines (Sec. IV-A.3).

All three bin frames into ``K`` contiguous, (near-)equal-length chunks
using edges ``e_k = floor(k * T / K)`` and then differ only in how a
single label is chosen per bin:

    (2) ES-Mean: argmax of the bin's *mean* similarity score.
    (3) ES-Vote: modal per-frame argmax winner within the bin
        (ties broken by the larger ES-Mean score).
    (4) ES-NRP:  a dynamic program over bins that adds a penalty for
        repeating the same label in adjacent bins, which avoids the
        degenerate "always predict the globally-best class" collapse
        that ES-Mean/ES-Vote can fall into on some videos.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ovtas.baselines.registry import BASELINES


def bin_edges(num_frames: int, num_bins: int) -> List[Tuple[int, int]]:
    """Partition ``[0, num_frames)`` into ``num_bins`` contiguous bins
    using ``e_k = floor(k * T / K)``, so bin sizes differ by at most
    one frame.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    edges = [int(np.floor(k * num_frames / num_bins)) for k in range(num_bins + 1)]
    return list(zip(edges[:-1], edges[1:]))


class _EqualSplitsBase:
    """Shared bin-count resolution logic for the Equal-Splits family."""

    def __init__(self, num_bins: Optional[int] = None):
        # Default: one bin per candidate action label -- a reasonable
        # training-free choice absent any other prior on segment count.
        self.num_bins = num_bins

    def _resolve_num_bins(self, num_frames: int, num_actions: int) -> int:
        num_bins = self.num_bins or num_actions
        return max(1, min(num_bins, num_frames))


@BASELINES.register("es_mean")
class EqualSplitsMeanBaseline(_EqualSplitsBase):
    """(2) ES-Mean: per-bin argmax of the mean similarity score."""

    def predict(self, similarity: np.ndarray) -> np.ndarray:
        similarity = np.asarray(similarity)
        T, N = similarity.shape
        num_bins = self._resolve_num_bins(T, N)
        bins = bin_edges(T, num_bins)

        labels = np.zeros(T, dtype=np.int64)
        for start, end in bins:
            if end <= start:
                continue
            mean_scores = similarity[start:end].mean(axis=0)  # (N,)
            labels[start:end] = int(np.argmax(mean_scores))
        return labels


@BASELINES.register("es_vote")
class EqualSplitsVoteBaseline(_EqualSplitsBase):
    """(3) ES-Vote: modal per-frame winner within each bin.

    Ties are broken by the larger ES-Mean score, matching the paper
    ("Ties, if any, are broken by larger s^mean_{k,c}").
    """

    def predict(self, similarity: np.ndarray) -> np.ndarray:
        similarity = np.asarray(similarity)
        T, N = similarity.shape
        num_bins = self._resolve_num_bins(T, N)
        bins = bin_edges(T, num_bins)

        per_frame_winner = np.argmax(similarity, axis=1)  # (T,)
        labels = np.zeros(T, dtype=np.int64)
        for start, end in bins:
            if end <= start:
                continue
            winners = per_frame_winner[start:end]
            values, counts = np.unique(winners, return_counts=True)
            max_count = counts.max()
            tied = values[counts == max_count]
            if len(tied) == 1:
                label = int(tied[0])
            else:
                mean_scores = similarity[start:end].mean(axis=0)
                label = int(tied[np.argmax(mean_scores[tied])])
            labels[start:end] = label
        return labels


@BASELINES.register("es_nrp")
class EqualSplitsNRPBaseline(_EqualSplitsBase):
    """(4) ES-Non-Repetition-Penalty: DP discouraging label repeats in
    adjacent bins.

        yhat_{1:K} = argmax_{y_1..y_K} sum_k s_{k,y_k}
                        - lambda * sum_k 1[y_k == y_{k-1}]

    with per-bin scores ``s_{k,c}`` taken as the ES-Mean similarity
    score, matching the paper's stated choice. Solved exactly with a
    Viterbi-style dynamic program (no greedy approximation).
    """

    def __init__(self, num_bins: Optional[int] = None, penalty: float = 1.0):
        super().__init__(num_bins=num_bins)
        self.penalty = penalty

    def predict(self, similarity: np.ndarray) -> np.ndarray:
        similarity = np.asarray(similarity)
        T, N = similarity.shape
        num_bins = self._resolve_num_bins(T, N)
        bins = bin_edges(T, num_bins)
        K = len(bins)

        bin_scores = np.zeros((K, N), dtype=np.float64)
        for k, (start, end) in enumerate(bins):
            if end > start:
                bin_scores[k] = similarity[start:end].mean(axis=0)

        bin_labels = self._viterbi_non_repetition(bin_scores, self.penalty)

        labels = np.zeros(T, dtype=np.int64)
        for k, (start, end) in enumerate(bins):
            labels[start:end] = bin_labels[k]
        return labels

    @staticmethod
    def _viterbi_non_repetition(bin_scores: np.ndarray, penalty: float) -> np.ndarray:
        """Exact DP solving the non-repetition objective.

        ``dp[k, c]``   = best cumulative score for bins ``0..k`` with
                         bin ``k`` assigned class ``c``.
        ``back[k, c]`` = the class chosen for bin ``k-1`` on that
                         optimal path (for backtracking).
        """
        K, N = bin_scores.shape
        if K == 0:
            return np.zeros(0, dtype=np.int64)

        dp = np.full((K, N), -np.inf, dtype=np.float64)
        back = np.zeros((K, N), dtype=np.int64)
        dp[0] = bin_scores[0]

        for k in range(1, K):
            for c in range(N):
                trans = dp[k - 1] - penalty * (np.arange(N) == c)
                best_prev = int(np.argmax(trans))
                dp[k, c] = trans[best_prev] + bin_scores[k, c]
                back[k, c] = best_prev

        labels = np.zeros(K, dtype=np.int64)
        labels[K - 1] = int(np.argmax(dp[K - 1]))
        for k in range(K - 1, 0, -1):
            labels[k - 1] = back[k, labels[k]]
        return labels
