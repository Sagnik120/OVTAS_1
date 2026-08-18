"""(1) Random-Uniform (RU) baseline.

    y_t ~ Categorical(1/C, ..., 1/C),  t = 1, ..., T

Each frame is independently labeled by a uniform draw over the C
candidate action classes, ignoring the similarity matrix entirely.
This is the weakest possible training-free baseline and serves as a
sanity floor for every other method.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ovtas.baselines.registry import BASELINES


@BASELINES.register("random_uniform")
class RandomUniformBaseline:
    """Uniform-random per-frame label assignment."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def predict(self, similarity: np.ndarray) -> np.ndarray:
        similarity = np.asarray(similarity)
        num_frames, num_actions = similarity.shape
        return self.rng.integers(0, num_actions, size=num_frames)
