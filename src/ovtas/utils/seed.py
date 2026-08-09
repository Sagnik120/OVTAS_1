"""Global RNG seeding for reproducibility.

OVTAS is training-free, so "reproducibility" mainly matters for:
  * the random action-order permutation used under action-set
    supervision (ovtas.stage2_smts.shuffle_action_order),
  * the Random-Uniform baseline,
  * any stochastic frame-sampling stride choices.
This helper seeds every RNG source OVTAS touches from one call.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's global RNG.

    Note: OVTAS's own modules accept an explicit
    ``np.random.Generator`` wherever randomness matters (preferred,
    see e.g. ``shuffle_action_order``), so this global seed is a
    convenience for scripts / notebooks, not something library code
    relies on internally.
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Build a fresh, isolated NumPy Generator (preferred over global state)."""
    return np.random.default_rng(seed)
