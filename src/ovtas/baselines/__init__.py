"""
Training-free, zero-shot, open-vocabulary baselines (Sec. IV-A.3).

All four baselines share the same frozen-encoder cosine-similarity
matrix ``S in R^{T x N}`` as OVTAS itself, so they can be swapped in
for a fair, apples-to-apples comparison against the full pipeline:

    (1) random_uniform - RandomUniformBaseline
    (2) es_mean         - EqualSplitsMeanBaseline
    (3) es_vote          - EqualSplitsVoteBaseline
    (4) es_nrp            - EqualSplitsNRPBaseline

New baselines register themselves against :data:`BASELINES` with
``@BASELINES.register("name")`` -- see ``random_uniform.py`` or
``equal_splits.py`` for a template.
"""

from ovtas.baselines.equal_splits import (
    EqualSplitsMeanBaseline,
    EqualSplitsNRPBaseline,
    EqualSplitsVoteBaseline,
    bin_edges,
)
from ovtas.baselines.random_uniform import RandomUniformBaseline
from ovtas.baselines.registry import BASELINES


def run_baseline(name: str, similarity, **kwargs):
    """Convenience one-liner: build + predict for a registered baseline."""
    baseline = BASELINES.build(name, **kwargs)
    return baseline.predict(similarity)


__all__ = [
    "BASELINES",
    "run_baseline",
    "bin_edges",
    "RandomUniformBaseline",
    "EqualSplitsMeanBaseline",
    "EqualSplitsVoteBaseline",
    "EqualSplitsNRPBaseline",
]
