"""
OVTAS - Open-Vocabulary Zero-Shot Temporal Action Segmentation
================================================================

A training-free, zero-shot, open-vocabulary implementation of the
Temporal Action Segmentation (TAS) pipeline described in:

    "Exploring Vision-Language Models for Open-Vocabulary Zero-Shot
    Action Segmentation" (OVTAS), Unmesh et al., 2026.

The pipeline has two stages:

    Stage 1 - FAES (Frame-Action Embedding Similarity):
        Embeds video frames and candidate action-label phrases with a
        frozen Vision-Language Model (VLM) and computes a cosine
        similarity matrix S in R^(T x N).

    Stage 2 - SMTS (Similarity-Matrix driven Temporal Segmentation):
        Decodes S into a temporally-consistent per-frame label
        sequence using an entropy-regularized Optimal Transport (OT)
        solver (Sinkhorn-Knopp, log-stabilized) with a temporal prior,
        following the ASOT decoder of Xu & Gould (2024).

This package is intentionally *modular*: every major component
(VLM encoders, OT solvers, datasets, metrics, baselines) is behind a
small registry so new variants can be dropped in without touching the
rest of the codebase. See ``ovtas.registry`` for details.
"""

from ovtas.config import OVTASConfig
from ovtas.pipeline import OVTASPipeline, OVTASResult
from ovtas.stage1_faes.faes_plus import FAESPlusConfig, FAESPlusOutput, compute_faes_plus
from ovtas.stage1_faes.similarity import FAESOutput, compute_faes
from ovtas.stage2_smts.asot_decoder import ASOTConfig, SMTSOutput, decode_asot
from ovtas.version import __version__

__all__ = [
    "__version__",
    "OVTASConfig",
    "OVTASPipeline",
    "OVTASResult",
    "FAESOutput",
    "FAESPlusConfig",
    "FAESPlusOutput",
    "compute_faes",
    "compute_faes_plus",
    "ASOTConfig",
    "SMTSOutput",
    "decode_asot",
]
