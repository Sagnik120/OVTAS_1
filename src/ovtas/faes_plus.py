"""
Stage 2 (Proposed): FAES+ (Frame-Action Embedding Similarity with Attention Refinement).

Top-level forwarding module for clean access to FAES+ components.
Implementation lives in :mod:`ovtas.stage1_faes.faes_plus`.
"""

from __future__ import annotations

from ovtas.stage1_faes.faes_plus import (
    FAESPlusConfig,
    FAESPlusOutput,
    compute_faes_plus,
    cross_attention_refinement,
    temporal_self_attention,
)

__all__ = [
    "FAESPlusConfig",
    "FAESPlusOutput",
    "compute_faes_plus",
    "temporal_self_attention",
    "cross_attention_refinement",
]
