"""Integration tests for 3-stage OVTAS+ pipeline (Encoder -> FAES+ -> SMTS)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.data import SyntheticTASDataset
from ovtas.encoders import MockEncoder
from ovtas.metrics import compute_all_metrics
from ovtas.pipeline import OVTASPipeline, OVTASResult
from ovtas.stage1_faes import FAESPlusConfig


@pytest.mark.integration
def test_pipeline_with_faes_plus_end_to_end():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=30, frame_size=(4, 4), seed=0)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    faes_plus_cfg = FAESPlusConfig(enabled=True, beta=0.5, tau=0.1)
    pipeline = OVTASPipeline(
        encoder=MockEncoder(embed_dim=16),
        faes_plus_config=faes_plus_cfg,
        seed=0,
    )
    result = pipeline.run(frames, ann.label_names, video_id=video_id)

    assert isinstance(result, OVTASResult)
    assert result.predicted_labels.shape == (30,)
    assert len(result.predicted_label_names) == 30
    assert result.faes_plus is not None
    assert result.faes_plus.raw_similarity is not None
    assert result.faes_plus.cross_attention is not None
    assert result.faes_plus.self_attention is not None

    # Check metrics computation passes on predictions
    scores = compute_all_metrics(result.predicted_labels, ann.frame_labels)
    assert 0.0 <= scores.accuracy <= 100.0


@pytest.mark.integration
def test_pipeline_with_faes_plus_windowed():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=25, frame_size=(4, 4), seed=1)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    faes_plus_cfg = FAESPlusConfig(enabled=True, beta=0.2, tau=0.05, temporal_window=5)
    pipeline = OVTASPipeline(
        encoder=MockEncoder(embed_dim=16),
        faes_plus_config=faes_plus_cfg,
        seed=1,
    )
    result = pipeline.run(frames, ann.label_names, video_id=video_id)

    assert isinstance(result, OVTASResult)
    assert result.faes_plus is not None
    assert result.predicted_labels.shape == (25,)
