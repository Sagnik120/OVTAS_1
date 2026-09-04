"""Integration tests for baseline 2-stage OVTAS pipeline (Encoder -> FAES -> SMTS)."""

from __future__ import annotations

import numpy as np
import pytest

from ovtas.data import SyntheticTASDataset
from ovtas.encoders import MockEncoder
from ovtas.metrics import compute_all_metrics
from ovtas.pipeline import OVTASPipeline, OVTASResult
from ovtas.stage2_smts import ASOTConfig


@pytest.mark.integration
def test_pipeline_runs_end_to_end_with_mock_encoder():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=40, frame_size=(4, 4), seed=0)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    pipeline = OVTASPipeline(encoder=MockEncoder(embed_dim=16), seed=0)
    result = pipeline.run(frames, ann.label_names, video_id=video_id)

    assert isinstance(result, OVTASResult)
    assert result.predicted_labels.shape == (40,)
    assert set(result.predicted_labels.tolist()).issubset(
        set(range(len(ann.label_names)))
    )
    assert len(result.predicted_label_names) == 40


@pytest.mark.integration
def test_pipeline_action_order_is_a_valid_permutation():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=20, frame_size=(4, 4), seed=1)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    pipeline = OVTASPipeline(encoder=MockEncoder(embed_dim=8), seed=1)
    result = pipeline.run(frames, ann.label_names, video_id=video_id)

    assert sorted(result.action_order.tolist()) == list(range(len(ann.label_names)))


@pytest.mark.integration
def test_pipeline_without_shuffle_uses_identity_order():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=15, frame_size=(4, 4), seed=2)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    pipeline = OVTASPipeline(
        encoder=MockEncoder(embed_dim=8), shuffle_actions=False, seed=2
    )
    result = pipeline.run(frames, ann.label_names, video_id=video_id)
    np.testing.assert_array_equal(result.action_order, np.arange(len(ann.label_names)))


@pytest.mark.integration
def test_pipeline_builds_encoder_from_registry_name():
    pipeline = OVTASPipeline(encoder_name="mock", encoder_kwargs={"embed_dim": 4})
    assert pipeline.encoder.embed_dim == 4


@pytest.mark.integration
def test_pipeline_is_scoreable_with_metrics_module():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=50, frame_size=(4, 4), seed=3)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    pipeline = OVTASPipeline(encoder=MockEncoder(embed_dim=16), seed=3)
    result = pipeline.run(frames, ann.label_names, video_id=video_id)

    scores = compute_all_metrics(result.predicted_labels, ann.frame_labels)
    for value in scores.as_dict().values():
        assert 0.0 <= value <= 100.0


@pytest.mark.integration
def test_pipeline_custom_asot_config_is_used():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=20, frame_size=(4, 4), seed=4)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    custom_cfg = ASOTConfig(epsilon=0.5, rho=0.1)
    pipeline = OVTASPipeline(
        encoder=MockEncoder(embed_dim=8), asot_config=custom_cfg, seed=4
    )
    result = pipeline.run(frames, ann.label_names, video_id=video_id)
    assert result.smts.sinkhorn is not None


@pytest.mark.integration
def test_pipeline_reproducible_with_same_seed():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=20, frame_size=(4, 4), seed=5)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)

    p1 = OVTASPipeline(encoder=MockEncoder(embed_dim=8), seed=99)
    p2 = OVTASPipeline(encoder=MockEncoder(embed_dim=8), seed=99)
    r1 = p1.run(frames, ann.label_names, video_id=video_id)
    r2 = p2.run(frames, ann.label_names, video_id=video_id)

    np.testing.assert_array_equal(r1.action_order, r2.action_order)
    np.testing.assert_array_equal(r1.predicted_labels, r2.predicted_labels)
