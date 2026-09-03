"""Unit tests for datasets (SyntheticTASDataset and GTEADataset loader contract)."""

from __future__ import annotations

import os
import numpy as np
import pytest

from ovtas.data import DATASETS, GTEADataset, SyntheticTASDataset


@pytest.mark.unit
def test_synthetic_dataset_registered():
    assert "synthetic" in DATASETS


@pytest.mark.unit
def test_synthetic_dataset_generates_requested_number_of_videos():
    ds = SyntheticTASDataset(num_videos=4, frames_per_video=20, seed=0)
    assert len(ds.list_videos()) == 4


@pytest.mark.unit
def test_synthetic_dataset_annotation_shapes():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=30, seed=1)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    assert ann.frame_labels.shape == (30,)
    assert ann.video_id == video_id
    assert set(np.unique(ann.frame_labels).tolist()).issubset(
        set(range(len(ds.label_names())))
    )


@pytest.mark.unit
def test_synthetic_dataset_frames_match_annotation_length():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=25, frame_size=(4, 4), seed=2)
    video_id = ds.list_videos()[0]
    ann = ds.load_annotation(video_id)
    frames = ds.get_frames(video_id)
    assert len(frames) == len(ann.frame_labels) == 25
    assert frames[0].shape == (4, 4, 3)


@pytest.mark.unit
def test_synthetic_dataset_is_reproducible_with_seed():
    ds_a = SyntheticTASDataset(num_videos=2, frames_per_video=20, seed=42)
    ds_b = SyntheticTASDataset(num_videos=2, frames_per_video=20, seed=42)
    vid = ds_a.list_videos()[0]
    np.testing.assert_array_equal(
        ds_a.load_annotation(vid).frame_labels,
        ds_b.load_annotation(vid).frame_labels,
    )


@pytest.mark.unit
def test_synthetic_dataset_unknown_video_raises():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=10)
    with pytest.raises(KeyError):
        ds.load_annotation("does_not_exist")
    with pytest.raises(KeyError):
        ds.get_frames("does_not_exist")


@pytest.mark.unit
def test_synthetic_dataset_ground_truth_has_segment_structure():
    ds = SyntheticTASDataset(num_videos=1, frames_per_video=60, seed=3)
    labels = ds.load_annotation(ds.list_videos()[0]).frame_labels
    same_as_prev = np.mean(labels[1:] == labels[:-1])
    assert same_as_prev > 0.5


@pytest.mark.unit
def test_gtea_dataset_missing_mapping_file_raises(tmp_path):
    empty_root = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        GTEADataset(root=empty_root)


@pytest.mark.unit
def test_gtea_dataset_missing_groundtruth_dir_raises(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "mapping.txt"), "w") as fh:
        fh.write("0 background\n1 pour coffee\n")
    ds = GTEADataset(root=root)
    with pytest.raises(FileNotFoundError):
        ds.list_videos()


@pytest.mark.unit
def test_gtea_dataset_loads_minimal_fixture(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "groundTruth"))
    os.makedirs(os.path.join(root, "videos", "S1_test"))

    with open(os.path.join(root, "mapping.txt"), "w") as fh:
        fh.write("0 background\n1 pour coffee\n2 add sugar\n")
    with open(os.path.join(root, "groundTruth", "S1_test.txt"), "w") as fh:
        fh.write("background\nbackground\npour coffee\nadd sugar\n")

    ds = GTEADataset(root=root)
    assert ds.list_videos() == ["S1_test"]
    assert ds.label_names() == ["background", "pour coffee", "add sugar"]

    ann = ds.load_annotation("S1_test")
    np.testing.assert_array_equal(ann.frame_labels, np.array([0, 0, 1, 2]))
    assert ann.frames_dir_or_video_path.endswith("S1_test")
