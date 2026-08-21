"""
Synthetic, in-memory TAS dataset.

Useful for unit tests, CI, and a quick end-to-end smoke-test of the
full pipeline on any machine, with **no** dataset download and **no**
video decoding required. Frames are plain in-memory ``np.ndarray``
"images" with a controllable, block-structured ground truth so tests
can assert the pipeline recovers something close to the known-correct
segmentation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ovtas.data.datasets import BaseTASDataset, VideoAnnotation
from ovtas.data.datasets import DATASETS


@DATASETS.register("synthetic")
class SyntheticTASDataset(BaseTASDataset):
    """Generates deterministic, block-structured toy videos in memory.

    Parameters
    ----------
    label_names:
        The open-vocabulary candidate action set.
    num_videos:
        How many synthetic videos to generate.
    frames_per_video:
        Number of frames per generated video.
    frame_size:
        ``(height, width)`` of each synthetic frame.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        label_names: List[str] = None,
        num_videos: int = 3,
        frames_per_video: int = 60,
        frame_size: tuple = (8, 8),
        seed: int = 0,
    ):
        self._label_names = label_names or [
            "background",
            "pour coffee",
            "add sugar",
            "stir",
            "take cup",
        ]
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        self.rng = np.random.default_rng(seed)
        self._frames_cache = {}
        self._videos = {
            f"synthetic_{i:03d}": self._generate_video(i) for i in range(num_videos)
        }

    def label_names(self) -> List[str]:
        return list(self._label_names)

    def list_videos(self) -> List[str]:
        return sorted(self._videos.keys())

    def _generate_video(self, index: int) -> VideoAnnotation:
        num_labels = len(self._label_names)
        # A random, contiguous sequence of 3-5 segments covering the
        # whole video, so ground truth has clear segment structure
        # (rather than i.i.d. per-frame noise).
        num_segments = self.rng.integers(3, 6)
        boundaries = sorted(
            self.rng.choice(
                range(1, self.frames_per_video), size=num_segments - 1, replace=False
            )
        )
        boundaries = [0] + list(boundaries) + [self.frames_per_video]

        frame_labels = np.zeros(self.frames_per_video, dtype=np.int64)
        frames = []
        h, w = self.frame_size
        for seg_idx in range(num_segments):
            label = int(self.rng.integers(0, num_labels))
            start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
            frame_labels[start:end] = label
            # Encode the label directly into pixel intensity so a
            # (mock or real) encoder can, in principle, recover it --
            # useful for pipeline smoke tests with MockEncoder.
            base_value = (label + 1) * (255 // (num_labels + 1))
            for _ in range(start, end):
                noise = self.rng.integers(-5, 6, size=(h, w, 3))
                frame = np.clip(base_value + noise, 0, 255).astype(np.uint8)
                frames.append(frame)

        video_id = f"synthetic_{index:03d}"
        self._frames_cache[video_id] = frames
        return VideoAnnotation(
            video_id=video_id,
            frame_labels=frame_labels,
            label_names=self.label_names(),
            frames_dir_or_video_path=f"<in-memory:{video_id}>",
        )

    def load_annotation(self, video_id: str) -> VideoAnnotation:
        if video_id not in self._videos:
            raise KeyError(f"Unknown synthetic video_id: {video_id}")
        return self._videos[video_id]

    def get_frames(self, video_id: str) -> List[np.ndarray]:
        """Synthetic-only helper: return the in-memory frame arrays
        generated alongside this video's annotation.
        """
        if video_id not in self._frames_cache:
            raise KeyError(f"Unknown synthetic video_id: {video_id}")
        return self._frames_cache[video_id]
