"""
Frame extraction utilities.

Loading raw video (``.mp4``/``.avi``) needs OpenCV or decord, which is
kept as the optional ``video`` extra (``pip install -e ".[video]"``)
so the rest of the package stays lightweight. If a dataset already
ships pre-extracted frame images (a directory of ``.jpg``/``.png``
files), only Pillow is needed for that path.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np


def list_frame_files(frames_dir: str) -> List[str]:
    """Return sorted image file paths inside ``frames_dir``."""
    valid_ext = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(valid_ext)]
    files.sort()
    return [os.path.join(frames_dir, f) for f in files]


def load_frames_from_directory(frames_dir: str, stride: int = 1) -> List["Image.Image"]:
    """Load every ``stride``-th frame image from a directory as PIL Images."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Loading frame images requires Pillow. Install it with:\n"
            '    pip install -e ".[vlm]"   (Pillow is bundled with the vlm extra)'
        ) from exc

    paths = list_frame_files(frames_dir)[::stride]
    return [Image.open(p).convert("RGB") for p in paths]


def extract_frames_from_video(
    video_path: str, stride: int = 1, max_frames: int = None
) -> List["Image.Image"]:
    """Decode frames from a video file at every ``stride``-th frame.

    Requires OpenCV (``pip install -e ".[video]"``).
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Extracting frames from a video file requires OpenCV. Install it with:\n"
            '    pip install -e ".[video]"'
        ) from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Extracting frames from a video also requires Pillow. Install it with:\n"
            '    pip install -e ".[vlm]"'
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames: List[Image.Image] = []
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
                frames.append(Image.fromarray(frame_rgb.astype(np.uint8)))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return frames


def load_frames(path: str, stride: int = 1, max_frames: int = None) -> List["Image.Image"]:
    """Dispatch to directory- or video-based loading based on ``path``."""
    if os.path.isdir(path):
        frames = load_frames_from_directory(path, stride=stride)
        return frames[:max_frames] if max_frames is not None else frames
    return extract_frames_from_video(path, stride=stride, max_frames=max_frames)
