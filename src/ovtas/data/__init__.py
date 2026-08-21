"""
Dataset loading utilities.

    DATASETS registry:
        "gtea"       -> GTEADataset        (real dataset, needs files on disk)
        "gtea_verb11"-> GTEAVerbDataset     (same data, 11-verb protocol)
        "synthetic"  -> SyntheticTASDataset (in-memory, no download needed)

Add a new dataset by subclassing BaseTASDataset and decorating it with
@DATASETS.register("your_name").
"""

from ovtas.data.datasets import (
    BaseTASDataset,
    DATASETS,
    GTEADataset,
    GTEAVerbDataset,
    VideoAnnotation,
)
from ovtas.data.synthetic import SyntheticTASDataset
from ovtas.data.video_utils import load_frames

__all__ = [
    "BaseTASDataset",
    "DATASETS",
    "GTEADataset",
    "GTEAVerbDataset",
    "VideoAnnotation",
    "SyntheticTASDataset",
    "load_frames",
]
