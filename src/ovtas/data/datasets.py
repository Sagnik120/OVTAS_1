"""
Dataset abstraction for Temporal Action Segmentation corpora.

The paper evaluates on GTEA, 50 Salads, and Breakfast (Sec. IV-A.1).
Given Sagnik's hardware constraints, this codebase ships a full,
runnable loader only for **GTEA** -- by far the smallest of the three
(28 videos, ~20 minutes total, 11 action classes), which is the
dataset explicitly recommended for constrained hardware in the README.
50 Salads / Breakfast can be added later by implementing the same
:class:`BaseTASDataset` interface and registering it, without changing
any downstream code (pipeline, metrics, baselines all only depend on
the interface below).

Expected on-disk layout for GTEA (standard layout used by most public
mirrors, e.g. the one released alongside the MS-TCN / ASOT codebases)::

    gtea/
      features/            (optional, unused here -- we re-encode from
                             raw frames/video with a VLM instead of
                             using precomputed I3D features)
      groundTruth/
        S1_Cheese_C1.txt   (one action label per line, one line/frame)
        ...
      splits/
        train.split1.bundle
        test.split1.bundle
      videos/
        S1_Cheese_C1.mp4   (or a directory of extracted .jpg frames)
      mapping.txt          ("0 background\\n1 pour\\n2 take\\n...")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ovtas.registry import Registry

DATASETS: Registry = Registry("dataset")


@dataclass
class VideoAnnotation:
    """Ground truth for a single video.

    Attributes
    ----------
    video_id:
        Unique identifier (e.g. the filename stem).
    frame_labels:
        Integer label per frame (index into ``label_names``).
    label_names:
        The dataset's full open-vocabulary action-label set, i.e. the
        candidate phrases handed to Stage 1 (FAES). This is the *set*
        of possible actions for the whole dataset, not just the ones
        appearing in this particular video (matching the paper's
        "action-set supervision" setting, Sec. III-D).
    frames_dir_or_video_path:
        Path to either a directory of extracted frame images or a
        single video file, resolved by the frame-extraction utility
        in ``video_utils.py``.
    """

    video_id: str
    frame_labels: np.ndarray
    label_names: List[str]
    frames_dir_or_video_path: str


class BaseTASDataset:
    """Common interface every dataset loader implements.

    Subclasses must implement :meth:`list_videos` and
    :meth:`load_annotation`. This mirrors ``BaseVLMEncoder``: the rest
    of the codebase (pipeline, metrics, evaluate.py) only ever talks
    to this interface, never to a dataset-specific format directly.
    """

    def list_videos(self) -> List[str]:
        raise NotImplementedError

    def load_annotation(self, video_id: str) -> VideoAnnotation:
        raise NotImplementedError

    def label_names(self) -> List[str]:
        raise NotImplementedError


def _read_mapping_file(path: str) -> List[str]:
    """Parse a ``mapping.txt`` of the form ``"<id> <label>"`` per line,
    returning labels ordered by their integer id.
    """
    id_to_label: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            idx_str, label = line.split(maxsplit=1)
            id_to_label[int(idx_str)] = label
    return [id_to_label[i] for i in sorted(id_to_label)]


def _read_groundtruth_file(path: str, label_to_id: Dict[str, int]) -> np.ndarray:
    """Parse a ``groundTruth/<video>.txt`` file (one label per frame)."""
    labels = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            labels.append(label_to_id[line])
    return np.asarray(labels, dtype=np.int64)


@DATASETS.register("gtea")
class GTEADataset(BaseTASDataset):
    """Loader for the Georgia Tech Egocentric Activities (GTEA) dataset.

    Parameters
    ----------
    root:
        Path to the GTEA root directory (see module docstring for the
        expected layout).
    videos_subdir:
        Subdirectory holding raw videos or per-video frame folders.
        Defaults to ``"videos"``.
    """

    def __init__(self, root: str, videos_subdir: str = "videos"):
        self.root = root
        self.videos_dir = os.path.join(root, videos_subdir)
        self.groundtruth_dir = os.path.join(root, "groundTruth")
        self.mapping_path = os.path.join(root, "mapping.txt")

        if not os.path.isfile(self.mapping_path):
            raise FileNotFoundError(
                f"GTEA mapping.txt not found at {self.mapping_path}. "
                "See ovtas.data.datasets module docstring for the "
                "expected directory layout, or scripts/download_gtea.py "
                "for download instructions."
            )
        self._label_names = _read_mapping_file(self.mapping_path)
        self._label_to_id = {name: i for i, name in enumerate(self._label_names)}

    def label_names(self) -> List[str]:
        return list(self._label_names)

    def list_videos(self) -> List[str]:
        if not os.path.isdir(self.groundtruth_dir):
            raise FileNotFoundError(
                f"GTEA groundTruth/ directory not found at {self.groundtruth_dir}."
            )
        video_ids = sorted(
            fname[:-4]
            for fname in os.listdir(self.groundtruth_dir)
            if fname.endswith(".txt")
        )
        return video_ids

    def load_annotation(self, video_id: str) -> VideoAnnotation:
        gt_path = os.path.join(self.groundtruth_dir, f"{video_id}.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        frame_labels = _read_groundtruth_file(gt_path, self._label_to_id)

        # Prefer a per-video frame directory; fall back to a single
        # video file with a common extension.
        frame_dir_candidate = os.path.join(self.videos_dir, video_id)
        if os.path.isdir(frame_dir_candidate):
            media_path = frame_dir_candidate
        else:
            media_path = None
            for ext in (".mp4", ".avi", ".mov"):
                candidate = os.path.join(self.videos_dir, video_id + ext)
                if os.path.isfile(candidate):
                    media_path = candidate
                    break
            if media_path is None:
                raise FileNotFoundError(
                    f"Could not find frames directory or video file for "
                    f"'{video_id}' under {self.videos_dir}."
                )

        return VideoAnnotation(
            video_id=video_id,
            frame_labels=frame_labels,
            label_names=self.label_names(),
            frames_dir_or_video_path=media_path,
        )


# ---------------------------------------------------------------------------
# GTEA verb-11 protocol (standard TAS evaluation)
# ---------------------------------------------------------------------------
#
# The raw GTEA mapping.txt contains 74 compound action phrases of the form
# "<verb> <object(s)>" (e.g. "pour honey bread").  The TAS literature
# evaluates on the 11-class verb protocol: 10 base action verbs + background.
#
# This dictionary was built by manually inspecting every compound label in
# mapping.txt (printed in full and verified in the OVTAS project log on
# 2026-08-03) and assigning the canonical verb.  It is intentionally
# hardcoded -- NOT derived by a runtime heuristic such as split()[0] -- so
# that the mapping is a static, auditable artefact suitable for a thesis
# appendix, and cannot silently misfire if a future label is added to
# mapping.txt without updating this dict.
#
# GTEA-11 verb set (11 classes, indices 0-10):
#   0  background
#   1  close
#   2  fold
#   3  open
#   4  pour
#   5  put
#   6  scoop
#   7  shake
#   8  spread
#   9  stir
#  10  take

_GTEA_COMPOUND_TO_VERB: Dict[str, str] = {
    # --- background ---
    "background":                  "background",
    # --- close (10 compounds) ---
    "close chocolate":             "close",
    "close coffee":                "close",
    "close honey":                 "close",
    "close jam":                   "close",
    "close ketchup":               "close",
    "close mayonnaise":            "close",
    "close mustard":               "close",
    "close peanut":                "close",
    "close sugar":                 "close",
    "close water":                 "close",
    # --- fold (1 compound) ---
    "fold bread":                  "fold",
    # --- open (12 compounds) ---
    "open cheese":                 "open",
    "open chocolate":              "open",
    "open coffee":                 "open",
    "open honey":                  "open",
    "open jam":                    "open",
    "open ketchup":                "open",
    "open mayonnaise":             "open",
    "open mustard":                "open",
    "open peanut":                 "open",
    "open sugar":                  "open",
    "open tea":                    "open",
    "open water":                  "open",
    # --- pour (10 compounds) ---
    "pour chocolate bread":        "pour",
    "pour coffee spoon cup":       "pour",
    "pour honey bread":            "pour",
    "pour honey cup":              "pour",
    "pour ketchup hotdog bread":   "pour",
    "pour mayonnaise cheese bread":"pour",
    "pour mustard cheese bread":   "pour",
    "pour mustard hotdog bread":   "pour",
    "pour sugar spoon cup":        "pour",
    "pour water cup":              "pour",
    # --- put (15 compounds) ---
    "put bread bread":             "put",
    "put bread cheese bread":      "put",
    "put cheese bread":            "put",
    "put chocolate":               "put",
    "put coffee":                  "put",
    "put honey":                   "put",
    "put hotdog bread":            "put",
    "put jam":                     "put",
    "put ketchup":                 "put",
    "put mayonnaise":              "put",
    "put mustard":                 "put",
    "put peanut":                  "put",
    "put sugar":                   "put",
    "put tea":                     "put",
    "put water":                   "put",
    # --- scoop (4 compounds) ---
    "scoop coffee spoon":          "scoop",
    "scoop jam spoon":             "scoop",
    "scoop peanut spoon":          "scoop",
    "scoop sugar spoon":           "scoop",
    # --- shake (1 compound) ---
    "shake tea cup":               "shake",
    # --- spread (2 compounds) ---
    "spread jam spoon bread":      "spread",
    "spread peanut spoon bread":   "spread",
    # --- stir (2 compounds) ---
    "stir cup":                    "stir",
    "stir spoon cup":              "stir",
    # --- take (16 compounds) ---
    "take bread":                  "take",
    "take cheese":                 "take",
    "take chocolate":              "take",
    "take coffee":                 "take",
    "take cup":                    "take",
    "take honey":                  "take",
    "take hotdog":                 "take",
    "take jam":                    "take",
    "take ketchup":                "take",
    "take mayonnaise":             "take",
    "take mustard":                "take",
    "take peanut":                 "take",
    "take spoon":                  "take",
    "take sugar":                  "take",
    "take tea":                    "take",
    "take water":                  "take",
}

# Ordered list of the 11 verb classes (matches GTEA-11 TAS literature).
_GTEA_VERB11_NAMES: List[str] = [
    "background",  # 0
    "close",       # 1
    "fold",        # 2
    "open",        # 3
    "pour",        # 4
    "put",         # 5
    "scoop",       # 6
    "shake",       # 7
    "spread",      # 8
    "stir",        # 9
    "take",        # 10
]

_GTEA_VERB11_TO_ID: Dict[str, int] = {
    v: i for i, v in enumerate(_GTEA_VERB11_NAMES)
}


@DATASETS.register("gtea_verb11")
class GTEAVerbDataset(GTEADataset):
    """GTEA dataset reduced to the standard 11-verb evaluation protocol.

    Inherits all file I/O from :class:`GTEADataset` (reads the same
    ``groundTruth/`` and ``mapping.txt`` files) but collapses the ~74
    compound action phrases down to 11 base verbs using the static
    :data:`_GTEA_COMPOUND_TO_VERB` dictionary, which was manually verified
    against the full mapping.txt on 2026-08-03.

    This matches the evaluation protocol used in TAS literature (MS-TCN,
    ASFormer, ASOT, OVTAS) where GTEA is reported as an 11-class problem.

    Parameters
    ----------
    root:
        Path to the GTEA root directory (same layout as :class:`GTEADataset`).
    videos_subdir:
        Subdirectory holding raw videos or per-video frame folders.
    """

    def __init__(self, root: str, videos_subdir: str = "videos") -> None:
        super().__init__(root, videos_subdir)

        # Validate: every compound label loaded from mapping.txt must appear
        # in our static dict.  Fail loudly if mapping.txt ever gains a new
        # label not yet covered, rather than silently mis-labelling frames.
        unknown = [
            lbl for lbl in self._label_names
            if lbl not in _GTEA_COMPOUND_TO_VERB
        ]
        if unknown:
            raise ValueError(
                f"GTEAVerbDataset: the following labels from mapping.txt are "
                f"not in _GTEA_COMPOUND_TO_VERB and need to be manually "
                f"assigned a verb before continuing: {unknown}"
            )

        # Pre-compute a per-compound-id -> verb-id lookup array for fast
        # remapping of frame_labels arrays.
        self._compound_id_to_verb_id: np.ndarray = np.array(
            [
                _GTEA_VERB11_TO_ID[_GTEA_COMPOUND_TO_VERB[lbl]]
                for lbl in self._label_names
            ],
            dtype=np.int64,
        )

    def label_names(self) -> List[str]:
        """Return the 11 verb class names in canonical order."""
        return list(_GTEA_VERB11_NAMES)

    def load_annotation(self, video_id: str) -> VideoAnnotation:
        """Load annotation and remap compound labels -> verb-11 ids."""
        # Load original 74-class annotation from GTEADataset.
        anno = super().load_annotation(video_id)
        # Remap: compound label integer -> verb-11 integer.
        mapped_labels = self._compound_id_to_verb_id[anno.frame_labels]
        return VideoAnnotation(
            video_id=anno.video_id,
            frame_labels=mapped_labels,
            label_names=self.label_names(),
            frames_dir_or_video_path=anno.frames_dir_or_video_path,
        )
