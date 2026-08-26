#!/usr/bin/env python
"""
GTEA dataset acquisition guide.

GTEA is not distributed by a single stable, script-friendly public
URL with an open license for automated download (the mirrors used by
prior TAS papers, e.g. the MS-TCN / ASOT authors' pre-processed
release, are hosted on shifting personal/Google Drive links), so this
script does **not** attempt to auto-download anything. Instead it
prints the manual steps and verifies your local copy has the layout
``ovtas.data.datasets.GTEADataset`` expects.

Run it with no arguments to just print instructions, or with
``--check-root <path>`` to validate a directory you've already
populated.
"""

from __future__ import annotations

import argparse
import os

INSTRUCTIONS = """
GTEA dataset -- manual download steps
======================================

1. Locate a GTEA download mirror. Commonly used options:
     - The official GTEA project page:
         http://cbs.ic.gatech.edu/fpv/
     - The pre-processed splits/features released alongside MS-TCN
       (Farha & Gall, 2019) / ASOT (Xu & Gould, 2024) code repos on
       GitHub, which already have `groundTruth/`, `mapping.txt`, and
       `splits/` in the layout below (search their READMEs for the
       current Drive/Cloud link, since these move over time).

2. Create the following directory layout under a root directory of
   your choice (e.g. `data/gtea/`):

       data/gtea/
         groundTruth/
           S1_Cheese_C1.txt      <- one action label per line, one per frame
           ...
         mapping.txt              <- "<id> <label>" per line, e.g.:
                                      0 background
                                      1 pour
                                      2 take
                                      ...
         videos/
           S1_Cheese_C1/           <- EITHER a directory of extracted
             000001.jpg               .jpg/.png frames (one per line of
             000002.jpg               the matching groundTruth file)...
             ...
           S1_Cheese_C1.mp4        <- ...OR a single raw video file
                                       (requires `pip install -e ".[video]"`
                                       so OpenCV can decode it on the fly).

3. Point your config at the root directory:

       # configs/gtea.yaml
       data:
         name: gtea
         root: data/gtea
         frame_stride: 2       # subsample frames for speed on modest hardware
         prompt_dataset: gtea  # use the GTEA-specific verb+noun prompt builder

4. Verify the layout:

       python scripts/download_gtea.py --check-root data/gtea

Why GTEA specifically?
-----------------------
Of the three datasets in the paper (GTEA, 50 Salads, Breakfast), GTEA
is by far the smallest: 28 videos totalling roughly 20 minutes, with
11 action classes -- the right choice when running everything,
including the VLM forward pass, on a single consumer GPU or even a
CPU-only laptop. The other two datasets are ~10-100x larger and are
left unimplemented here for that reason (see README.md).
""".strip()


def check_root(root: str) -> None:
    from ovtas.data import GTEADataset

    print(f"Checking GTEA layout at: {root}\n")
    dataset = GTEADataset(root=root)  # raises FileNotFoundError with a clear message
    video_ids = dataset.list_videos()
    print(f"  mapping.txt      : OK ({len(dataset.label_names())} labels)")
    print(f"  groundTruth/     : OK ({len(video_ids)} videos)")
    if video_ids:
        sample = dataset.load_annotation(video_ids[0])
        print(f"  sample video     : {sample.video_id} ({len(sample.frame_labels)} frames)")
        print(f"  media path found : {sample.frames_dir_or_video_path}")
    print("\nLooks good -- you're ready to run scripts/extract_embeddings.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-root",
        default=None,
        help="If given, validate an existing GTEA root directory instead of "
        "just printing instructions.",
    )
    args = parser.parse_args()

    if args.check_root:
        check_root(args.check_root)
    else:
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
