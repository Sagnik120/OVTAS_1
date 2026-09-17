# OVTAS: Open-Vocabulary Zero-Shot Temporal Action Segmentation

A **training-free, zero-shot, open-vocabulary** implementation of Temporal
Action Segmentation (TAS), following the two-stage OVTAS pipeline:

```
        frozen VLM                    entropy-regularized OT
video ───────────────► similarity ───────────────────────────► per-frame
frames                  matrix S       (Sinkhorn + temporal          labels
                        (FAES)          prior, ASOT decoder)
                                            (SMTS)
```

* **Stage 1 -- FAES** (Frame-Action Embedding Similarity): encode video
  frames and candidate action-label phrases with a frozen Vision-Language
  Model, and compute their cosine similarity matrix.
* **Stage 2 -- SMTS** (Similarity-Matrix driven Temporal Segmentation):
  decode that similarity matrix into a temporally-consistent label sequence
  using an entropy-regularized Optimal Transport solver (log-stabilized
  Sinkhorn-Knopp) with a diagonal temporal prior (the ASOT decoder of
  Xu & Gould, 2024).

No training, no fine-tuning, no gradient steps anywhere -- every component
runs a frozen model or a closed-form optimization.

> **Honest scope note:** this repository is a from-scratch, from-the-paper
> re-implementation for study/reproduction purposes. It is **not** the
> original authors' code. I have not been able to verify every
> hyperparameter/implementation detail against the paper text alone (a few
> things, like the exact Sinkhorn iteration count, aren't fully specified in
> the paper) -- see the "Known gaps vs. the paper" section below for exactly
> what is and isn't nailed down, and the docstring in
> `src/ovtas/stage2_smts/asot_decoder.py` for the specific hyperparameter
> caveat around the balanced vs. unbalanced OT formulation.

---

## Why this file structure?

This is a proper installable Python **package** (`src/ovtas/`), not a pile
of scripts, and it's organized so that adding something new never means
editing existing, already-tested code:

```
OVTAS/
├── src/ovtas/                  # the installable "ovtas" package
│   ├── registry.py             # generic name -> class plug-in registry
│   ├── config.py                # dataclass-based YAML config schema
│   ├── pipeline.py               # OVTASPipeline: wires stages together
│   ├── version.py
│   │
│   ├── encoders/                 # Stage 0: frozen VLM backbones
│   │   ├── base.py               #   BaseVLMEncoder interface
│   │   ├── registry.py           #   ENCODERS registry
│   │   ├── mock_encoder.py       #   dependency-free encoder (tests/CI/demo)
│   │   ├── clip_encoder.py       #   OpenAI CLIP via open_clip (optional dep)
│   │   └── siglip_encoder.py     #   SigLIP via transformers (optional dep)
│   │
│   ├── stage1_faes/              # Stage 1: similarity matrix
│   │   ├── prompts.py            #   label -> natural-language phrase
│   │   └── similarity.py         #   S = X A^T, softmax
│   │
│   ├── stage2_smts/              # Stage 2: OT-based decoding
│   │   ├── sinkhorn.py           #   log-stabilized Sinkhorn-Knopp solver
│   │   └── asot_decoder.py       #   cost + temporal prior + decode
│   │
│   ├── metrics/                  # Accuracy, Edit score, F1@{10,25,50}
│   ├── baselines/                # RU, ES-Mean, ES-Vote, ES-NRP
│   ├── data/                     # dataset + video-frame loading
│   └── utils/                    # seeding, logging
│
├── scripts/                      # thin CLIs built on top of the package
│   ├── extract_embeddings.py     # encoder -> cached .npz embeddings
│   ├── run_pipeline.py           # FAES + SMTS -> predictions
│   ├── evaluate.py               # predictions -> metrics table / CSV
│   └── download_gtea.py          # dataset acquisition guide + validator
│
├── configs/                       # YAML configs (no code changes needed
│   ├── default.yaml               #   to change encoder/dataset/hyperparams)
│   ├── gtea.yaml
│   └── gtea_clip_lightweight.yaml
│
├── tests/                         # pytest suite, one file per module,
│                                   # 96 tests, all offline / dependency-free
├── pyproject.toml
├── requirements.txt
└── .github/workflows/tests.yml    # CI: installs + runs the test suite
```

**Adding something new never touches existing files.** Examples:

* A new VLM backbone → new file in `encoders/`, decorated with
  `@ENCODERS.register("your_name")`. `pipeline.py` and every test that uses
  `ENCODERS.build(...)` picks it up automatically.
* A new baseline → new file in `baselines/`, `@BASELINES.register(...)`.
* A new dataset → subclass `BaseTASDataset` in `data/`, `@DATASETS.register(...)`.
* A new loss / decoder for Stage 2 → new module in `stage2_smts/`; swap it in
  via `OVTASPipeline(..., asot_config=...)` or a new sibling to
  `decode_asot`.

This is the "registry" pattern (see `src/ovtas/registry.py` -- ~80 lines,
fully unit tested), used consistently across encoders/baselines/datasets.

---

## Why only GTEA (and not 50 Salads / Breakfast)?

The paper evaluates on three datasets: **GTEA** (28 videos, ~20 minutes
total, 11 classes), **50 Salads** (50 videos, ~4.5 hours, 17/52 classes),
and **Breakfast** (1712 videos, ~66 hours, 48 classes). Given you mentioned
wanting to run this on your own machine rather than a lab cluster, **GTEA is
the only one implemented end-to-end here** -- it is small enough that even
the VLM forward pass (the expensive part; everything else is closed-form
math) is tractable on a single consumer GPU, or slowly but successfully on a
CPU-only laptop with a small `frame_stride`.

`50 Salads` / `Breakfast` support can be added later by implementing
`BaseTASDataset` for their (very similar) `groundTruth/` + `mapping.txt`
layout -- see `src/ovtas/data/datasets.py`'s `GTEADataset` as a template.

---

## Installation

```bash
# 1. Clone your repo (after you've pushed this code -- see "Pushing to
#    GitHub" below) and cd into it, OR just cd into this folder now.
cd OVTAS

# 2. Create and activate a virtual environment.
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with the core (lightweight)
#    dependencies only -- numpy/scipy/pyyaml/tqdm. This is enough to run
#    every test and the full pipeline with the dependency-free MockEncoder.
pip install --upgrade pip
pip install -e .

# 4. (Only when you're ready for REAL results) install the optional extras:
pip install -e ".[vlm]"     # torch + transformers + open_clip + Pillow
pip install -e ".[video]"   # opencv-python-headless + decord, for raw .mp4 decoding
pip install -e ".[dev]"     # pytest, ruff, black -- for development
```

`torch`/`transformers`/`open_clip` are **optional** on purpose: the core
math (Sinkhorn/OT, metrics, baselines, prompt building) is pure NumPy and
installs/runs anywhere, so you (or CI) can validate the whole algorithmic
core without ever downloading a model checkpoint.

---

## Step-by-step: running everything

### Step 1 -- verify the install with the test suite

```bash
pytest -q
```

You should see `96 passed`. This runs entirely offline using the
dependency-free `MockEncoder` and an in-memory `SyntheticTASDataset` --
no downloads, no GPU needed.

### Step 2 -- smoke-test the full CLI pipeline (still no downloads)

This proves every script works end-to-end before you touch a real dataset:

```bash
python scripts/run_pipeline.py --config configs/default.yaml \
    --out-dir outputs/smoke_predictions

python scripts/evaluate.py --predictions-dir outputs/smoke_predictions \
    --csv outputs/smoke_results.csv
```

Scores will be near-random -- `configs/default.yaml` uses `MockEncoder`,
which produces semantically meaningless (hash-based) embeddings by design.
That's expected; this step only checks plumbing.

### Step 3 -- get GTEA and install the VLM extras

```bash
python scripts/download_gtea.py                 # prints acquisition steps
# ... follow the printed steps to populate e.g. data/gtea/ ...
python scripts/download_gtea.py --check-root data/gtea   # validates the layout

pip install -e ".[vlm]" ".[video]"
```

### Step 4 -- extract embeddings once, decode many times

```bash
# Pick ONE, based on your hardware:
python scripts/extract_embeddings.py --config configs/gtea.yaml \
    --out-dir outputs/embeddings                      # SigLIP (paper's best)
# OR, on a lighter machine:
python scripts/extract_embeddings.py --config configs/gtea_clip_lightweight.yaml \
    --out-dir outputs/embeddings                      # CLIP ViT-B/32

python scripts/run_pipeline.py --config configs/gtea.yaml \
    --embeddings-dir outputs/embeddings \
    --out-dir outputs/predictions

python scripts/evaluate.py --predictions-dir outputs/predictions \
    --csv outputs/results.csv
```

Splitting extraction from decoding means you can re-run Step 4's second
command with a different `asot.epsilon` / `asot.rho` in the config in a few
seconds, without re-running the VLM.

### Step 5 -- (optional) sweep ASOT hyperparameters

Copy `configs/gtea.yaml`, change `asot.epsilon` / `asot.rho`, and re-run
`run_pipeline.py --embeddings-dir outputs/embeddings ...` with a new
`--out-dir` per setting, then compare their `evaluate.py --csv` outputs.

---

## Known gaps vs. the paper (read this before citing numbers)

I'm being explicit about this rather than quietly guessing, per how I like
to work through papers:

* **Balanced vs. unbalanced OT.** The paper states its grid search found
  the *balanced* OT formulation (Eq. 1, what this repo implements) works
  best, but also separately reports three hyperparameters (`r`,
  `lambda_frames`, `lambda_actions`) that belong to the *unbalanced*
  ASOT formulation of Xu & Gould (2024). This repo implements balanced OT
  only; those three values are stored on `ASOTConfig` for documentation
  parity but are currently inert. An unbalanced solver could be added as a
  new sibling to `log_sinkhorn` without touching existing code.
* **Sinkhorn iteration count / convergence tolerance** are not stated
  explicitly in the paper text I extracted; I used generous defaults
  (`num_iters=100`, `tol=1e-6`) that converge reliably on the synthetic
  tests here -- verify convergence (`SinkhornResult.converged`) on your
  actual data and raise `sinkhorn_iters` if needed.
* **Frame sampling rate / preprocessing** (resize, exact frame extraction
  FPS) for each dataset is not fully specified; `configs/*.yaml` exposes
  `frame_stride` / `max_frames` for you to tune against your hardware and
  the paper's reported frame counts.
* This implementation has only been exercised on synthetic/mock data in
  this environment (no GPU, no internet access to model hubs here) -- you
  are the first to run it against real GTEA + real SigLIP/CLIP weights.
  Please open an issue (in your own repo) or otherwise note anything that
  looks off numerically.

---

## Pushing this to GitHub (`Sagnik120`)

See **`GITHUB_GUIDE.md`** for full step-by-step `git` commands, including
initializing the repo, writing a clean multi-commit history (20+ commits,
one logical step at a time, matching how this was built), and pushing to
`https://github.com/Sagnik120/OVTAS`.

## License

MIT -- see `LICENSE`.
