# Pushing OVTAS to GitHub (`Sagnik120`)

This guide assumes you've unzipped the delivered `OVTAS.zip` somewhere on
your machine and have a terminal open **inside the `OVTAS/` folder**.

## 0. One-time setup (skip if already done on this machine)

```bash
git --version                      # confirm git is installed
git config --global user.name  "Sagnik120"
git config --global user.email "YOUR_GITHUB_EMAIL@example.com"
```

## 1. Create the (empty) GitHub repository

Go to https://github.com/new in your browser, logged in as **Sagnik120**:

* Repository name: `OVTAS` (or any name you like -- update the `git remote`
  URL below to match)
* **Do NOT** check "Add a README", "Add .gitignore", or "Choose a license"
  -- this project already has all three, and letting GitHub create them too
  will cause a conflict on your first push.
* Click **Create repository**. Leave the "Quick setup" page open; you'll
  need the URL it shows you.

## 2. Initialize git locally and make the first commit

```bash
cd OVTAS
git init -b main
git add .gitignore README.md LICENSE pyproject.toml requirements.txt
git commit -m "chore: initialize repo scaffolding (gitignore, license, readme, packaging)"
```

## 3. Build up a clean, logical commit history

The project was built stage-by-stage, tested at each stage -- committing it
the same way (rather than one giant commit) makes the history readable and
matches how the code was actually developed and verified:

```bash
# --- Core plug-in infrastructure ---
git add src/ovtas/__init__.py src/ovtas/version.py src/ovtas/registry.py
git commit -m "feat(core): add package skeleton and generic registry pattern"

git add tests/__init__.py tests/test_registry.py
git commit -m "test(core): unit tests for the registry pattern"

# --- Stage 1: FAES ---
git add src/ovtas/stage1_faes/prompts.py
git commit -m "feat(faes): add prompt construction / label normalization"

git add src/ovtas/stage1_faes/similarity.py
git commit -m "feat(faes): add cosine similarity matrix computation (S = X A^T)"

git add src/ovtas/stage1_faes/__init__.py
git commit -m "feat(faes): expose Stage 1 public API"

git add tests/test_stage1_faes.py
git commit -m "test(faes): unit tests for prompts and similarity computation"

# --- Stage 2: SMTS ---
git add src/ovtas/stage2_smts/sinkhorn.py
git commit -m "feat(smts): add log-stabilized Sinkhorn-Knopp OT solver"

git add src/ovtas/stage2_smts/asot_decoder.py
git commit -m "feat(smts): add ASOT decoder (cost + temporal prior + argmax)"

git add src/ovtas/stage2_smts/__init__.py
git commit -m "feat(smts): expose Stage 2 public API"

git add tests/test_stage2_smts.py
git commit -m "test(smts): unit tests for Sinkhorn solver and ASOT decoder"

# --- Metrics ---
git add src/ovtas/metrics/
git commit -m "feat(metrics): add Accuracy, Edit score, and F1@k segmentation metrics"

git add tests/test_metrics.py
git commit -m "test(metrics): unit tests for all segmentation metrics"

# --- Baselines ---
git add src/ovtas/baselines/
git commit -m "feat(baselines): add RU, ES-Mean, ES-Vote, ES-NRP training-free baselines"

git add tests/test_baselines.py
git commit -m "test(baselines): unit tests for all four baselines"

# --- Encoders ---
git add src/ovtas/encoders/base.py src/ovtas/encoders/registry.py
git commit -m "feat(encoders): add BaseVLMEncoder interface and encoder registry"

git add src/ovtas/encoders/mock_encoder.py
git commit -m "feat(encoders): add dependency-free MockEncoder for tests/CI/demo"

git add src/ovtas/encoders/clip_encoder.py
git commit -m "feat(encoders): add CLIP encoder backend (open_clip, optional dep)"

git add src/ovtas/encoders/siglip_encoder.py
git commit -m "feat(encoders): add SigLIP encoder backend (transformers, optional dep)"

git add src/ovtas/encoders/__init__.py
git commit -m "feat(encoders): expose encoder public API with optional-dep-safe imports"

git add tests/test_encoders.py
git commit -m "test(encoders): unit tests for the encoder interface and MockEncoder"

# --- Data ---
git add src/ovtas/data/datasets.py
git commit -m "feat(data): add BaseTASDataset interface and GTEA dataset loader"

git add src/ovtas/data/video_utils.py
git commit -m "feat(data): add frame extraction from videos/frame directories"

git add src/ovtas/data/synthetic.py
git commit -m "feat(data): add in-memory SyntheticTASDataset for tests/demo"

git add src/ovtas/data/__init__.py
git commit -m "feat(data): expose data-loading public API"

git add tests/test_data.py
git commit -m "test(data): unit tests for synthetic dataset and GTEA loader"

# --- Utilities and config ---
git add src/ovtas/utils/
git commit -m "feat(utils): add seeding and logging helpers"

git add src/ovtas/config.py
git commit -m "feat(config): add dataclass-based YAML configuration schema"

git add tests/test_config.py
git commit -m "test(config): unit tests for config load/save round-trip"

# --- Pipeline ---
git add src/ovtas/pipeline.py
git commit -m "feat(pipeline): add OVTASPipeline wiring encoder + FAES + SMTS"

git add tests/test_pipeline.py
git commit -m "test(pipeline): end-to-end tests using MockEncoder + SyntheticTASDataset"

# --- Scripts ---
git add scripts/__init__.py scripts/extract_embeddings.py
git commit -m "feat(scripts): add extract_embeddings.py CLI"

git add scripts/run_pipeline.py
git commit -m "feat(scripts): add run_pipeline.py CLI (fast + full modes)"

git add scripts/evaluate.py
git commit -m "feat(scripts): add evaluate.py CLI for metrics reporting"

git add scripts/download_gtea.py
git commit -m "docs(scripts): add GTEA dataset acquisition guide + layout validator"

# --- Configs ---
git add configs/
git commit -m "feat(configs): add default/GTEA(SigLIP)/GTEA(CLIP-lightweight) configs"

# --- CI ---
git add .github/workflows/tests.yml
git commit -m "ci: add GitHub Actions workflow to run the test suite"

# --- Anything left over ---
git add -A
git commit -m "chore: final packaging touches" --allow-empty
```

> Tip: if you'd rather not type all of the above, `git add -A && git commit
> -m "feat: full OVTAS implementation"` in one shot also works fine -- the
> broken-out history above is just for readability, not required by git.

## 4. Connect to GitHub and push

Copy the URL GitHub showed you in Step 1 (it looks like
`https://github.com/Sagnik120/OVTAS.git`), then:

```bash
git remote add origin https://github.com/Sagnik120/OVTAS.git
git push -u origin main
```

If prompted for a password, GitHub no longer accepts your account password
for `git push` over HTTPS -- use a **Personal Access Token** instead
(GitHub → Settings → Developer settings → Personal access tokens → generate
one with `repo` scope, then paste it in place of your password), or set up
SSH keys and use `git@github.com:Sagnik120/OVTAS.git` as the remote instead.

## 5. Verify

Refresh `https://github.com/Sagnik120/OVTAS` in your browser -- you should
see the full file tree, the README rendered on the repo homepage, and (after
a minute) a green check mark from the `tests` GitHub Actions workflow on
your commit.

## 6. (Optional) Also publish to Hugging Face / Kaggle

* **Hugging Face** (`Sagnik120`): if you later want to share cached
  embeddings, predictions, or a model card, create a new Space or Dataset
  repo at https://huggingface.co/new and `git push` to it the same way
  (Hugging Face repos are also git repos).
* **Kaggle** (`sagnikchandra027`): Kaggle Datasets/Notebooks are uploaded
  via the Kaggle web UI or the `kaggle` CLI (`kaggle datasets create`) --
  not `git push` -- so that's a separate, later step once you have real
  GTEA results to share.

Neither is required to finish this task; they're just there if useful later.
