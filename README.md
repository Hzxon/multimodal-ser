# Multimodal Speech Emotion Recognition (audio + ASR transcript) on MELD

Pipeline: audio → Whisper transcript → audio encoder + text encoder → fusion → emotion class.
The full plan with per-phase gates is in [PLAN.md](PLAN.md); gate results are logged in
[reports/gates.md](reports/gates.md).

## Setup

```bash
uv sync                 # everything: data, EDA, torch, transformers, opensmile, captum, shap, gradio, pytest
```

Requires `ffmpeg` on the PATH (`brew install ffmpeg`). The deep-learning stack is part of the
base dependencies on purpose: a plain `uv sync` drops packages that live only in an extra.

## Data

MELD (Poria et al., ACL 2019), GPL-3.0 licensed, from the authors' repository.

1. Download `MELD.Raw.tar.gz` (10.9 GB) from
   `https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz`
   (mirror: `http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`) into `data/raw/`.
2. The label CSVs `data/raw/{train,dev,test}_sent_emo.csv` come from
   `https://github.com/declare-lab/MELD/tree/master/data/MELD`.
3. Run everything for Phase 1 and 2:

```bash
bash scripts/run_data_pipeline.sh
```

This extracts the archive, converts every clip to mono 16 kHz WAV under
`data/processed/wav16k/`, builds `data/manifests/{train,dev,test}.csv`, runs the Gate 1
dataset checks (`reports/gate1.md`) and the EDA (`reports/eda_summary.md`,
`reports/gate2.md`, figures in `reports/figures/`).

Individual steps:

```bash
bash scripts/prepare_raw.sh                  # extract nested archives
.venv/bin/python -m src.data.extract_audio   # mp4 -> wav16k
.venv/bin/python -m src.data.build_manifest  # manifests (+ --labels-only before audio exists)
.venv/bin/python scripts/check_dataset.py    # Gate 1
.venv/bin/python -m src.eda                  # Gate 2
```

## Phases 3-7: ASR, features, models, evaluation, explanations

Every stage reads the frozen manifests through `src/data/dataset.py` (data policy applied in
one place) and accepts `--manifest-dir`, `--limit` and a device option, so the whole pipeline
can be exercised on the 48-row smoke manifests before any full run.

```bash
bash scripts/run_all.sh                                  # full ladder, seeds 13 42 2024; log in runs/run_all_<timestamp>.log
bash scripts/run_all.sh --skip-data --skip-asr --device cuda   # reuse manifests and the local ASR cache (not in git; needs an earlier stage 1)
bash scripts/run_all.sh --smoke --skip-asr --device cpu  # plumbing check on 48 rows per split, about 26 min on an M1 CPU
```

Stages: 0 data, 1 ASR, 2 features, 3 baselines, 4 training, 5 sanity (M1, M2), 6 class-weight
ablation (M7), 7 evaluation (test, once per run), 8 explanations, 9 unit tests. Select with
`--only 7,8` or drop with `--skip 3,5`. The full ladder is a GPU job: about 6-9 T4-hours for
three seeds. On an Apple M1: whisper-small ASR 3-4 h, eGeMAPS 3 min, WavLM cache 20 min,
RoBERTa 25 min/epoch, WavLM fine-tune 50 min/epoch, fusion models about 3 h/epoch.

Individual steps (each CLI has `--help`):

```bash
.venv/bin/python -m src.asr.transcribe --split all        # Whisper, cached in data/manifests/asr/<tag>/
.venv/bin/python -m src.asr.quality --tag whisper-small   # WER per split and class
.venv/bin/python scripts/check_asr.py --tag whisper-small --n 100          # Gate 3: F1, F2
.venv/bin/python -m src.features.egemaps --split all --fit-scaler          # eGeMAPS + train-only scaler
.venv/bin/python -m src.features.ssl_embed --split all --batch-size 1      # WavLM 13-layer cache
.venv/bin/python -m src.features.text_embed --fit-tfidf --text-source asr --asr-tag whisper-small
.venv/bin/python scripts/check_leakage.py --n 50          # Gate 3: F3-F8
.venv/bin/python -m src.baselines --config configs/text_tfidf.yaml --seed 13   # models 0-3
.venv/bin/python -m src.train --config configs/fusion_gated.yaml --seed 13     # models 4-8
.venv/bin/python scripts/sanity_checks.py --families text_roberta,audio_wavlm_cached,fusion_gated --steps 200   # Gate 4: M1, M2
.venv/bin/python scripts/hparam_grid.py --family text_roberta --dry-run   # dev-only grid, at most 12 configs
.venv/bin/python -m src.evaluate run --run runs/fusion_gated/13 --text-conditions --noise-snr 20,10   # one test pass
.venv/bin/python -m src.evaluate summary --runs-root runs --out reports/results.md   # Gate 5 tables, CIs, McNemar
.venv/bin/python -m src.explain --run runs/fusion_gated/13 --text-run runs/text_roberta/13 \
    --audio-run runs/audio_wavlm_ft/13 --egemaps-run runs/audio_egemaps_lr/13 --n 200    # Gate 6
.venv/bin/python -m pytest tests/
```

Smoke-test variants point every CLI at the 48-row subset:
`scripts/make_subset_manifests.py --n 48 --out data/processed/smoke_manifests`, then
`--manifest-dir data/processed/smoke_manifests --feature-root data/processed/smoke_manifests/features --runs-root data/processed/smoke_runs`.

Final ASR model: `bash scripts/run_all.sh --asr-tag whisper-large-v3-turbo` transcribes with the
final Whisper model from `configs/asr.yaml` and trains every text-consuming model on those
transcripts (`whisper-small` is the iteration default).

Hyper-parameter grids run apart from `run_all.sh`, on dev only, before the final ladder:
`scripts/hparam_grid.py --family <text_roberta|audio_wavlm_ft|fusion_gated|fusion_xattn>` writes
`reports/hparam_<family>.md` and `configs/<family>_best.yaml`; copy the winning values into the
family config by hand before running the ladder.

Before the report: fill `reports/literature.yaml` with three cited MELD results on the same split
and metric (Gate V7), and annotate the manual columns of `reports/error_analysis.csv` (Gate V8).

Outputs: `runs/<name>/<seed>/{config.yaml, metrics.json, predictions.csv, best.pt or model.pkl,
git_commit.txt, test_metrics.json, test_predictions*.csv}`; `reports/results.md`,
`reports/gate3_asr.md`, `reports/gate3_features.md`, `reports/gate4_sanity.md`,
`reports/gate5_eval.md`, `reports/error_analysis.csv`, `reports/xai/` (Gate 6 report, figures,
case studies), `reports/figures/`.

Notebooks: `notebooks/00_project_results.ipynb` is the presentation of the project results in one
place (pipeline diagram, data, ASR quality, results with confidence intervals, training curves,
per-class F1 and confusion matrix, significance, robustness, ablations, explanations, gates); open
it after the full run and use Run All. `notebooks/02_asr_quality.ipynb`, `03_results.ipynb`,
`04_xai.ipynb` are thin viewers over the reports and `runs/` for the detail of each phase.

## Full run from a fresh clone

Git holds the code, configs, frozen manifests and reports. The audio, the Whisper transcripts
(`data/manifests/asr/`), the feature caches and every run under `runs/` are local, so a fresh
clone rebuilds them. Needs `uv`, `ffmpeg`, internet (MELD and the Hugging Face models
roberta-base, wavlm-base-plus and whisper-small download on first use), a CUDA GPU and roughly
50-60 GB of free disk (estimate: archive and extracted clips about 22 GB, checkpoints about
10 GB per seed in the smoke run).

1. Clone and install (`uv` picks Python 3.11 and creates `.venv/`, which `run_all.sh` uses):

   ```bash
   git clone https://github.com/Hzxon/multimodal-ser.git && cd multimodal-ser
   ```

   ```bash
   uv sync
   ```

2. Download MELD (10.9 GB archive plus the label CSVs) into `data/raw/`:

   ```bash
   mkdir -p data/raw && curl -L -o data/raw/MELD.Raw.tar.gz https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
   ```

   ```bash
   for s in train dev test; do curl -L -o data/raw/${s}_sent_emo.csv https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/${s}_sent_emo.csv; done
   ```

3. Build the audio once, then check the setup on the 48-row smoke subset (about 40 min; smoke
   mode skips stage 0 and reuses the WAVs):

   ```bash
   bash scripts/run_all.sh --only 0 --device cuda
   ```

   ```bash
   bash scripts/run_all.sh --smoke --device cuda
   ```

4. Run the full ladder in `tmux` or with `nohup`. Stage 0 is already done; stage 1 (Whisper)
   must run, because the transcripts are not in git:

   ```bash
   bash scripts/run_all.sh --skip-data --device cuda
   ```

   The script prints its time estimate at the start (about 6-9 T4-hours for three seeds plus ASR
   and the stage-6 ablation arms). If it stops, re-run with the finished stages skipped, e.g.
   `--skip-data --skip 1,2,3`. Stage 7 resumes where it stopped; stages 4 and 6 retrain from
   scratch, so skip them once they have finished.

5. Before the report: fill `reports/literature.yaml` (Gate V7) and the manual columns of
   `reports/error_analysis.csv` (Gate V8). The run rewrites tracked files under `reports/` and
   `data/manifests/`, so agree on who commits the results.

## Demo

```bash
.venv/bin/python app/demo.py --run runs/fusion_gated/13   # http://127.0.0.1:7860, CPU
```

Microphone or file, then Whisper transcript, emotion probabilities, Integrated-Gradients token
heat-map, 250 ms occlusion saliency under the waveform, and the gate value `g` as the modality
contribution. Falls back to the text-only run with a notice when the fusion checkpoint is missing.

## Citation

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., Mihalcea, R. (2019).
MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019.
