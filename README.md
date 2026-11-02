# Multimodal Speech Emotion Recognition (audio + ASR transcript) on MELD

Pipeline: audio → Whisper transcript → audio encoder + text encoder → fusion → emotion class.
The full plan with per-phase gates is in [PLAN.md](PLAN.md); gate results are logged in
[reports/gates.md](reports/gates.md).

## Setup

```bash
uv sync                 # base environment (data, EDA)
uv sync --extra dl      # adds torch, transformers, opensmile, captum, shap, gradio
```

Requires `ffmpeg` on the PATH (`brew install ffmpeg`).

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

## Citation

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., Mihalcea, R. (2019).
MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019.
