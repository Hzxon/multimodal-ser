#!/usr/bin/env bash
# Phase 1 + 2 end to end: extract archive, convert audio, build manifests, run Gate 1 and Gate 2.
# Prerequisite: data/raw/MELD.Raw.tar.gz (or an already extracted data/raw/MELD.Raw/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
bash scripts/prepare_raw.sh
$PY -m src.data.extract_audio --jobs "${JOBS:-6}"
$PY -m src.data.build_manifest
$PY scripts/check_dataset.py
$PY -m src.eda --jobs "${JOBS:-6}" > /dev/null
echo "done: reports/gate1.md reports/gate2.md reports/eda_summary.md"
