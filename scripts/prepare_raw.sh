#!/usr/bin/env bash
# Extract MELD.Raw.tar.gz and its nested train/dev/test archives into data/raw/MELD.Raw.
# Idempotent: skips archives whose target directory already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
ARCHIVE=$RAW/MELD.Raw.tar.gz

if find "$RAW" -name "dia0_utt0.mp4" -not -name "._*" | grep -q .; then
  echo "clips already extracted under $RAW; skipping archive extraction"
  find "$RAW" -name "*.mp4" -not -name "._*" | sed 's|/[^/]*$||' | sort | uniq -c
  exit 0
fi
if [ ! -d "$RAW/MELD.Raw" ]; then
  [ -f "$ARCHIVE" ] || { echo "missing $ARCHIVE"; exit 1; }
  echo "sha256 of archive (recorded in reports/gates.md):"
  shasum -a 256 "$ARCHIVE" | tee "$RAW/MELD.Raw.tar.gz.sha256"
  tar -xzf "$ARCHIVE" -C "$RAW"
fi

cd "$RAW/MELD.Raw"
for s in train dev test; do
  if [ -f "$s.tar.gz" ]; then
    echo "extracting $s.tar.gz"
    tar -xzf "$s.tar.gz"
  fi
done
echo "mp4 count per directory:"
find . -name "*.mp4" -not -name "._*" | sed 's|/[^/]*$||' | sort | uniq -c
