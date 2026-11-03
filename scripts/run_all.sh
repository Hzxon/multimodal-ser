#!/usr/bin/env bash
# One-command reproduction of every number in the report (PLAN.md Gate M10).
#
# Usage: bash scripts/run_all.sh [--smoke] [--skip-data] [--skip-asr] [--seeds "13 42 2024"]
#                                [--device auto|cpu|mps|cuda] [--jobs N]
#                                [--skip N[,N...]] [--only N[,N...]]
#                                [--asr-tag whisper-small|whisper-large-v3-turbo]
#
# --asr-tag: ASR transcript set used by every ASR-text model (default: `tag` in configs/asr.yaml; the
#            final model is `final.tag` there). It is passed to src.asr.*, text_embed (TF-IDF) and, as
#            --set data.asr_tag=<tag>, to src.baselines, src.train, sanity_checks and the late-fusion
#            pseudo-run. Other evaluate runs read the tag from the run's own config.yaml.
# Stages: 0 data pipeline, 1 ASR, 2 features, 3 baselines (models 0-3), 4 training (models 4, 5, 7, 8 and the
#         gold-text twins), 5 sanity checks M1/M2, 6 ablations M7 (class weights) + undersampling (PLAN 5),
#         7 evaluation (full mode skips runs that already have test_metrics.json, so a failed stage 7
#         resumes on re-run), 8 explanations, 9 pytest.
# Full mode writes runs/ and reports/ and logs to runs/run_all_<timestamp>.log. --smoke uses the 48-row
# manifests under data/processed/smoke_manifests, one seed, a few optimiser steps, and writes
# data/processed/smoke_runs/, data/processed/smoke_reports/ and data/processed/smoke_runs/run_all.log.
# Smoke numbers are plumbing checks and never go into the report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

SMOKE=0
SEEDS_ARG=""
DEVICE=auto
JOBS=6
SKIP=" "
ONLY=""
ASR_TAG_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --skip-data) SKIP="$SKIP 0 " ;;
    --skip-asr) SKIP="$SKIP 1 " ;;
    --seeds) SEEDS_ARG="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --jobs) JOBS="$2"; shift ;;
    --skip) SKIP="$SKIP $(echo "$2" | tr ',' ' ') "; shift ;;
    --only) ONLY="$(echo "$2" | tr ',' ' ')"; shift ;;
    --asr-tag) ASR_TAG_ARG="$2"; shift ;;
    -h|--help) sed -n 2,21p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$DEVICE" in auto|cpu|mps|cuda) ;; *) echo "--device must be auto|cpu|mps|cuda" >&2; exit 2 ;; esac

if [ "$SMOKE" = 1 ]; then
  MDIR=data/processed/smoke_manifests
  FROOT=$MDIR/features
  RUNS=data/processed/smoke_runs
  REPORTS=data/processed/smoke_reports
  LOG=$RUNS/run_all.log
  SEEDS=${SEEDS_ARG:-13}
  mkdir -p "$RUNS" "$REPORTS"
  : > "$LOG"
else
  MDIR=""
  FROOT=""
  RUNS=runs
  REPORTS=reports
  mkdir -p runs
  LOG=runs/run_all_$(date +%Y%m%d_%H%M%S).log
  SEEDS=${SEEDS_ARG:-"13 42 2024"}
  : > "$LOG"
fi

# ---------------------------------------------------------------------------- argument sets
# bash 3.2 (macOS) treats an empty array under `set -u` as unset, hence the ${a[@]+...} idiom.
DATA_ARGS=()
DEV_ARGS=()
[ -n "$MDIR" ] && DATA_ARGS=(--manifest-dir "$MDIR" --feature-root "$FROOT")
[ "$DEVICE" != auto ] && DEV_ARGS=(--device "$DEVICE")
BASE_ARGS=(${DATA_ARGS[@]+"${DATA_ARGS[@]}"} --runs-root "$RUNS")
TRAIN_ARGS=(${DATA_ARGS[@]+"${DATA_ARGS[@]}"} --runs-root "$RUNS" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"})
EVAL_ARGS=(${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"})
if [ "$SMOKE" = 1 ]; then
  BASE_ARGS+=(--limit 16)
  TRAIN_ARGS+=(--batch-size 4 --max-steps 6 --max-epochs 2 --limit 16)
  EVAL_ARGS+=(--batch-size 4 --limit 16)
  CHECK_N=8
  SANITY_STEPS=30   # sanity_checks.py: --steps bounds M1 only; --m2-epochs / --m2-max-steps bound M2
  SANITY_ARGS=(--batch-size 4 --limit 16 --m2-epochs 2)
  SSL_BS=4          # smoke: speed. Padding changes WavLM embeddings, so the real cache is batch size 1
  SSL_ARGS=(--batch-size "$SSL_BS")
  SANITY_FAMILIES=text_roberta,audio_wavlm_cached,fusion_gated
  EXPLAIN_N=8
  XAI_DIR=$REPORTS/xai
else
  CHECK_N=100
  SANITY_STEPS=200
  SANITY_ARGS=(--m2-epochs 3)   # M2 runs 3 full epochs on shuffled labels (PLAN M2); --steps caps M1 only
  SSL_BS=1          # batch size 1: zero padding changes WavLM embeddings (README, reports/gates.md)
  SSL_ARGS=(--batch-size "$SSL_BS")
  SANITY_FAMILIES=text_roberta,audio_wavlm_cached,audio_wavlm_ft,fusion_gated,fusion_xattn
  EXPLAIN_N=200
  XAI_DIR=$REPORTS/xai
fi
ASR_TAG=${ASR_TAG_ARG:-$($PY -c "import yaml; print(yaml.safe_load(open('configs/asr.yaml'))['tag'])")}
ALLOWED_TAGS=$($PY -c "import yaml; c = yaml.safe_load(open('configs/asr.yaml')); print(c['tag'], c['final']['tag'])")
case " $ALLOWED_TAGS " in *" $ASR_TAG "*) ;; *) echo "--asr-tag must be one of: $ALLOWED_TAGS (configs/asr.yaml)" >&2; exit 2 ;; esac
ASR_SET=(--set "data.asr_tag=$ASR_TAG")   # for every config that reads ASR text; harmless for gold-text configs
BASE_ARGS+=("${ASR_SET[@]}")
TRAIN_ARGS+=("${ASR_SET[@]}")
FIRST_SEED=${SEEDS%% *}
ABL_SEED=$FIRST_SEED
for s in $SEEDS; do [ "$s" = 13 ] && ABL_SEED=13; done

# ---------------------------------------------------------------------------- logging helpers
STATUS=""
FAILED=0
T_START=$SECONDS

say() { echo "$*" | tee -a "$LOG"; }

# run CMD...: echo the command, run it, log output and wall time; aborts the script on failure.
run() {
  local t0=$SECONDS rc=0
  say "\$ $*"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  say "  -> exit $rc in $((SECONDS - t0))s"
  if [ "$rc" -ne 0 ]; then
    say "FAILED: $*"
    STATUS="$STATUS
  stage $CUR_STAGE: FAILED ($*)"
    summary
    exit "$rc"
  fi
}

# run_check CMD...: same, but a non-zero exit (a failed gate check) is recorded and the run continues.
run_check() {
  local t0=$SECONDS rc=0
  say "\$ $*"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  say "  -> exit $rc in $((SECONDS - t0))s"
  if [ "$rc" -ne 0 ]; then
    say "CHECK FAILED (continuing): $*"
    FAILED=1
    STAGE_NOTE="$STAGE_NOTE check-failed"
  fi
}

in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

# skip_reason N: prints why stage N does not run, nothing when it runs.
skip_reason() {
  local n=$1
  if [ -n "$ONLY" ] && ! in_list "$n" "$ONLY"; then echo "not in --only"; return; fi
  if in_list "$n" "$SKIP"; then echo "skipped by flag"; return; fi
  if [ "$n" = 0 ] && [ "$SMOKE" = 1 ]; then echo "smoke mode (smoke manifests are reused)"; return; fi
}

CUR_STAGE=""
STAGE_NOTE=""
T_STAGE=0
begin_stage() {
  CUR_STAGE=$1
  STAGE_NOTE=""
  local reason
  reason=$(skip_reason "$1")
  say ""
  say "=============================================================================="
  say "== Stage $1: $2"
  say "=============================================================================="
  if [ -n "$reason" ]; then
    say "skipped: $reason"
    STATUS="$STATUS
  stage $1: skipped ($reason)  $2"
    return 1
  fi
  T_STAGE=$SECONDS
  STAGE_TITLE=$2
  return 0
}

end_stage() {
  local el=$((SECONDS - T_STAGE))
  say "stage $CUR_STAGE finished in ${el}s"
  STATUS="$STATUS
  stage $CUR_STAGE: ran in ${el}s${STAGE_NOTE}  $STAGE_TITLE"
}

note_skip() {  # a stage that ran but skipped part of its work
  say "$1"
  STAGE_NOTE="$STAGE_NOTE; $1"
}

summary() {
  say ""
  say "=============================================================================="
  say "== Summary ($([ "$SMOKE" = 1 ] && echo smoke || echo full) mode, seeds: $SEEDS, device: $DEVICE)"
  say "=============================================================================="
  say "$STATUS"
  say ""
  say "total wall time $((SECONDS - T_START))s; log: $LOG"
  if [ "$FAILED" = 1 ]; then say "at least one gate check failed (see CHECK FAILED lines above)"; fi
}

# ---------------------------------------------------------------------------- header
say "run_all.sh started $(date '+%Y-%m-%d %H:%M:%S')  mode=$([ "$SMOKE" = 1 ] && echo smoke || echo full)  seeds=\"$SEEDS\"  device=$DEVICE"
say "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)   python: $($PY --version 2>&1)"
say "runs root: $RUNS   reports root: $REPORTS   asr tag: $ASR_TAG   log: $LOG"
say ""
if [ "$SMOKE" = 1 ]; then
  say "Estimated runtime (smoke): 15-45 minutes on an Apple M1, dominated by the fusion steps (4-40 s/step)."
  say "The numbers it produces are plumbing checks, not results."
else
  say "Estimated runtime (full ladder). THE FULL LADDER IS A GPU JOB."
  say "  Timings measured on an Apple M1 (mps); T4 figures only where measured or stated."
  say "  stage 0 data pipeline            ~30-60 min (archive extraction, 13 715 clips to 16 kHz WAV, EDA)"
  say "  stage 1 whisper-small ASR        0.7-1.0 s/clip x 13 715 clips = 3-4 h on M1; the cache is <manifest dir>/asr/<tag>/*.csv"
  say "                                   (local, not in git: --skip-asr needs a previous stage-1 run on this machine)"
  say "  stage 2 features                 eGeMAPS ~3 min (6 jobs), WavLM cache (batch size 1) ~20 min or more, TF-IDF seconds"
  say "  stage 3 baselines                a few minutes per seed"
  say "  stage 4 training, per seed (early stopping, assuming ~6 of the 15 epochs):"
  say "      text_roberta, text_roberta_gold   25 min/epoch each on M1  -> ~2.5 h each"
  say "      audio_wavlm_cached                 3 min/epoch on M1       -> ~20 min"
  say "      audio_wavlm_ft                    50 min/epoch on M1       -> ~5 h"
  say "      fusion_gated, fusion_gated_gold, fusion_xattn   3 h/epoch on M1  -> ~18 h each; on a T4 8-10 min/epoch -> ~1 h each"
  say "  stage 5 sanity checks (M1, M2)   ~1-2 h on a GPU"
  say "  stage 6 ablations (seed 13)      3 arms x (text_roberta + audio_wavlm_cached + audio_wavlm_ft + fusion_gated) + 1 undersampled"
  say "                                   text_roberta (fewer rows, cheaper); the 3 extra audio_wavlm_ft runs alone take ~15 h on M1"
  say "  stage 7 evaluation               one test pass per run plus noise/transcript conditions, ~1 h on a GPU"
  say "  stage 8 explanations (n=200)     ~30-60 min on a GPU"
  say "  stage 9 pytest                   under a minute"
  say "  Total: about 6-9 GPU-hours (T4) for three seeds; on the M1 the fusion runs alone take days. Run it on a GPU."
fi

# ---------------------------------------------------------------------------- stage 0: data
if begin_stage 0 "data pipeline (extract, convert, manifests, Gate 1/2, EDA)"; then
  JOBS=$JOBS run bash scripts/run_data_pipeline.sh
  end_stage
fi
if [ "$SMOKE" = 1 ] && [ ! -f "$MDIR/train.csv" ]; then
  say "smoke manifests missing; regenerating"
  run $PY scripts/make_subset_manifests.py --n 48 --out "$MDIR"
fi

# ---------------------------------------------------------------------------- stage 1: ASR
if begin_stage 1 "ASR: transcribe all splits, quality report, check_asr (tag $ASR_TAG)"; then
  run $PY -m src.asr.transcribe --split all --tag "$ASR_TAG" ${MDIR:+--manifest-dir "$MDIR"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
  run $PY -m src.asr.quality --tag "$ASR_TAG" ${MDIR:+--manifest-dir "$MDIR"}
  run_check $PY scripts/check_asr.py --tag "$ASR_TAG" --n "$CHECK_N" ${MDIR:+--manifest-dir "$MDIR"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
  end_stage
fi
# The ASR transcripts are a local cache (not in git). Without stage 1 they must exist from an earlier run,
# otherwise every ASR-text stage fails later with a less obvious error.
ASR_DIR=${MDIR:-$($PY -c "import yaml; print(yaml.safe_load(open('configs/data.yaml'))['manifest_dir'])")}/asr/$ASR_TAG
missing_asr=""
for split in train dev test; do [ -f "$ASR_DIR/$split.csv" ] || missing_asr="$missing_asr $split"; done
needs_asr=0
for n in 2 3 4 5 6 7 8; do
  if [ -z "$(skip_reason "$n")" ]; then needs_asr=1; fi
done
if [ -n "$missing_asr" ] && [ "$needs_asr" = 1 ]; then
  say "no ASR transcripts for tag $ASR_TAG (missing$missing_asr under $ASR_DIR); they are not in git."
  say "Run stage 1 (drop --skip-asr) to create them."
  summary
  exit 1
fi

# ---------------------------------------------------------------------------- stage 2: features
if begin_stage 2 "features: eGeMAPS + scaler, WavLM cache, TF-IDF (gold, asr), leakage checks"; then
  run $PY -m src.features.egemaps --split all --fit-scaler --jobs "$JOBS" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
  run $PY -m src.features.ssl_embed --split all "${SSL_ARGS[@]}" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
  run $PY -m src.features.text_embed --fit-tfidf --text-source gold ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
  run $PY -m src.features.text_embed --fit-tfidf --text-source asr --asr-tag "$ASR_TAG" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
  run_check $PY scripts/check_leakage.py --n "$CHECK_N" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
    --out "$REPORTS/gate3_features.md"
  end_stage
fi

# ---------------------------------------------------------------------------- stage 3: baselines
if begin_stage 3 "baselines (models 0-3)"; then
  for seed in $SEEDS; do
    for cfg in majority shortcut_lr text_tfidf text_tfidf_gold audio_egemaps_lr audio_egemaps_mlp; do
      run $PY -m src.baselines --config "configs/$cfg.yaml" --seed "$seed" "${BASE_ARGS[@]}"
    done
  done
  end_stage
fi

# ---------------------------------------------------------------------------- stage 4: training
if begin_stage 4 "training (models 4, 5, 7, 8; text_roberta_gold and fusion_gated_gold for the transcript-condition table)"; then
  for seed in $SEEDS; do
    run $PY -m src.train --config configs/text_roberta.yaml --seed "$seed" "${TRAIN_ARGS[@]}"
    run $PY -m src.train --config configs/text_roberta_gold.yaml --seed "$seed" "${TRAIN_ARGS[@]}"
    run $PY -m src.train --config configs/audio_wavlm_cached.yaml --seed "$seed" "${TRAIN_ARGS[@]}"
    # init_from in these configs is a {seed} template that train.py resolves under --runs-root
    run $PY -m src.train --config configs/audio_wavlm_ft.yaml --seed "$seed" "${TRAIN_ARGS[@]}"
    for fam in fusion_gated fusion_xattn fusion_gated_gold; do
      run $PY -m src.train --config "configs/$fam.yaml" --seed "$seed" "${TRAIN_ARGS[@]}"
    done
  done
  end_stage
fi

# ---------------------------------------------------------------------------- stage 5: sanity checks
if begin_stage 5 "sanity checks M1 (overfit one batch) and M2 (shuffled labels), $SANITY_STEPS steps"; then
  # model.init_from=null: the sanity runs start from the pretrained encoders only, so they do not
  # depend on the stage-4 checkpoints.
  run_check $PY scripts/sanity_checks.py --families "$SANITY_FAMILIES" --steps "$SANITY_STEPS" --seed "$FIRST_SEED" \
    ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ${SANITY_ARGS[@]+"${SANITY_ARGS[@]}"} \
    --runs-root "$RUNS/sanity" --set model.init_from=null "${ASR_SET[@]}" --out "$REPORTS/gate4_sanity.md"
  end_stage
fi

# ---------------------------------------------------------------------------- stage 6: ablations
if begin_stage 6 "ablations on seed $ABL_SEED: class weights M7 (text_roberta, audio_wavlm_cached, audio_wavlm_ft, fusion_gated), undersampling (text_roberta)"; then
  # Every arm uses the same loss scheme along its whole init chain (audio_wavlm_cached -> audio_wavlm_ft
  # -> fusion_gated, text_roberta -> fusion_gated), so fusion_gated_cwnone starts from unweighted
  # unimodal runs instead of the class-weighted stage-4 runs. The {seed} templates are resolved under --runs-root.
  for arm in cwnone cwinv cwfocal; do
    case "$arm" in
      cwnone) ARM_SET=(--set train.class_weight=none) ;;
      cwinv) ARM_SET=(--set train.class_weight=inv) ;;
      cwfocal) ARM_SET=(--set train.loss=focal) ;;
    esac
    run $PY -m src.train --config configs/text_roberta.yaml --seed "$ABL_SEED" "${TRAIN_ARGS[@]}" \
      "${ARM_SET[@]}" --name "text_roberta_$arm"
    run $PY -m src.train --config configs/audio_wavlm_cached.yaml --seed "$ABL_SEED" "${TRAIN_ARGS[@]}" \
      "${ARM_SET[@]}" --name "audio_wavlm_cached_$arm"
    run $PY -m src.train --config configs/audio_wavlm_ft.yaml --seed "$ABL_SEED" "${TRAIN_ARGS[@]}" \
      "${ARM_SET[@]}" --set "model.init_from=runs/audio_wavlm_cached_$arm/{seed}" --name "audio_wavlm_ft_$arm"
    run $PY -m src.train --config configs/fusion_gated.yaml --seed "$ABL_SEED" "${TRAIN_ARGS[@]}" \
      "${ARM_SET[@]}" --set "model.init_from.text=runs/text_roberta_$arm/{seed}" \
      --set "model.init_from.audio=runs/audio_wavlm_ft_$arm/{seed}" --name "fusion_gated_$arm"
  done
  # PLAN 5: one undersampled run, only to show the cost of discarding majority-class data
  run $PY -m src.train --config configs/text_roberta.yaml --seed "$ABL_SEED" "${TRAIN_ARGS[@]}" \
    --set train.undersample=true --name text_roberta_undersample
  end_stage
fi

# ---------------------------------------------------------------------------- stage 7: evaluation
if begin_stage 7 "evaluation: one test pass per run, late-fusion pseudo-run, summary"; then
  if [ ! -f src/evaluate.py ]; then
    note_skip "evaluate.py not present, skipping"
  else
    if [ "$SMOKE" = 1 ]; then
      # smoke runs are retrained on every invocation, so test results of an earlier invocation are stale;
      # in full mode the one-test-pass-per-run guard (Gate V1) stays in force
      find "$RUNS" -maxdepth 3 \( -name test_metrics.json -o -name test_predictions.csv \) -not -path "*/sanity/*" -delete
      rm -rf "$RUNS/fusion_late"
    fi
    # Full mode: a run that already has test_metrics.json keeps its single test pass (Gate V1) and is
    # skipped, so the stage can be re-run after a failure and resumes with the runs not yet evaluated.
    already_evaluated() {
      if [ "$SMOKE" != 1 ] && [ -f "$1/test_metrics.json" ]; then
        note_skip "already evaluated, kept: $1"
        return 0
      fi
      return 1
    }
    for seed in $SEEDS; do
      for name in majority shortcut_lr text_tfidf text_tfidf_gold audio_egemaps_lr audio_egemaps_mlp \
                  text_roberta text_roberta_gold audio_wavlm_cached audio_wavlm_ft fusion_gated fusion_xattn fusion_gated_gold; do
        already_evaluated "$RUNS/$name/$seed" && continue
        EXTRA=()
        case "$name" in text_roberta|text_roberta_gold|fusion_gated|fusion_xattn|fusion_gated_gold) EXTRA+=(--text-conditions) ;; esac
        case "$name" in audio_wavlm_ft|fusion_gated|fusion_xattn|fusion_gated_gold) EXTRA+=(--noise-snr 20,10) ;; esac
        run $PY -m src.evaluate run --run "$RUNS/$name/$seed" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
      done
      # late fusion (model 6): src.evaluate builds the pseudo-run from configs/fusion_late.yaml and the two component runs
      # (--set only matters for this pseudo-run: other runs carry their resolved asr_tag in config.yaml)
      already_evaluated "$RUNS/fusion_late/$seed" && continue
      run $PY -m src.evaluate run --run "$RUNS/fusion_late/$seed" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} "${ASR_SET[@]}"
    done
    run $PY -m src.evaluate summary --runs-root "$RUNS" --out "$REPORTS/results.md" \
      --fusion auto --unimodal text_roberta,audio_wavlm_ft ${MDIR:+--manifest-dir "$MDIR"}
  fi
  end_stage
fi

# ---------------------------------------------------------------------------- stage 8: explanations
if begin_stage 8 "explanations on fusion_gated seed $FIRST_SEED"; then
  if [ ! -f src/explain.py ]; then
    note_skip "explain.py not present, skipping"
  else
    run $PY -m src.explain --run "$RUNS/fusion_gated/$FIRST_SEED" \
      --text-run "$RUNS/text_roberta/$FIRST_SEED" --audio-run "$RUNS/audio_wavlm_ft/$FIRST_SEED" \
      --egemaps-run "$RUNS/audio_egemaps_lr/$FIRST_SEED" \
      ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} --n "$EXPLAIN_N" --out-dir "$XAI_DIR"
  fi
  end_stage
fi

# ---------------------------------------------------------------------------- stage 9: tests
if begin_stage 9 "unit tests"; then
  run_check $PY -m pytest tests/ -q
  end_stage
fi

summary
[ "$FAILED" = 1 ] && exit 1
exit 0
