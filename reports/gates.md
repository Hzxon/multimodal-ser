# Gate log

Each phase's gate from PLAN.md, with the command that produced the evidence. Status
values: PASS, WARN (recorded defect, decision noted), SKIP (needs audio, pending), FAIL.

## Gate 1 — Dataset (full, 2026-11-03)

Commands: `.venv/bin/python scripts/ffprobe_inventory.py`, `.venv/bin/python -m src.data.extract_audio`, `.venv/bin/python -m src.data.build_manifest`, `.venv/bin/python scripts/check_dataset.py` (full output `reports/gate1.md`), `.venv/bin/python scripts/check_preprocessing.py` (stages 0–2, `reports/gate3_stage0-2.md`). The archive was downloaded and extracted by hand to `data/raw/MELD-RAW/MELD.Raw/`; the archive hash is therefore not recorded, the file inventory is.

| ID | Status | Finding |
|---|---|---|
| A1 | PASS | Raw inventory: 9 989 / 1 112 / 2 615 mp4 (train/dev/test, excluding 2 062 AppleDouble `._` files). All audio is 48 kHz AAC; 2-channel and 6-channel (5.1) clips occur in every split (train 1 117 / 8 871, dev 203 / 909, test 277 / 2 338), so the mix is not split-specific. One clip has no readable container (`train/dia125_utt3.mp4`, "moov atom not found"). dev lacks `dia110_utt7` and carries 4 unlabelled clips; test carries 5 unlabelled clips. |
| A2 | PASS | The test folder ships 132 utterances twice: `diaD_uttU.mp4` and `final_videos_testdiaD_uttU.mp4`. The prefixed files are the authors' corrected cuts: all 132 are shorter than the plain ones, and 33 plain counterparts are byte-identical to another utterance's clip. `extract_audio.py` prefers the corrected cut and records the source name in a `.src` file next to the WAV. |
| L1–L5 | PASS | Labels structurally clean (7 classes, sentiment consistent, no empty text, no mojibake) in the GitHub CSVs. |
| L6 | WARN → resolved | `test_dia155_utt3` has StartTime == EndTime in the CSV but its clip is 1.62 s of audio; kept, in the clean subset. |
| L7 | PASS | The CSVs inside the archive are cp1252 / double-encoded copies of the GitHub CSVs; metadata differs in 2 train rows (`dia490_utt0` episode number, `dia556_utt6` speaker, text and timestamps), where the GitHub row matches the clip (1.9 s clip, "What do I do?") and the archive row does not (claims a 10 s line). Label source: GitHub CSVs. |
| D2 | PASS | Rows without decodable audio: train 1 (`dia125_utt3`), dev 1 (`dia110_utt7`, no file), test 0. |
| D3 | PASS | Train counts before exclusion: neutral 4 710, joy 1 743, surprise 1 205, anger 1 109, sadness 683, disgust 271, fear 268; imbalance ratio 17.6 (train), 21.4 (dev), 25.1 (test). After the data policy, train: neutral 4 546, joy 1 666, surprise 1 156, anger 1 071, sadness 671, disgust 267, fear 262. |
| D4 | PASS | After exclusions no utterance id or (season, episode, start, end) key is shared across splits. Train rows excluded: 350 = 306 fragments under 0.5 s + 38 shared-audio clips + 5 timestamp duplicates + 1 unreadable. Dev and test: 0 excluded. Episodes are shared across splits (train/test 142) by design. |
| D5 | PASS | Content hash (PCM samples) of every WAV. Across splits, after exclusions: 0 identical clips. The two train/test repeats (`train_dia12_utt7/8` = `test_dia85_utt0/1`) and `train_dia887_utt0` = `test_dia117_utt8` are byte-identical audio; the two train/dev timestamp collisions are different audio (the timestamps were wrong, the clips are not). Within split, identical audio serving two utterances: train 19 pairs (excluded), dev 1 pair (`dia84_utt0/1`), test 4 pairs (`dia45_utt4/5`, `dia34_utt13/14`, `dia184_utt1/2`, `dia78_utt6/7`), kept in dev/test with `clean_subset = False`. |
| D6 | PASS (documented) | 6 main-cast speakers in all splits, 83–86 % of rows. Dialogue-disjoint, not speaker-disjoint. |
| D7 | PASS | Paths contain ids only. |
| D8 | PASS | Audio duration: median 2.47 s train, 2.44 s dev, 2.47 s test; mean 3.1–3.3 s; min 0.06 s; max 41.0 s (train `dia309_utt0`), 28.5 s (dev), 305 s (test `dia38_utt4`). Clips above 15 s: 25 / 6 / 6. |
| D9 | PASS | WAV length equals the CSV timestamp span within 1 s for all but 3 clips (train 1, test 2), so the sub-0.5 s spans are real fragments: 309 train, 38 dev, 85 test clips under 0.5 s. |
| P1 | PASS | Data policy applied in the manifests: `exclude_reason` on train only; `clean_subset` (decodable, 0.5–20 s, not shared audio) = 9 637 / 9 989 train, 1 066 / 1 109 dev, 2 504 / 2 610 test. |
| S1.1–S1.5 | PASS | Every WAV is 16 kHz mono PCM16; WAV length equals the source stream length within 22 ms (AAC padding); mono downmix within 6 dB of the loudest source channel on 200 sampled clips (no phase cancellation); no constant or clipped files; re-extracting 20 clips reproduces the samples. |
| S2.1–S2.8 | PASS | Manifest rows equal the CSVs; ids unique; schema non-null; exclusions only in train; wav ids match row ids and content hashes recompute; manifest hashes train `6e5a3dc1480f10b6`, dev `dc7f71f1a28f536c`, test `f5909e120e0bb5c1`. |

## Gate 2 — EDA (full, 2026-11-03)

Command: `.venv/bin/python -m src.eda`. Tables: `reports/eda_summary.md`; figures: `reports/figures/`. Train rows used: 9 639 (after the data policy).

| ID | Status | Finding |
|---|---|---|
| E1 | PASS | Shortcut floor on dev with metadata only (token count, nominal and audio duration, RMS, dynamic range): macro F1 0.176, weighted F1 0.190; majority class macro F1 0.085, weighted F1 0.252. Loudness and length carry a little label information (anger is the loudest class at −30.0 dB, sadness the quietest at −31.6 dB and the longest at 3.05 s median), so every model must clear macro F1 0.18 by a wide margin before it is credited with anything. |
| E2 | PASS | Class-conditional table in `eda_summary.md`. Median F0: anger 244 Hz, surprise 236 Hz, disgust 225 Hz, fear 220 Hz, joy 219 Hz, sadness 213 Hz, neutral 198 Hz. Median duration 2.1 s (surprise) to 3.05 s (sadness). Median tokens 5 (surprise) to 9 (sadness). Figures: `f0_per_class.png`, `rms_per_class.png`, `duration_per_class.png`, `tokens_per_class.png`. |
| E3 | PASS | Audio quality on kept train rows: 256 clips (2.7 %) with RMS below −40 dBFS (median length 1.1 s, mostly short interjections, 17 test clips below −50 dBFS are effectively silent, e.g. `test_dia245_utt1` "Hi!" at −62 dB); 403 clips (4.2 %) with frame-level dynamic range under 10 dB (steady background: laugh track or music); 0 clipped clips. Test: 76 near-silent (2.9 %), 140 low dynamic range (5.4 %). `spectrogram_examples.png` shows harmonic speech followed by broadband laugh-track energy in several clips (e.g. the second disgust example from 3.8 s). Figure `dyn_range_hist.png`. |
| E4 | PASS | Findings below. |

Findings and the design decisions they fix:

1. The lexical signal is real: "sorry" (89) and "lost" mark sadness, "god" (113), "wow", "whoa" mark surprise, "great" (105), "amazing", "glad" mark joy, "stupid" (16), "hell", "mad" mark anger. A TF-IDF baseline reaches dev weighted F1 0.43 (macro 0.33 with balanced weights). Decision: the text branch is the primary branch; the TF-IDF baseline is the first model on the ladder.
2. The acoustic signal exists but is small: F0 separates anger and surprise (244, 236 Hz) from neutral (198 Hz); sadness is quieter and longer. Decision: audio is evaluated on the classes where it should help (anger, surprise, sadness) and on the generic short lines, not on the overall score alone. No denoising (it strips the non-verbal cues).
3. 170 of 1 109 dev utterances (15 %) are short generic lines seen verbatim in train, where the train majority label is right only 62 % of the time ("what" is surprise 74 % of the time, "no" is anger 19 %). Decision: error analysis and the fusion claim are stratified by utterance length; these lines are where fusion must show its gain.
4. Imbalance after the policy: train neutral 4 546 vs fear 262 and disgust 267 (ratio 17.4); test fear 50, disgust 68. Decision: square-root inverse-frequency class weights, macro F1 next to weighted F1, bootstrap CIs, non-zero minority recall gate (M6).
5. The clips, not the labels, are the dirty part of MELD: 3 % fragments under 0.5 s, 2.7 % near-silent clips, 4.2 % laugh-track-dominated clips, 19 train pairs and 5 dev/test pairs of identical audio for two utterances, and 132 test clips whose corrected cuts ship under a different name. Decision: the data policy in the manifests (train exclusions, `clean_subset` flag, corrected test cuts) plus reporting every test number on the full split and on the clean subset.
6. Speaker is not a shortcut (neutral share 0.41–0.53 for every main-cast member) and length is only a weak one (E1). No speaker balancing.

## Usability audit (2026-11-02)

`reports/meld_usability_audit.md`: verdict, label-quality evidence, clip-segmentation defects measured on a dev-split audio mirror, published ceilings, alternatives, and the data policy (fragment exclusion, clean-test subset, no denoising, secondary evaluation on corrected labels) to ratify in Gate 1 once the official archive is extracted.

## Gates 3–6 — implementation status (2026-11-03)

Phases 3–7 are implemented and exercised end to end on the 48-row smoke manifests
(`bash scripts/run_all.sh --smoke --skip-asr --device cpu`, 29 min on an M1 CPU, exit 0, 45 unit tests pass; log in
`data/processed/smoke_runs/run_all.log`, reports under `data/processed/smoke_reports/`). The
smoke numbers prove the plumbing, not the models: every gate below is re-evaluated on the full
data when `bash scripts/run_all.sh` has run. Full-data artefacts built so far: eGeMAPS caches and
the train-only scaler (`n_fit = 9639`), the gold TF-IDF vocabulary (10 813 terms; 99.99 % of
train rows fit in 64 RoBERTa tokens, one truncated: `train_dia455_utt11`).

| Gate | Script / evidence | Smoke status | Full-data status |
|---|---|---|---|
| F1, F2 | `scripts/check_asr.py`, `src/asr/quality.py` → `reports/gate3_asr.md`, `reports/asr_quality_<tag>.md` | PASS (8/8 identical re-transcriptions; WER train 23.6 / dev 27.4 / test 26.0 on 48 rows) | pending `src.asr.transcribe --split all` (whisper-small, about 3–4 h on M1) |
| F3–F8 | `scripts/check_leakage.py --n 50` → `reports/gate3_features.md` | PASS, 0 WARN, 0 SKIP | **PASS on full data (2026-11-03)**: scaler `fit_split=train`, `n_fit=9639`; TF-IDF refit vocabulary identical (10 813 terms); 6/6 caches raise for the wrong split and match the manifest order; eGeMAPS 88 finite values on 9988/1108/2610 rows; WavLM (N,13,768) float16, no all-zero vector, per-layer L2 norm 2.2–4.3; recompute 150/150 eGeMAPS and 30/30 WavLM rows identical; log loudness vs EDA `rms_db` r = 0.90, F0 (semitones) vs EDA F0 median r = 0.80 (n = 13 443 voiced); clips cropped to 10 s: train 212 (2.1 %), dev 23 (2.1 %), test 48 (1.8 %) |
| F6 | `tests/test_dataset.py::test_transforms_train_only`, `tests/test_augment.py` | PASS | same code path |
| M1, M2 | `scripts/sanity_checks.py` → `reports/gate4_sanity.md` | PASS for text_roberta, audio_wavlm_cached, fusion_gated (99 % train accuracy by step 9 / 2 / 3; shuffled-label dev macro F1 0.045 / 0.045 / 0.086 inside the chance band [0.015, 0.148]) | pending `--steps 200` and 3-epoch shuffled-label runs on full data (5 families) |
| M3, M4 | `src/train.py::assert_no_test_in_data_config`, `make_loader` refuses `test`; one dev evaluation per epoch counted in `metrics.json` | PASS | same code path |
| M5–M9 | `src/evaluate.py summary` (seeds, M6 verdict row, M7 class-weight table from dev scores with the Section 5 rejection rule and the undersample arm, fusion family auto-selected on dev, M8 gain, M9 gap) | machinery runs; M6 and M8 FAIL on smoke (6-step models) | pending full runs |
| M10 | `scripts/run_all.sh` | ran end to end in smoke mode | pending full mode (GPU job) |
| V1–V6, V8 | `src/evaluate.py` → `reports/gate5_eval.md`, `reports/results.md`, `reports/error_analysis.csv` | V1–V4, V6 PASS (one test pass per run enforced on 14 runs; bootstrap CIs; McNemar for every fusion family; noise 20/10 dB on 5 models); V5 PARTIAL (2×2 complete for the text family and for `fusion_gated`/`fusion_gated_gold`; the auto-selected `fusion_xattn` has no gold-trained twin); V8 PARTIAL (12 errors on 16 smoke rows, manual columns empty) | pending full runs (30 errors); V8 manual columns to annotate; add `fusion_xattn_gold` if model 8 wins on dev |
| V7 | `reports/literature.yaml` | FAIL until three cited MELD results are filled in | to do before the report |
| X1–X6 | `src/explain.py` → `reports/xai/gate6_xai.md` | X1–X5 PASS in the end-to-end smoke run (X3 on mean |ρ|: text 0.216, audio below 0.3); X6 PARTIAL (8 of 11 cases on 16 test rows) | pending full runs; X3 text should be watched even then |

Deviations from PLAN.md recorded so far: ASR transcripts are joined from
`data/manifests/asr/<tag>/<split>.csv` at load time instead of being written into the frozen
manifests; the eGeMAPS scaler lives next to its cache (`features/egemaps_<version>/scaler.pkl`),
not at `features/scaler.pkl`; `requirements.txt` is replaced by `pyproject.toml` + `uv.lock`;
SpecAugment is the WavLM encoder's own feature masking (`mask_time_prob`), active in train mode
only; the WavLM cache is built with `--batch-size 1` because zero padding changes the embedding
(group norm over time in the first conv layer).

### F7 — encoder pretraining overlap (documented)

WavLM-base-plus was pretrained self-supervised on Libri-Light (60 k h), GigaSpeech (10 k h)
and VoxPopuli (24 k h) (Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full
Stack Speech Processing", arXiv:2110.13900); none of these carries MELD emotion labels.
RoBERTa-base was pretrained on BookCorpus, English Wikipedia, CC-News, OpenWebText and Stories
(Liu et al., arXiv:1907.11692), web text without MELD labels. No emotion-pretrained encoder
(e.g. emotion2vec) is used. Whisper's training data is undisclosed and may contain Friends
subtitles, so ASR quality on MELD may be optimistic compared with unseen audio (PLAN.md 1.1);
the gold-transcript condition (V5) bounds the effect of that caveat on the text branch.
