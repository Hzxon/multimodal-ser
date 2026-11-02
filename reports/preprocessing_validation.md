# Preprocessing validation plan

After the audit and the EDA, the preprocessing stage turns the raw archive into model
inputs: WAV files, manifests, ASR transcripts, feature caches, tokenised text and the
training-time augmentation. Each of those can be silently wrong in a way that still
trains a model and still produces a number. This document lists, per stage, what has to
be verified, how, and what passes. Everything automatable goes into
`scripts/check_preprocessing.py` (Gate 3); the two manual checks are listed as such.

The rule behind every check: compare the output against an independent source of truth
(the source file, the label CSV, a second decoder, a human ear, or a recomputation), not
against the pipeline's own log.

## Stage 0 · Archive and raw files

| Check | How | Pass |
|---|---|---|
| Archive is the official one | sha256 of `MELD.Raw.tar.gz` recorded in `reports/gates.md`; compare with the Hugging Face file size 10 878 146 150 bytes | Size matches; hash recorded for the report |
| Every labelled utterance has a clip, every clip has a label | Set difference between `dia{D}_utt{U}` in the CSV and the mp4 names per split | Missing and extra files listed with counts (dev is known to miss `dia110_utt7` and to carry 4 unlabelled clips); nothing else unexpected |
| Clips decode | ffprobe on every mp4: audio stream present, codec, sample rate, channels, duration | Failures logged; count per split under 1 %; `train_dia125_utt3` expected |

## Stage 1 · Audio extraction (mp4 → 16 kHz mono PCM16 WAV)

| Check | How | Pass |
|---|---|---|
| Output format | `soundfile.info` on every WAV: 16 000 Hz, 1 channel, PCM_16 | 100 % |
| Length preserved | WAV duration vs ffprobe duration of the source mp4 | Difference under 20 ms for every clip |
| Downmix did not cancel the speech | For clips whose source is stereo, RMS of the mono mix vs RMS of the louder channel | Mono RMS within 6 dB of the louder channel; a clip where the mix is 20 dB quieter has phase-inverted channels and must be re-extracted from one channel |
| No empty or constant output | Max absolute sample and RMS per WAV | No all-zero files; RMS distribution matches the dev probe (median about −30 dBFS); clips below −40 dBFS listed |
| Clipping not introduced | Fraction of samples at ±1.0 | 0 clips above 0.1 % |
| Idempotent | Re-run extraction; compare sha256 of 100 random WAVs | Identical |
| Manual: speech matches transcript | Listen to 10 random clips per split (30 total) with the gold text on screen | Speech audible and matching in at least 28; every mismatch recorded with its utt_id |

## Stage 2 · Manifests and the data policy

| Check | How | Pass |
|---|---|---|
| Row counts | `len(manifest) == len(CSV)` per split | 9 989 / 1 109 / 2 610 |
| Keys | `utt_id` unique within and across splits; `(dialogue_id, utterance_id)` unique within split | No duplicates |
| Schema | Required columns present, no NaN in `emotion`, `gold_text`, `speaker`; `duration_s` NaN only where `has_audio` is False; `emotion` in the 7-class list | Assertion |
| Exclusion rules applied as written | Count rows with each `exclude_reason`; dev and test have zero exclusions | train: 5 duplicates + clips under 0.5 s (expected 283) + decode failures; dev/test: 0 |
| Clean-test subset defined once | `clean_test` flag = `has_audio and 0.5 s ≤ duration_s ≤ 20 s`, stored in the manifest | Count reported; the same flag file is used by every evaluation |
| Audio matches label row | For 50 random rows, re-derive `dia{D}_utt{U}` from `wav_path` and compare with the row's ids; recompute sha256 and compare | 50 / 50 |
| Cross-split leakage after processing | D4 and D5 rerun on the final manifests (timestamp key and audio sha256) | 0 overlaps among non-excluded rows |
| Manifest frozen | sha256 of each manifest file written to `reports/gates.md`; later stages assert the hash | Hash matches on every run |

## Stage 3 · ASR transcripts (Whisper)

| Check | How | Pass |
|---|---|---|
| Deterministic | Transcribe 100 random clips twice (temperature 0, fixed seed, language forced to `en`) | Identical strings |
| Pinned | Model id, library version, decoding options stored in `configs/asr.yaml` and in the manifest column `asr_model` | Present |
| Coverage | Every row with `has_audio` has a non-null `asr_text` | 100 %; empty strings counted and listed |
| Hallucination screen | Flag transcripts that (a) are empty, (b) are 3× longer in words than the gold text, (c) repeat a 3-gram more than 3 times, (d) equal a known Whisper filler ("Thank you.", "Subtitles by…") on a clip with RMS below −40 dBFS | Flagged rows listed; under 2 % per split; flagged rows keep their transcript but are tagged `asr_flag` for the error analysis |
| Quality | WER per split and per class (jiwer, both sides normalised: lower-case, punctuation stripped, curly quotes unified) | Reported; dev WER within 10 points of test WER (otherwise something differs between splits) |
| Length sanity | Distribution of ASR word count vs gold word count | Median ratio between 0.8 and 1.2 |
| Manual: ASR errors are plausible | Read 30 highest-WER rows with audio | Errors are acoustic (overlap, laughter, shouting), not systematic (wrong language, truncated audio) |

## Stage 4 · Text normalisation and tokenisation

| Check | How | Pass |
|---|---|---|
| One normaliser for both text sources | The same function maps gold and ASR text: unify ’ → ', collapse whitespace, keep case and punctuation for the transformer, lower-case only inside TF-IDF | Unit test with 10 fixed strings |
| No speaker or label strings in the text | grep the text columns for the 7 class names as standalone tokens and for the 6 main-cast names; report counts | Counts reported; class-name mentions ("I'm so sad") are legitimate and stay, but their rate per class is known |
| Token length | RoBERTa token count per utterance | 64 tokens covers at least 99 % of rows; the truncated rows listed |
| TF-IDF vocabulary from train only | Vectoriser fitted on train rows with `exclude_reason == ""`; its `fit_split` stored next to the pickle | Assertion on load |

## Stage 5 · Audio features (eGeMAPS, WavLM)

| Check | How | Pass |
|---|---|---|
| Shape and dtype | eGeMAPS: 88 finite floats per row; WavLM: 13 layers × 768 float16, finite, no all-zero row | Assertion over every row |
| One-to-one | Feature file index == manifest `utt_id` order; stored `utt_id` array compared with the manifest | Exact match |
| Cache isolation | Each cache file stores `split` and `extractor_version`; loader asserts both | Assertion |
| Scaler fit on train only | `scaler.pkl` metadata `fit_split == "train"`; mean of the scaled dev features is not 0 (it would be 0 only if dev were used to fit) | Assertion |
| Recomputation | Recompute features for 50 random rows from the WAV and compare with the cache (`allclose`, atol 1e-3 for float16) | 50 / 50 |
| Feature sanity against the EDA | Correlation of the eGeMAPS loudness feature with the EDA `rms_db`, and of its F0 feature with `f0_median_hz` | Pearson r above 0.8 for both; a low value means the feature extractor ran on the wrong files |
| Length policy | Clips longer than 10 s are centre-cropped before WavLM at evaluation and random-cropped at training; count of cropped rows per split logged | Logged |

## Stage 6 · Augmentation (train only)

| Check | How | Pass |
|---|---|---|
| Only the train loader augments | Build dev and test loaders and assert their transform list is empty | Assertion |
| Augmented audio is valid | 100 augmented samples: finite, same length policy, RMS within 10 dB of the original | 100 / 100 |
| Noise level is what the config says | Measure actual SNR of 20 augmented samples against the intended 10–30 dB | Within 2 dB |
| Visible sanity | Spectrogram before and after for 3 clips saved to `reports/figures/augmentation_examples.png` | Looked at once, referenced in the gate log |

## Stage 7 · Dataset loader and labels

| Check | How | Pass |
|---|---|---|
| Label mapping is fixed and saved | `configs/data.yaml` `emotions` list defines index ↔ name; the loader and every metric use it; no `LabelEncoder` fitted at run time | Unit test: `dataset[i].label == emotions.index(manifest.emotion[i])` for 200 random i |
| Item integrity | `__getitem__(i)` returns the audio, text and label of the same `utt_id`; test by returning the id and comparing with the manifest | 200 / 200 |
| Collation | Batch shapes, padding masks sum to the true lengths, no label padding value colliding with a class index | Unit test |
| Class weights from the right rows | Weights computed from train rows after exclusions; printed at start of training | Match a hand computation |
| Test is unreachable from training | `train.py` asserts no manifest path containing `test` is in its config; grep the training code for the string `test.csv` | Assertion passes; grep returns nothing |

## Stage 8 · Golden set and end-to-end regression

| Check | How | Pass |
|---|---|---|
| Golden set | 50 utterances (7 per class, plus 1) chosen once, listened to by hand, with gold text, ASR text, label and audio sha256 recorded in `data/manifests/golden.csv` | File exists and is reviewed |
| Pipeline regression | After any change to extraction, ASR or features, rerun the pipeline on the golden set; compare WAV sha256, ASR text and feature vectors with the stored ones | Identical (features `allclose`) |
| Smoke training | Train the TF-IDF + LR baseline from the processed text and the eGeMAPS + LR baseline from the processed audio; compare dev weighted F1 with the EDA-time values (TF-IDF 0.427) | Within 0.02; a large drop means the processed inputs differ from what the EDA saw |
| Shuffled-label control | Train the same baselines on permuted train labels | Dev macro F1 at chance; if not, the features carry the label through a leak |
| Reproducibility | Rerun `scripts/run_data_pipeline.sh` from the archive; compare manifest hashes and 100 feature rows | Identical |

## Stage 9 · Documentation

- A data card in the report: counts per split per class before and after exclusions, clean-test size, WER per split, decode failures, fragment counts, the duplicate list, the golden set.
- The gate log (`reports/gates.md`) quotes every check's output; no check is marked passed without pasted evidence.

## What "good and correct" means, in one line each

- Correct: every model input can be traced back to one official clip and one official label row, and recomputing it from the source gives the same bytes.
- Good: the inputs still carry the signal the EDA found (lexical cues, F0 and loudness differences), and nothing from dev or test touched a fitted object.
