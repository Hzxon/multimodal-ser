# Multimodal Speech Emotion Recognition — Final Project Plan

Pipeline: audio → ASR transcript → (audio encoder + text encoder) → fusion → emotion class.

This document is the working plan. Every phase ends with a **gate**: a checklist of
constraints that must pass before the next phase starts. Each gate item names the
check, the pass criterion, and the script or artifact that proves it. Gate results are
recorded in `reports/gates.md` with the command output pasted in, so the final report
can cite them.

---

## 0. Summary

| Item | Decision |
|---|---|
| Task | Utterance-level emotion classification from speech, using the audio signal and the ASR transcript of that audio |
| Primary dataset | MELD (Multimodal EmotionLines Dataset), 7 classes, public download, gold transcripts included |
| Stretch dataset | IEMOCAP 4-class (neutral, happy+excited, sad, angry), only if the USC license arrives before week 5 |
| ASR | OpenAI Whisper (`openai/whisper-small` for iteration, `openai/whisper-large-v3-turbo` for the final run), run once, outputs cached |
| Audio encoder | `microsoft/wavlm-base-plus` (frozen first, then partial fine-tune); eGeMAPS (openSMILE) for the interpretable baseline |
| Text encoder | `roberta-base`; TF-IDF + logistic regression for the baseline |
| Fusion | Late fusion (prob average) as baseline, gated concatenation and one cross-attention block as the main model |
| Primary metric | Weighted F1 on the MELD test split (the benchmark convention); macro F1 reported next to it because of class imbalance |
| Framework | Python 3.11, PyTorch 2.x, `transformers`, `torchaudio`, `opensmile`, `jiwer`, `captum`, `shap`, `scikit-learn` |
| Compute | One T4/L4-class GPU (Colab or Kaggle) or Apple MPS for small runs |
| Final deliverables | Repository, written report, gate log, notebooks for EDA/evaluation/XAI, Gradio demo (upload audio → transcript → emotion + explanation) |

Definition of done for the whole project: every gate in Sections 3.1–3.6 passes, the
multimodal model beats the best unimodal model on dev with a significant margin on test
(or the report explains why it does not), and the XAI section includes a faithfulness
test, not only pictures.

---

## 1. Scope and key decisions

### 1.1 Why MELD

- Public download, no license wait. IEMOCAP needs a signed request to USC, which can take days to weeks.
- Text carries emotion. Acted datasets such as CREMA-D (12 sentences) and RAVDESS (2 sentences) have no lexical signal, so a text branch would be meaningless.
- Gold transcripts exist, so Whisper output can be scored with WER, and the model can be evaluated on both ASR and gold transcripts.
- Fixed train/dev/test split, so results compare to published numbers.

Known MELD problems the plan must handle:

- Heavy class imbalance. Approximate train counts: neutral 4710, joy 1743, surprise 1205, anger 1109, sadness 683, disgust 271, fear 268 (verify in Phase 1; these are from memory).
- Audio is extracted from TV clips: laugh track, overlapping speech, music. Expect high WER.
- Speakers (the six main cast) appear in all splits. The split is dialogue-disjoint, not speaker-disjoint. This must be stated in the report; a speaker-independent claim is not possible on MELD.
- A few clips are known to be corrupt or unreadable. Log and drop them, report the count.
- Whisper's training data may include Friends subtitles. ASR quality on MELD may therefore be optimistic compared with unseen audio. State this as a caveat.

### 1.2 Transcript condition

The deployed pipeline never sees gold transcripts, so:

- Main condition: the text branch is trained and evaluated on **Whisper transcripts**.
- Upper-bound condition: the same model trained and evaluated on gold transcripts.
- Report both. The gap between them is the "ASR tax" and is itself a finding.

### 1.3 Non-goals

- No dialogue-context modelling (previous utterances). Utterance-level only. Context is listed as future work.
- No visual modality.
- No speaker diarization; MELD utterances are already segmented.

---

## 2. Repository layout and reproducibility

```
deepLearning_v2/
  PLAN.md
  README.md
  requirements.txt            # pinned versions
  configs/                    # one YAML per experiment
    data.yaml
    audio_baseline.yaml
    text_baseline.yaml
    fusion_gated.yaml
  data/
    raw/                      # untouched download (git-ignored)
    processed/                # wav16k/, transcripts/, features/ (git-ignored)
    manifests/                # train.csv dev.csv test.csv (committed; id, path, label, speaker, duration, gold_text, asr_text)
  src/
    data/      download.py, extract_audio.py, build_manifest.py, dataset.py
    asr/       transcribe.py
    features/  egemaps.py, ssl_embed.py, text_embed.py
    models/    audio_model.py, text_model.py, fusion.py
    train.py   evaluate.py   explain.py
  scripts/
    check_dataset.py          # Phase 1 gate
    check_leakage.py          # Phase 3/4 gate
    sanity_checks.py          # overfit-one-batch, shuffled-label run
    run_all.sh                # reproduces every number in the report
  notebooks/
    01_eda.ipynb  02_asr_quality.ipynb  03_results.ipynb  04_xai.ipynb
  reports/
    gates.md                  # gate log with pasted outputs
    figures/
  app/demo.py                 # Gradio
```

Rules:

- Every run reads one config file and writes to `runs/<name>/<seed>/` with `config.yaml`, `metrics.json`, `predictions.csv`, and the git commit hash.
- Seeds: 13, 42, 2024. Report mean ± std over the three.
- `torch.use_deterministic_algorithms(True)` where it does not break the encoders; otherwise document which ops are non-deterministic.
- Feature caches are keyed by `(split, extractor_name, extractor_version)`. A cache file never contains more than one split.
- The test manifest is read only by `evaluate.py`. `train.py` asserts that no path containing `test` is in its data config.

---

## 3. Phases and gates

### 3.1 Phase 1 — Dataset acquisition and manifest

Steps:

1. Download MELD raw (mp4 clips + CSV labels). Record URL, date, SHA-256 of the archives.
2. Extract mono 16 kHz WAV with ffmpeg. Log every ffmpeg failure.
3. Build `manifests/{train,dev,test}.csv` with: `utt_id`, `dialogue_id`, `utterance_id`, `speaker`, `emotion`, `sentiment`, `gold_text`, `wav_path`, `duration_s`, `sha256`.
4. Freeze the manifests in git. They are the single source of truth for the rest of the project.

Gate 1 (`scripts/check_dataset.py`):

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| D1 | License and attribution | MELD license permits academic use; citation added to README | README section |
| D2 | All clips decode | Every manifest row has a readable WAV; corrupt clips dropped and counted | Script prints dropped count per split |
| D3 | Label distribution measured | Table of counts per class per split, imbalance ratio (max/min) per split | `reports/figures/class_dist.png`, table in gates.md |
| D4 | Split integrity | `utt_id` sets are pairwise disjoint across train/dev/test; `dialogue_id` sets are pairwise disjoint | Script asserts, prints set sizes |
| D5 | Duplicate audio | No identical `sha256` across splits; identical hashes within a split listed | Script output |
| D6 | Speaker overlap documented | Speakers shared across splits listed with counts; report states the split is dialogue-disjoint, not speaker-disjoint | gates.md paragraph |
| D7 | No label in file names or paths | Paths contain only ids | Manual inspection, one line in gates.md |
| D8 | Duration range | Min, max, mean duration per split; clips above 15 s flagged for truncation policy | Histogram |

"Dataset already balanced" is **not** a gate, because MELD is not balanced and forcing
balance by discarding majority data throws away most of the training set. The balance
policy is in Section 5 and is checked at the model gate (M6, M7) instead.

### 3.2 Phase 2 — Exploratory data analysis

Questions to answer in `notebooks/01_eda.ipynb`, each with a figure or table:

Audio
- Duration distribution per class. Does length predict label (a shortcut)?
- Loudness (RMS) and pitch (F0 median) per class. Expected: anger and surprise louder, sadness quieter.
- Signal-to-noise estimate per clip; fraction of clips with laugh track or music.
- Spectrogram examples, two per class.

Text
- Token count per class. Length as a shortcut, same concern as duration.
- Top unigrams per class after stop-word removal; emotion lexicon hit rate (NRC lexicon) per class.
- Exact-duplicate utterances ("Oh my God!", "What?") and their label spread. These show the ceiling of text-only accuracy.

Cross-modal
- Speaker × emotion table. Do some speakers carry one emotion disproportionately?
- Sentiment vs. emotion agreement (MELD has both).

Gate 2:

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| E1 | Shortcut audit | Duration-only and length-only logistic-regression classifiers trained on train, evaluated on dev; their macro F1 recorded as the "shortcut floor" | Table |
| E2 | Class-conditional stats | Table of mean duration, RMS, F0, token count per class | Table |
| E3 | Noise estimate | Fraction of clips flagged noisy per split | Table |
| E4 | Findings written | Five-bullet EDA summary with design consequences (e.g., truncation length, need for class weights) | gates.md |

### 3.3 Phase 3 — ASR and feature engineering

Steps:

1. Transcribe every clip once with Whisper. Pin model id and library version. Store `asr_text` in the manifests with a `asr_model` column.
2. Compute WER against `gold_text` per split, overall and per class.
3. Audio features:
   - eGeMAPS (88 functionals) via openSMILE for the interpretable baseline.
   - WavLM-base-plus hidden states, all 13 layers, mean-pooled over time, cached as float16. The model learns a softmax over layer weights.
   - Mean/std scaler for eGeMAPS **fit on train only**, saved to `features/scaler.pkl`.
4. Text features:
   - TF-IDF (fit on train only) for the baseline.
   - RoBERTa token embeddings computed inside the model at training time (not cached), so fine-tuning is possible.
5. Augmentation (train only, applied on the fly, never pre-split): additive noise at random SNR 10–30 dB, speed perturbation 0.9–1.1, SpecAugment on the WavLM input features. Text: none in the main run; token dropout 10 % as an ablation.

Gate 3 (`scripts/check_leakage.py` plus `notebooks/02_asr_quality.ipynb`):

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| F1 | ASR reproducibility | Re-running transcription on 100 random clips gives identical text (temperature 0, fixed seed) | Script output |
| F2 | WER reported | WER per split and per class; the report explains why some classes have higher WER (shouting, overlap) | Table |
| F3 | Scalers fit on train only | `scaler.pkl` metadata records `fit_split == "train"`; test statistics not used anywhere | Assertion in script |
| F4 | No per-speaker normalization using test data | If speaker normalization is used, stats come from train utterances of that speaker only; otherwise per-utterance normalization | Code review note |
| F5 | Feature cache isolation | Each cache file holds one split; loader asserts the split name in the file matches the requested split | Assertion |
| F6 | Augmentation after split | Augmentation is a `Dataset.__getitem__` transform on train only; dev/test loaders have no transforms | Unit test |
| F7 | Encoder pretraining overlap | WavLM and RoBERTa pretraining corpora do not include MELD labels (WavLM: Libri-Light, GigaSpeech, VoxPopuli; RoBERTa: web text). If an emotion-pretrained encoder such as emotion2vec is tried, its fine-tuning sets are listed and MELD/IEMOCAP overlap is ruled out or the encoder is dropped | Paragraph in gates.md with sources |
| F8 | Feature sanity | eGeMAPS features have no NaN; WavLM embeddings have unit-scale norms and no all-zero rows | Script output |

### 3.4 Phase 4 — Model development

Build in this order. Each row must beat the one above it on dev, or the report explains why.

| # | Model | Purpose |
|---|---|---|
| 0 | Majority class | Floor |
| 1 | Duration + token-count LR | Shortcut floor (from E1) |
| 2 | TF-IDF + LR (ASR text) | Text baseline |
| 3 | eGeMAPS + LR / small MLP | Audio baseline, interpretable |
| 4 | RoBERTa-base fine-tuned (ASR text) | Strong text |
| 5 | WavLM-base-plus + layer-weighted mean pool + MLP (frozen, then top-4 layers unfrozen) | Strong audio |
| 6 | Late fusion: average of 4 and 5 probabilities, weight tuned on dev | Fusion baseline |
| 7 | Gated concatenation: `[h_a ; h_t ; g ⊙ h_a ; (1−g) ⊙ h_t]`, `g = σ(W[h_a;h_t])` | Main model, candidate A |
| 8 | One cross-attention block each way (text queries audio frames, audio queries text tokens), then concat | Main model, candidate B |

Training recipe (same for 4–8):

- Loss: cross-entropy with class weights `w_c = (N / (K · n_c))^0.5` (square-root inverse frequency). Ablate against unweighted CE and focal loss (γ = 2).
- Optimizer AdamW; lr 2e-5 for pretrained encoders, 1e-3 for heads; linear warm-up 10 %, cosine decay; weight decay 0.01.
- Batch 16, max audio 10 s (random crop on train, centre crop on eval; log how many clips are cropped), max text 64 tokens.
- Modality dropout 0.15 on the fusion models: with that probability one modality's embedding is zeroed during training. This prevents the text branch from dominating and is needed for the XAI modality-contribution analysis.
- Early stopping on **dev weighted F1**, patience 3, max 15 epochs. Checkpoint with the best dev score is the only one evaluated on test.
- Hyperparameter search: small grid on dev only (lr × class-weight scheme × modality dropout). At most 12 configs per model family.

Gate 4 (`scripts/sanity_checks.py` plus run logs):

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| M1 | Overfit one batch | A single batch of 16 reaches ≥ 99 % train accuracy within 200 steps for every model family | Log |
| M2 | Shuffled-label control | Training on randomly permuted train labels gives dev macro F1 within ±3 points of chance (≈ 14 % for 7 classes with majority bias measured) | Log. A score far above chance means leakage or a shortcut |
| M3 | Test never used for selection | `train.py` never opens the test manifest (assertion); the hyperparameter table lists dev scores only | Code + table |
| M4 | Dev used once per config | Each config has exactly one dev evaluation per epoch; no repeated manual peeking documented | Run log |
| M5 | Seeds | Every reported model has three seeds; std reported | metrics.json ×3 |
| M6 | Minority recall | Every class has test recall > 0 for the final model; fear and disgust recall reported explicitly | Confusion matrix |
| M7 | Class-weight ablation | Weighted CE vs. unweighted CE compared on dev macro F1 and weighted F1; the chosen scheme justified | Table |
| M8 | Unimodal → multimodal gain | Best fusion model ≥ best unimodal model on dev weighted F1. If not, the report analyses why (text dominance, noisy audio) and still reports it | Table |
| M9 | Train/dev gap | Train F1 − dev F1 < 15 points for the chosen checkpoint; otherwise add regularisation and re-run | Curves |
| M10 | Reproducible | `scripts/run_all.sh` regenerates every number in the results table from the manifests | Script exists and ran once end-to-end |

### 3.5 Phase 5 — Evaluation

Protocol:

- Test set evaluated once per final model (3 seeds). No further changes after this.
- Metrics: weighted F1 (primary), macro F1, accuracy, per-class precision/recall/F1, confusion matrix (row-normalised).
- Uncertainty: 95 % bootstrap confidence intervals over test utterances (1000 resamples) for weighted and macro F1.
- Significance: McNemar's test between the fusion model and the best unimodal model on paired test predictions. Report p-value.
- Transcript conditions: 2 × 2 table (train on gold/ASR × test on gold/ASR) for the text and fusion models.
- Robustness: test with added noise at 20 dB and 10 dB SNR; report degradation per modality and for fusion.
- Comparison to literature: published MELD weighted F1 is roughly 62–65 for text-only RoBERTa-class models, 40–48 for audio-only, and 63–67 for utterance-level multimodal models without dialogue context (ranges from memory; cite exact papers in the report and update the numbers).
- Error analysis: 30 misclassified test utterances sampled stratified by true class; each annotated with (a) ASR error present, (b) modality disagreement, (c) plausibly ambiguous label. Summarised as a table.

Gate 5:

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| V1 | Single test pass | Test metrics computed once per seed; timestamped in `runs/` | File timestamps |
| V2 | Confidence intervals | Every headline number has a CI | Results table |
| V3 | Significance | McNemar p-value for fusion vs. best unimodal reported | Table |
| V4 | Per-class reporting | Per-class F1 and confusion matrix in the report | Figure |
| V5 | ASR tax | 2 × 2 transcript-condition table present | Table |
| V6 | Robustness | Noise-degradation table present | Table |
| V7 | Literature comparison | At least three cited results with the same split and metric | Table with citations |
| V8 | Error analysis | 30 annotated errors with category counts | Table |

### 3.6 Phase 6 — Explainable AI

Three levels, each with a faithfulness check.

Global (what the model uses in general)
- eGeMAPS baseline: SHAP values → ranked acoustic features per class (expected: loudness and F0 range for anger/surprise, low loudness and slow rate for sadness). This connects the deep model to known phonetics.
- WavLM layer weights: which layers the audio head relies on (lower layers = acoustic, upper = phonetic/semantic).
- Text: aggregated Integrated Gradients token attributions over the test set → top-20 tokens per class.

Local (why this utterance got this label)
- Text: Integrated Gradients (Captum) on RoBERTa input embeddings, visualised as token heat-maps.
- Audio: time-segment occlusion. Mask 250 ms windows (replace with silence), measure the drop in the predicted-class probability, plot as a saliency bar under the spectrogram. Grad-CAM is used only if a CNN-on-spectrogram ablation is added; it has no meaning for the eGeMAPS MLP.
- Fusion: the gate value `g` (model 7) or cross-attention weights (model 8) as a per-utterance modality-contribution score, plus a modality-ablation score: Δ in predicted probability when zeroing each modality.

Case studies
- Five utterances where audio and text branches disagree and fusion is right.
- Five where fusion is wrong, with the explanation showing why (e.g., ASR dropped the key word; laugh track dominated the audio).
- One sarcasm example: positive text, negative audio.

Gate 6:

| ID | Check | Pass criterion | Evidence |
|---|---|---|---|
| X1 | Faithfulness, text | Deletion curve: removing the top-k attributed tokens lowers the predicted-class probability more than removing k random tokens, averaged over 200 test utterances (AOPC). Reported for k = 1, 3, 5 | Plot |
| X2 | Faithfulness, audio | Same deletion test with occluded 250 ms segments vs. random segments | Plot |
| X3 | Sanity, parameter randomisation | Attributions from a randomly initialised model have Spearman correlation < 0.3 with attributions from the trained model (Adebayo et al. sanity check) | Number |
| X4 | Modality contribution | Distribution of the per-utterance contribution score per class; report which classes are audio-driven vs. text-driven | Figure |
| X5 | Global audio features match phonetics | Top-5 SHAP eGeMAPS features per class listed and compared against the EDA class-conditional stats (E2) | Table |
| X6 | Case studies | At least 11 cases as listed above, each with both modality explanations | Figures |

### 3.7 Phase 7 — Report and demo

- Report sections mirror Sections 3.1–3.6; each section ends with its gate table copied from `reports/gates.md`.
- Demo: `app/demo.py` (Gradio). Input microphone or file → Whisper transcript → emotion probabilities → token heat-map and audio saliency bar. Runs on CPU with whisper-small and the cached encoders.
- README has the one-command reproduction (`bash scripts/run_all.sh`) and expected runtime.

---

## 4. Leakage threat model

Each row is a way this specific project could leak, and the control that prevents it. The gate IDs refer to Section 3.

| Threat | Control | Gate |
|---|---|---|
| Same utterance or dialogue in two splits | Use MELD's official split; assert disjoint `utt_id` and `dialogue_id` sets | D4 |
| Duplicate audio across splits | SHA-256 comparison | D5 |
| Scaler, TF-IDF vocabulary, or class weights computed on dev/test | Fit on train only; metadata records fit split | F3 |
| Per-speaker normalisation with test statistics | Per-utterance normalisation, or speaker stats from train only | F4 |
| Feature cache mixing splits | One split per cache file, asserted on load | F5 |
| Augmentation before splitting, creating near-duplicates across splits | Augmentation is an on-the-fly train transform | F6 |
| Pretrained encoder saw MELD labels | Use self-supervised encoders; document pretraining corpora; reject emotion-pretrained encoders with overlap | F7 |
| Early stopping or hyperparameter selection on test | Dev only; `train.py` cannot open the test manifest | M3, M4 |
| Repeated test evaluation ("test-set tuning") | Test evaluated once per seed, timestamped | V1 |
| Label encoded in file name (true for CREMA-D, not MELD) | Paths contain ids only | D7 |
| Shortcut features (duration, length, speaker) | Shortcut-floor models and shuffled-label control | E1, M2 |
| Speaker identity carrying label | Cannot be removed on MELD; documented as a limitation; speaker × emotion table in EDA | D6 |
| Whisper memorised Friends subtitles | Cannot be controlled; stated as a caveat; gold-transcript condition bounds its effect | V5 |

---

## 5. Class balance policy

Measured first (D3), then handled, never faked.

- Do not undersample majority classes for the main model; it discards most of the data. Run one undersampled ablation only to show the cost.
- Do not oversample by copying audio; copies inflate train scores and make the shuffled-label control harder to interpret.
- Handle imbalance with square-root inverse-frequency class weights in the loss (ablate: unweighted, inverse-frequency, focal).
- Report macro F1 next to weighted F1 in every table, and per-class recall for the two smallest classes (fear, disgust).
- Model selection uses dev weighted F1 (benchmark convention). A model whose macro F1 is lower than the unweighted baseline's is rejected even if its weighted F1 is higher.
- Gate: every class has non-zero test recall (M6).

---

## 6. Expected numbers (to be replaced by measured ones)

| Model | Dev weighted F1 (expected) | Note |
|---|---|---|
| Majority class | ~ 30 | neutral share of dev |
| TF-IDF + LR (ASR) | 50–55 | |
| eGeMAPS + MLP | 35–42 | |
| RoBERTa (ASR) | 58–63 | gold transcripts add 1–3 points |
| WavLM + MLP | 42–48 | |
| Late fusion | 60–64 | |
| Gated / cross-attention fusion | 62–66 | target: ≥ +1.5 over RoBERTa with p < 0.05 |

These are rough expectations from the literature, from memory. They exist so that a
result far outside the range triggers investigation (M2, M8), not so the project is
judged against them.

---

## 7. Timeline (8 weeks)

| Week | Work | Gate |
|---|---|---|
| 1 | Repo, environment, download, ffmpeg extraction, manifests | Gate 1 |
| 2 | EDA notebook, shortcut floors, Whisper transcription + WER | Gate 2, F1–F2 |
| 3 | eGeMAPS, WavLM caches, scalers, leakage script, baselines 0–3 | Gate 3 |
| 4 | RoBERTa and WavLM models (4, 5), sanity checks | M1–M5 |
| 5 | Fusion models (6–8), class-weight ablation, hyperparameter grid | Gate 4 |
| 6 | Test evaluation, CIs, McNemar, robustness, error analysis | Gate 5 |
| 7 | XAI: attributions, faithfulness tests, case studies | Gate 6 |
| 8 | Report, demo, `run_all.sh` end-to-end reproduction | Phase 7 |

If IEMOCAP access arrives by week 5, add a leave-one-session-out 4-class run in week 6 as
the speaker-independent complement to MELD. Otherwise mention it as future work.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Whisper WER on MELD is very high (> 40 %) and the text branch collapses | Medium | Use `large-v3-turbo` for the final transcription; report the gold-transcript upper bound; this is a legitimate finding about the pipeline |
| Fusion does not beat text-only | Medium | Modality dropout, gated fusion, noise-robustness test where audio helps; report honestly with McNemar |
| GPU time for WavLM fine-tuning | Medium | Cache frozen embeddings first; unfreeze only the top 4 layers; use fp16 |
| Corrupt MELD clips | Certain, small | Drop and count (D2) |
| Speaker leakage criticism | Certain | Documented limitation; IEMOCAP stretch run if license arrives |
| XAI attributions are pretty but unfaithful | Medium | Gates X1–X3 are mandatory |

---

## 9. Generic rubric mapping

| Rubric item (typical DL course) | Where it is satisfied |
|---|---|
| Problem definition and motivation | Section 0, report introduction |
| Dataset description and preprocessing | Phase 1, Gate 1 |
| Exploratory analysis with insights | Phase 2, Gate 2 |
| Feature engineering with justification | Phase 3, Gate 3 |
| Correct train/validation/test methodology, no leakage | Section 4, Gates D4–D7, F3–F7, M3–M4, V1 |
| Baselines and ablations | Phase 4 model ladder, M7, M8 |
| Model architecture and training details | Section 3.4 recipe |
| Evaluation with appropriate metrics and uncertainty | Phase 5, Gates V1–V8 |
| Explainability | Phase 6, Gates X1–X6 |
| Reproducibility | Section 2, M10 |
| Report quality and demo | Phase 7 |

When the actual course rubric is available, replace this table with a row per rubric
line and the gate IDs that satisfy it.
