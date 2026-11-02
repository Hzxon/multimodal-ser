# MELD usability audit

Date: 2026-11-02. Scope: is MELD fit for this project (audio → ASR transcript → audio + text → emotion class), and under what conditions.

Evidence used:

1. Our own checks on the official label CSVs (13 708 rows): `scripts/check_dataset.py`, `src/eda.py`, results in `reports/gate1.md`, `reports/eda_summary.md`.
2. An audio probe on the dev split only, using a third-party 16 kHz FLAC mirror of the official clips (Hugging Face `ajyy/MELD_audio`, 1 112 files, 121 MB), analysed with `scripts/probe_audio_dir.py`; per-clip numbers in `reports/probe_dev_mirror_audio_stats.csv`. The official 10.9 GB archive is not downloaded yet, so train/test audio findings are extrapolated from the label CSVs and must be confirmed in Gate 1.
3. The MELD paper, three later papers that measured MELD label quality or audio/text ceilings, and the issue tracker of the authors' repository (links in Section 8).

## 1. Verdict

MELD is usable for this project, with four conditions. Without them the project will either leak, over-claim, or chase an audio branch that cannot work.

| # | Condition | Why |
|---|---|---|
| 1 | Treat the labels as noisy and report every number with a confidence interval; add a secondary evaluation on an independently corrected label set | Fleiss' κ of the original annotation is 0.43; an independent re-annotation of 200 utterances also gets κ = 0.43; a 2026 relabeling changed 24.2 % of the labels. Single-label accuracy on MELD has a ceiling well below 100 % that no model can pass. |
| 2 | Apply the exclusion rules in Section 6 (cross-split duplicates, mis-cut fragments, the corrupt clip) and keep dev/test otherwise intact | The official split contains two exact train/test duplicates, about 3 % of clips are speech fragments under 0.5 s, two "utterances" are 4 and 5 minutes long, and one train clip is unreadable. |
| 3 | Expect a small multimodal gain and design the evaluation to detect it | Published utterance-level numbers: audio-only 39–49 weighted F1, ASR-transcript text 51.5, gold-transcript text 55–61, early fusion without context 55.5. The fusion gain over text is a few points, so three seeds, bootstrap CIs and a paired test are mandatory, not optional. |
| 4 | State that the result is speaker-dependent and sitcom-domain | The six main cast members speak 83–86 % of every split; audio carries a laugh track. Nothing learned here transfers to a clean-speech claim without a second dataset. |

Why MELD rather than the alternatives: it is the only public, no-license-wait, English dataset where the transcript carries emotion, which the text branch needs, and where gold transcripts exist to measure the ASR tax. Section 7 compares the alternatives.

## 2. Access and licence

- Repository `declare-lab/MELD` is GPL-3.0 (GitHub API, checked today). The authors have not answered whether GPL applies to trained weights (issue 19, open since 2020). For a course project that publishes code and a report, GPL is not a problem. State the licence in the report.
- The raw archive is 10.9 GB of mp4 (audio+video) from the authors' Hugging Face repo or the UMich mirror. Download speed from here was about 1 MB/s, so plan 3 hours. No other official distribution exists; the audio-only mirrors on Hugging Face are third-party re-encodings (the one probed here is 16 kHz mono FLAC, 1.5 GB for all splits).

## 3. Label quality

Facts:

| Source | Finding |
|---|---|
| MELD paper (Poria et al., 2019) | Three graduate-student annotators per utterance, majority vote, Fleiss' κ = 0.43 overall; utterances with three different votes were dropped. EmotionLines (the text-only parent) had κ = 0.34; IEMOCAP 0.40. |
| "Exposing Weaknesses in Emotion Recognition in Conversations" (arXiv 2609.05806) | 200 MELD utterances re-annotated by four people; average pairwise Cohen's κ = 0.43. High-agreement cases are dominated by neutral; fear, disgust and sadness sit mostly in the 1/5–3/5 agreement range. Errors concentrate on negations, exclamations and interjections. |
| EmoS benchmark (arXiv 2605.08847) | Re-aggregated the 7-class labels on a 5 000-clip MELD subset: 1 031 labels (24.2 %) changed; only 1 993 of 5 000 corrected labels have confidence ≥ 0.7. Found clips "duplicated but assigned inconsistent emotion labels". Released corrected CSVs (`newdataset-MELD/`, fields `basic_original`, `basic_corrected`). |
| Our checks (`gate1.md` L1–L5) | Structurally clean: all rows in the 7 classes, sentiment consistent with emotion for every row, no empty transcripts, no mojibake. |

Consequences:

- A model that agrees with the gold label as often as a second human (κ ≈ 0.43) is at the ceiling. Weighted F1 in the mid-60s with dialogue context is the best published; without context, high 50s to low 60s.
- The minority classes are both rare (fear: 268 train, 50 test) and the least reliably labelled. Fear/disgust F1 will have a bootstrap CI of roughly ±0.10 on test; do not interpret small differences there.
- Mitigation adopted: report macro and weighted F1 with 95 % bootstrap CIs; add the EmoS-corrected labels as a secondary evaluation set (requires obtaining their CSV and mapping clip ids), which shows whether "errors" are label noise.

## 4. Split integrity and clip segmentation

Facts from the label CSVs (all splits):

- Two train utterances are literally repeated in test with the same speaker, text and timestamps: `train_dia12_utt7/8` = `test_dia85_utt0/1`. Three more train rows share exact timestamps with a dev/test row but have different text (`train_dia53_utt0`/`dev_dia52_utt2`, `train_dia433_utt1`/`dev_dia77_utt0`, `train_dia887_utt0`/`test_dia117_utt8`). No whole dialogue is duplicated across splits. Episodes are shared (train/test share 142 episodes) because MELD splits by dialogue.
- Clip boundaries come from subtitle timestamps. In the dev probe, every one of 1 108 matched clips has exactly the length of its CSV timestamp span (0 clips deviate by more than 1 s). The CSV span is therefore the clip length for all splits, and the timestamp defects below are audio defects:

| Defect | train | dev | test |
|---|---|---|---|
| Clip shorter than 0.5 s (speech fragment, often a full sentence in the transcript) | 283 (2.8 %) | 34 (3.1 %) | 89 (3.4 %) |
| Clip shorter than 1.0 s | 912 (9.1 %) | 105 (9.5 %) | 269 (10.3 %) |
| Clip longer than 20 s | 7 | 2 | 2 |
| Extreme mis-cuts | `dia309_utt0` 41 s, transcript contains a scene description | | `dia220_utt0` 235 s ("What's that smell?"), `dia38_utt4` 305 s |

Fragments under 0.5 s are concentrated in neutral, joy, surprise and anger (short interjections: "Hi!", "Hey!", "What?"), so excluding them shifts the class balance slightly toward the long-utterance classes sadness and disgust.

- Dev probe: `dia84_utt0` (Monica, "Oh, did you catch him?!") and `dia84_utt1` (Ross, "Wh-what did he say?!") have byte-identical audio. The same segment was cut for two utterances. Issues 9 and 30 on the repository report the alignment as "pretty bad" in general; the authors did not fix it and do not distribute the source episodes.
- Dev probe: `dev_dia110_utt7` has no clip (confirmed by issues 22 and 27); four clips (`dia49_utt4/5`, `dia66_utt9/10`) exist without labels. Expect the same pattern in train/test.
- Train `dia125_utt3.mp4` is unreadable (issues 34 and 40). Issues 47 and 49 report test clips with no audio track; the official extraction will show whether that is a decoder problem or real silence (ffmpeg failures are logged and counted in Gate D2).

Consequence: with dialogue-level splitting and shared cast, MELD is dialogue-disjoint only. Speaker-independent generalisation cannot be claimed.

## 5. Audio quality and the audio signal

Facts:

- Laugh track: the EmoS authors call MELD "heavily contaminated by canned laughter" and keep only 5 000 of 13 708 clips after removing sub-1 s segments and low-SNR segments. Our dev probe, which cannot detect laughter directly, finds 72 clips (6.5 %) with less than 10 dB of frame-level dynamic range (steady background, median length 1.2 s) and 34 clips (3 %) with RMS below −40 dBFS, 13 of them fragments under 1 s. No clipping (0 clips above 0.1 % clipped samples). All probe clips are 16 kHz mono; the official mp4 audio will be converted to the same format by `src/data/extract_audio.py`, and its original sample rate and channel count are recorded in Gate 1.
- The acoustic class signal exists despite the noise. Dev per-class means from the probe: median F0 for anger 261 Hz, fear 255 Hz, surprise 244 Hz versus neutral 208 Hz; sadness is the quietest (−32.0 dB RMS versus −29 to −31 for the others) and longest (4.0 s versus 2.4–3.5 s).
- Generic denoising hurts: applying DEMUCS before SSL feature extraction lowered weighted F1 at almost every layer in the 2025 lexical-features study, because it strips non-verbal vocalisations. Do not denoise.
- Published audio-only ceilings on the official test split (weighted F1): bcLSTM audio 39.1 and DialogueRNN audio 41.8 (MELD paper); SpeechFormer++ 47.0; DWFormer 48.5; TF-Mamba with WavLM-large 48.5 (WavLM-base 42.5, HuBERT-base 37.7, Whisper-large encoder 40.2); XLS-R-2B layer 26 + MLP 49.3. Audio alone will not beat text; it has to earn its place through fusion gains on specific classes (anger, surprise) and on the generic short lines.

## 6. Text signal and ASR

Facts:

- Published text-only ceilings on the official test split (weighted F1): text-CNN without context 55.0 (MELD paper); Whisper transcripts + DeBERTa layer 19 + MLP 51.5 versus the same pipeline on gold transcripts 60.9 (so the ASR tax on MELD is about 9 points); RoBERTa with dialogue context 65.6 (EmoBERTa).
- Our 10-minute baseline, TF-IDF (1–2-grams) + logistic regression on gold text, train → dev: weighted F1 0.427, macro F1 0.281 (0.325 with balanced class weights), accuracy 0.485. Per-class F1 with balanced weights: neutral 0.54, surprise 0.46, joy 0.37, sadness 0.30, fear 0.26, anger 0.25, disgust 0.11.
- 171 of 1 109 dev utterances (15 %) are short generic lines that also occur verbatim in train ("hey", "yeah", "what", "okay"); predicting the train majority label for them gives only 61 % on that subset. "what" is surprise 73 % of the time, "no" is anger 23 % of the time. These are the cases where audio can add information.
- Class-specific vocabulary is sensible: sorry, lost, died → sadness; god, wow, whoa, really → surprise; great, amazing, glad → joy; stupid, hell, mad, crap → anger; scared, hurt → fear.

Consequence for the pipeline: Whisper will cost roughly 9 weighted-F1 points on the text branch. The project must report the gold-transcript condition as the upper bound and the ASR condition as the deployed number, as planned. Published multimodal utterance-level numbers without context (cMKL 55.5) are below the best text-only-without-context number from a modern encoder, so the honest target is "fusion beats our own text branch by a statistically significant margin", not a leaderboard position.

## 7. Alternatives considered

Numbers below are from memory of each dataset's documentation and are marked as such; verify before citing.

| Dataset | Text carries emotion | Transcripts | Audio | Licence / access | Verdict for this project |
|---|---|---|---|---|---|
| MELD | Yes | Gold, with ASR tax measurable | TV audio, laugh track, 3 % fragments | GPL-3.0, instant | Chosen |
| IEMOCAP | Yes (scripted and improvised dyads) | Gold | Clean studio, 10 speakers, speaker-independent CV possible | Signed request to USC, days to weeks | Stretch dataset if the licence arrives by week 5; gives the speaker-independent result MELD cannot |
| MSP-Podcast | Yes (naturalistic) | ASR-based in recent releases | Podcast speech, large (over 100 k segments, from memory) | Academic licence agreement | Too large for the course timeline on one GPU; licence wait |
| CMU-MOSEI | Yes | Gold | YouTube monologues, 23 k sentences | Free via SDK | Labels are multi-label intensities, not a single class; would change the task |
| CREMA-D | No (12 fixed sentences) | Trivial | Clean, acted, 91 actors | Free | Text branch meaningless |
| RAVDESS | No (2 sentences) | Trivial | Clean, acted, 24 actors | CC BY-NC-SA | Text branch meaningless |

## 8. Data policy adopted (to be ratified in Gate 1 on the official archive)

| Rule | Scope | Effect |
|---|---|---|
| Exclude the five train rows that duplicate a dev/test row by timestamp, and any train row whose audio sha256 equals a dev/test clip | train | −5 rows (already in the manifests as `exclude_reason`) |
| Exclude clips shorter than 0.5 s from training | train | −283 rows (2.8 %); the audio branch cannot learn from 0.1 s fragments and the text is a one-word interjection |
| Exclude the unreadable clip `train_dia125_utt3` and any clip ffmpeg cannot decode | all, logged | count reported in D2 |
| Keep dev and test complete for the headline numbers, and additionally report a "clean test" subset (clips ≥ 0.5 s and ≤ 20 s) | dev, test | headline stays comparable to the literature; the clean subset shows the real audio contribution |
| Truncate training audio to 10 s, centre-crop at evaluation; the 4 and 5 minute test clips are evaluated on their first 10 s and listed in the error analysis | all | removes the two mis-cut giants from influencing anything |
| Do not denoise | all | see Section 5 |
| Secondary evaluation on the EmoS-corrected labels where ids can be matched | test subset | separates label noise from model error |
| Report weighted F1 (benchmark convention), macro F1, per-class F1 with 95 % bootstrap CIs, three seeds, McNemar fusion vs text | all | required by Sections 3 and 6 |

## 9. Open items, resolved on the official archive (2026-11-03)

1. D2: one undecodable clip (`train/dia125_utt3.mp4`, no moov atom); no test clip lacks an audio track. The "no audio in test" reports on GitHub are explained by 6-channel (5.1) AAC that some players cannot decode; 2-channel and 6-channel clips occur in every split and ffmpeg downmixes both without cancellation (checked on 200 clips).
2. D5: the two train/test repeats are byte-identical audio, and so is `train_dia887_utt0` = `test_dia117_utt8`; the two train/dev timestamp collisions are different audio. In addition, 19 train pairs, 1 dev pair and 4 test pairs share identical audio within a split (the same segment cut for two utterances).
3. The test folder contains corrected cuts for 132 utterances under the prefix `final_videos_test`; preferring them reduced identical-audio pairs in test from 22 to 4 and removed every timestamp disagreement except two mis-cut giants.
4. Fragments and silence measured on all splits: clips under 0.5 s are 309 train / 38 dev / 85 test; near-silent (below −40 dBFS) 256 train / 76 test; low dynamic range (laugh track) 403 train / 140 test.
5. The archive's own CSVs are cp1252/double-encoded copies of the GitHub CSVs with two wrong train rows; the GitHub CSVs are the label source.
6. Still open: Whisper WER per split and per class (Phase 3).

## 10. Sources

- Poria et al., MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations, ACL 2019. https://arxiv.org/abs/1810.02508 (κ = 0.43, Table 8 statistics, Table 11 baselines).
- Exposing Weaknesses in Emotion Recognition in Conversations. https://arxiv.org/abs/2609.05806 (re-annotation, κ = 0.43 on 200 MELD utterances).
- EmoS: A High-Fidelity Multimodal Benchmark for Fine-grained Streaming Emotional Understanding. https://arxiv.org/abs/2605.08847 (24.2 % labels changed, laugh-track contamination, duplicated clips, corrected CSVs).
- Combei, On the Contribution of Lexical Features to Speech Emotion Recognition. https://arxiv.org/abs/2509.05634 (Whisper text 51.5 vs gold 60.9 vs acoustic 49.3; denoising hurts).
- Temporal-Frequency State Space Duality (TF-Mamba). https://arxiv.org/abs/2412.16904 (Table II: WavLM/HuBERT/Whisper audio-only weighted F1 on MELD).
- EmoBERTa. https://arxiv.org/abs/2108.12009 (65.6 weighted F1 with context).
- MELD repository and issues. https://github.com/declare-lab/MELD (licence; issues 9, 19, 20, 22, 27, 30, 34, 40, 47, 49).
- Dev-split audio mirror used for the probe. https://huggingface.co/datasets/ajyy/MELD_audio (third-party re-encoding; not used for training).
