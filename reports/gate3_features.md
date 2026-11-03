# Gate 3, feature checks

Command: `python scripts/check_leakage.py --manifest-dir None --feature-root data/processed/features --n 50`

| ID | Status | Criterion | Evidence |
|---|---|---|---|
| F3 | PASS | egemaps_v02-opensmile2.6.0: scaler fit on train only, n_fit matches, dev mean is not 0 | fit_split=train; n_fit=9639 vs non-excluded train rows=9639 (cache rows 9988); scaler mean equals mean of non-excluded train rows: True; max \|mean\| of scaled train = 2.99e-07; scaled dev mean: max \|mean\| = 0.203, columns within 1e-3 of 0: 1/88 |
| F3-tfidf | PASS | tfidf_gold_gold: fit_split == train, n_fit matches, vocabulary equals a refit on non-excluded train text | fit_split=train, n_fit=9639 vs 9639, vocab=10813, refit vocabulary identical: True |
| F4 | PASS | no per-speaker statistics in feature or model code (code-review note) | 0 mentions of 'speaker' in src/features, src/models, none combined with a statistic or scaler; normalisation is per feature over train rows only |
| F5 | PASS | every cache file loads with its own split and raises AssertionError for the other two | 6/6 files OK |
| F5-1to1 | PASS | cache utt_id array equals the manifest utt_id array (rows with audio, same order) | 6/6 caches match |
| F7 | PASS | encoder pretraining corpora contain no MELD labels (documented) | WavLM-base-plus is pretrained on Libri-Light (60k h), GigaSpeech (10k h) and VoxPopuli (24k h), 94k hours of unlabelled English speech (Chen et al., WavLM, arXiv:2110.13900). RoBERTa-base is pretrained on BookCorpus, English Wikipedia, CC-News, OpenWebText and Stories (Liu et al., arXiv:1907.11692). Neither corpus carries MELD emotion labels, and both encoders are self-supervised, so no label leakage through pretraining. Caveat (PLAN.md 1.1): Whisper's training data may include Friends subtitles, so ASR quality on MELD can be optimistic; the gold-transcript condition bounds that effect. Pretraining sets were not checked by hashing against MELD clips; this is a documentation-level check. |
| F8-egemaps | PASS | eGeMAPS: 88 finite values in every row | train: (9988, 88) float32; dev: (1108, 88) float32; test: (2610, 88) float32 |
| F8-wavlm | PASS | WavLM: (N, 13, 768) float16, finite, no all-zero vector; per-layer L2 norm reported (sanity) | train: (9988, 13, 768) float16; dev: (1108, 13, 768) float16; test: (2610, 13, 768) float16 |
| S5-egemaps-recompute | PASS | eGeMAPS recomputed from the WAV (opensmile process_file) for 50 random rows per split, allclose atol 1e-3 rtol 1e-4 | 150/150 rows match, max abs diff 0.00e+00 |
| S5-wavlm-recompute | PASS | WavLM recomputed from the WAV (device mps, original batch composition, batch size [1]) for 10 random rows per split, allclose atol 2e-2 rtol 1e-2 on float16, same frame count | 30/30 rows match, max abs diff 0.00e+00 |
| S5-eda-corr | PASS | Pearson r > 0.8 between log eGeMAPS Loudness_sma3_amean and rms_db, and F0semitone_amean and the EDA F0 median in semitones | files ['audio_stats_dev.csv', 'audio_stats_test.csv', 'audio_stats_train.csv']; log loudness vs rms_db: r=0.903 (Spearman 0.895, n=13706); F0 semitone vs semitone(f0_median_hz): r=0.803 (Spearman 0.846, n=13443 voiced) |
| S5-crop | PASS | clips longer than the max audio length are centre-cropped; count per split logged | train: 212/9988 cropped (2.1%) [v1-L13-mean-10s]; dev: 23/1108 cropped (2.1%) [v1-L13-mean-10s]; test: 48/2610 cropped (1.8%) [v1-L13-mean-10s] |

## Details

### F4

```
grep -rni speaker src/features, src/models
(no matches)
```

### F8-wavlm

```
train L2 norm per layer (mean +- std over rows):
  L0:2.5+-0.4  L1:2.7+-0.3  L2:2.7+-0.5  L3:2.5+-0.5  L4:2.6+-0.6  L5:2.5+-0.6  L6:2.6+-0.7  L7:2.7+-0.7  L8:2.7+-0.7  L9:3.0+-0.7  L10:3.2+-0.7  L11:4.2+-0.7  L12:2.2+-0.3
dev L2 norm per layer (mean +- std over rows):
  L0:2.5+-0.4  L1:2.6+-0.3  L2:2.8+-0.5  L3:2.6+-0.5  L4:2.6+-0.6  L5:2.6+-0.6  L6:2.6+-0.7  L7:2.7+-0.7  L8:2.8+-0.7  L9:3.0+-0.7  L10:3.3+-0.8  L11:4.3+-0.7  L12:2.2+-0.3
test L2 norm per layer (mean +- std over rows):
  L0:2.5+-0.4  L1:2.7+-0.3  L2:2.7+-0.5  L3:2.5+-0.5  L4:2.6+-0.6  L5:2.5+-0.6  L6:2.6+-0.7  L7:2.7+-0.8  L8:2.7+-0.7  L9:3.0+-0.7  L10:3.3+-0.8  L11:4.2+-0.7  L12:2.2+-0.4
```

