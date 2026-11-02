# Gate 3, stages 0-2: raw inventory, extraction, manifests

| ID | Status | Criterion | Evidence |
|---|---|---|---|
| S1.1 | PASS | Every WAV is 16000 Hz, mono, PCM_16 | 13706 files checked; wrong format: [] (0) |
| S1.2 | PASS | WAV duration equals the source audio stream duration within 50 ms (AAC frame padding is about 21 ms) | max diff 0.022 s; over 50 ms: 0 {} |
| S1.3 | PASS | Mono downmix is within 6 dB of the loudest source channel (sample of 200) | clips where the mix is more than 6 dB quieter than a source channel: [] (0) |
| S1.4 | PASS | No empty/constant WAVs and no clipping (sample of 200) | constant: []; clipped>0.1%: []; RMS below -40 dBFS (information): 5 [('test_dia213_utt7', -49.0), ('test_dia198_utt5', -41.0), ('train_dia410_utt7', -43.4), ('train_dia668_utt0', -61.1), ('train_dia946_utt12', -46.8)] |
| S1.5 | PASS | Re-extracting 20 random clips reproduces the stored samples and content hash | mismatches: [] |
| S2.1 | PASS | Manifest row counts equal the label CSVs | {'train': 9989, 'dev': 1109, 'test': 2610} |
| S2.2 | PASS | utt_id unique across splits; (dialogue_id, utterance_id) unique within split | ok |
| S2.3 | PASS | Schema: required columns non-null; duration_s null exactly where has_audio is False; labels in the class list | nulls {'emotion': 0, 'gold_text': 0, 'speaker': 0, 'split': 0, 'exclude_reason': 0, 'clean_subset': 0, 'has_audio': 0}; duration-null pattern ok True |
| S2.4 | PASS | Exclusions only in train; dev/test complete | {'train': {'kept': 9639, 'fragment_lt_0.5s': 306, 'shared_audio_group': 38, 'timestamp_duplicate_of': 5, 'no_audio': 1}, 'dev': {'kept': 1109}, 'test': {'kept': 2610}} |
| S2.5 | PASS | clean_subset flag stored once in the manifests | {'train': 9637, 'dev': 1066, 'test': 2504} |
| S2.6 | PASS | 50 random rows: wav path ids equal row ids and the content hash recomputes | bad: [] |
| S2.7 | PASS | No identical audio across splits among non-excluded rows | cross-split identical groups: 0; within-split identical groups: 5 |
| S2.8 | PASS | Manifest hashes recorded | {'train': '6e5a3dc1480f10b6', 'dev': 'dc7f71f1a28f536c', 'test': 'f5909e120e0bb5c1'} |
