# Gate 1: dataset checks

Manifests: `data/manifests`; audio columns present: True

| ID | Status | Criterion | Evidence |
|---|---|---|---|
| L1 | PASS | Every emotion label is one of the 7 expected classes | unexpected labels: {'train': [], 'dev': [], 'test': []} |
| L2 | PASS | Sentiment labels are positive/negative/neutral | {'train': ['negative', 'neutral', 'positive'], 'dev': ['negative', 'neutral', 'positive'], 'test': ['negative', 'neutral', 'positive']} |
| L3 | PASS | Emotion and sentiment agree (joy→positive, neutral→neutral, anger/disgust/fear/sadness→negative, surprise→any) | inconsistent rows per split: {'train': 0, 'dev': 0, 'test': 0} |
| L4 | PASS | No empty transcripts | empty per split: {'train': 0, 'dev': 0, 'test': 0} |
| L5 | PASS | No mojibake or replacement characters in transcripts | rows with suspicious characters per split: {'train': 0, 'dev': 0, 'test': 0} |
| L6 | WARN | EndTime is after StartTime for every row | non-positive nominal durations per split: {'train': 0, 'dev': 0, 'test': 1}; rows: ['test_dia155_utt3']. Known MELD annotation defect; decided by the audio length in D9. |
| A1 | PASS | Raw inventory: every clip has one decodable audio stream; label/file set differences listed | train: files 9989, labelled rows 9989, rows without file 0 [], files without label 0, errors 1 ['dia125_utt3.mp4'], channels {6.0: 8871, 2.0: 1117}, sample rates {48000.0: 9988} |
| L7 | PASS | Archive CSVs vs GitHub CSVs compared; GitHub (UTF-8, corrected) is the label source | train: rows 9989 vs 9989; metadata differs in 2 rows ['dia490_utt0', 'dia556_utt6']; text differs beyond encoding/punctuation in 4 rows; dev: rows 1109 vs 1109; metadata differs in 0 rows []; text differs beyond encoding/punctuation in 1 rows; test: rows 2610 vs 2610; metadata differs in 0 rows []; text differs beyond encoding/punctuation in 1 rows |
| D2 | PASS | Fewer than 1 % of rows per split lack a decodable clip; count reported | rows without audio per split: {'train': 1, 'dev': 1, 'test': 0}; ffmpeg failures logged: 1 |
| D3 | PASS | Class counts and imbalance ratio (max/min) reported per split | counts: |
| D4 | PASS | No non-excluded utterance (by id, and by season/episode/start/end timestamp) appears in two splits | train rows excluded: 350 by reason {'fragment_lt_0.5s': 306, 'shared_audio_group': 38, 'timestamp_duplicate_of': 5, 'no_audio': 1}; timestamp duplicates: ['timestamp_duplicate_of:dev_dia52_utt2', 'timestamp_duplicate_of:dev_dia77_utt0', 'timestamp_duplicate_of:test_dia117_utt8', 'timestamp_duplicate_of:test_dia85_utt0', 'timestamp_duplicate_of:test_dia85_utt1']; after exclusion, utt_id overlap: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; timestamp-key overlap: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; episodes shared (information only, dialogues are the split unit): {'train/dev': 81, 'train/test': 142, 'dev/test': 67} |
| D5 | PASS | No identical audio file (sha256) across splits after exclusions | cross-split identical files: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; duplicates within split: {'train': 0, 'dev': 1, 'test': 4} |
| D6 | PASS | Speaker overlap across splits is documented (MELD is dialogue-disjoint, not speaker-disjoint) | 21 speakers appear in all three splits: ['All', 'Bob', 'Carol', 'Chandler', 'Cliff', 'Elizabeth', 'Frank', 'Gary', 'Guy', 'Joanna', 'Joey', 'Mark', 'Monica', 'Mrs. Geller', 'Phoebe', 'Rachel', 'Ross', 'Susan', 'Tag', 'The Dry Cleaner', 'Woman']; fraction of rows spoken by them per split: {'train': 0.862, 'dev': 0.935, 'test': 0.854} |
| D7 | PASS | No wav path contains its emotion label | paths containing the label: 0 |
| D8 | PASS | Duration stats per split (duration_s); clips above 15 s counted |  |
| D9 | PASS | Extracted audio length agrees with the CSV timestamps within 1 s for at least 98 % of clips (WARN only: MELD timestamps are known to be unreliable; the WAV length is the ground truth downstream) | clips differing by more than 1 s per split: {'train': 1, 'dev': 0, 'test': 2} |
| P1 | PASS | Data policy applied: train exclusions by reason; dev/test never excluded; clean_subset sizes | train exclusion reasons: {'kept': 9639, 'fragment_lt_0.5s': 306, 'shared_audio_group': 38, 'timestamp_duplicate_of': 5, 'no_audio': 1}; clean_subset per split: {'train': '9637/9989', 'dev': '1066/1109', 'test': '2504/2610'}; train class counts after exclusion: {'neutral': 4546, 'joy': 1666, 'surprise': 1156, 'anger': 1071, 'sadness': 671, 'disgust': 267, 'fear': 262} |

## Details

### L1 (PASS)

Every emotion label is one of the 7 expected classes

unexpected labels: {'train': [], 'dev': [], 'test': []}

### L2 (PASS)

Sentiment labels are positive/negative/neutral

{'train': ['negative', 'neutral', 'positive'], 'dev': ['negative', 'neutral', 'positive'], 'test': ['negative', 'neutral', 'positive']}

### L3 (PASS)

Emotion and sentiment agree (joy→positive, neutral→neutral, anger/disgust/fear/sadness→negative, surprise→any)

inconsistent rows per split: {'train': 0, 'dev': 0, 'test': 0}

### L4 (PASS)

No empty transcripts

empty per split: {'train': 0, 'dev': 0, 'test': 0}

### L5 (PASS)

No mojibake or replacement characters in transcripts

rows with suspicious characters per split: {'train': 0, 'dev': 0, 'test': 0}

### L6 (WARN)

EndTime is after StartTime for every row

non-positive nominal durations per split: {'train': 0, 'dev': 0, 'test': 1}; rows: ['test_dia155_utt3']. Known MELD annotation defect; decided by the audio length in D9.

### A1 (PASS)

Raw inventory: every clip has one decodable audio stream; label/file set differences listed

train: files 9989, labelled rows 9989, rows without file 0 [], files without label 0, errors 1 ['dia125_utt3.mp4'], channels {6.0: 8871, 2.0: 1117}, sample rates {48000.0: 9988}
dev: files 1112, labelled rows 1109, rows without file 1 ['dia110_utt7'], files without label 4, errors 0 [], channels {6.0: 909, 2.0: 203}, sample rates {48000.0: 1112}
test: files 2615, labelled rows 2610, rows without file 0 [], files without label 5, errors 0 [], channels {6.0: 2338, 2.0: 277}, sample rates {48000.0: 2615}

### L7 (PASS)

Archive CSVs vs GitHub CSVs compared; GitHub (UTF-8, corrected) is the label source

train: rows 9989 vs 9989; metadata differs in 2 rows ['dia490_utt0', 'dia556_utt6']; text differs beyond encoding/punctuation in 4 rows; dev: rows 1109 vs 1109; metadata differs in 0 rows []; text differs beyond encoding/punctuation in 1 rows; test: rows 2610 vs 2610; metadata differs in 0 rows []; text differs beyond encoding/punctuation in 1 rows

### D2 (PASS)

Fewer than 1 % of rows per split lack a decodable clip; count reported

rows without audio per split: {'train': 1, 'dev': 1, 'test': 0}; ffmpeg failures logged: 1

### D3 (PASS)

Class counts and imbalance ratio (max/min) reported per split

counts:

| emotion   |   train |   dev |   test |
|:----------|--------:|------:|-------:|
| neutral   |    4710 |   470 |   1256 |
| joy       |    1743 |   163 |    402 |
| surprise  |    1205 |   150 |    281 |
| anger     |    1109 |   153 |    345 |
| sadness   |     683 |   111 |    208 |
| disgust   |     271 |    22 |     68 |
| fear      |     268 |    40 |     50 |

imbalance ratio: {'train': 17.6, 'dev': 21.4, 'test': 25.1}

### D4 (PASS)

No non-excluded utterance (by id, and by season/episode/start/end timestamp) appears in two splits

train rows excluded: 350 by reason {'fragment_lt_0.5s': 306, 'shared_audio_group': 38, 'timestamp_duplicate_of': 5, 'no_audio': 1}; timestamp duplicates: ['timestamp_duplicate_of:dev_dia52_utt2', 'timestamp_duplicate_of:dev_dia77_utt0', 'timestamp_duplicate_of:test_dia117_utt8', 'timestamp_duplicate_of:test_dia85_utt0', 'timestamp_duplicate_of:test_dia85_utt1']; after exclusion, utt_id overlap: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; timestamp-key overlap: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; episodes shared (information only, dialogues are the split unit): {'train/dev': 81, 'train/test': 142, 'dev/test': 67}

### D5 (PASS)

No identical audio file (sha256) across splits after exclusions

cross-split identical files: {'train/dev': 0, 'train/test': 0, 'dev/test': 0}; duplicates within split: {'train': 0, 'dev': 1, 'test': 4}

### D6 (PASS)

Speaker overlap across splits is documented (MELD is dialogue-disjoint, not speaker-disjoint)

21 speakers appear in all three splits: ['All', 'Bob', 'Carol', 'Chandler', 'Cliff', 'Elizabeth', 'Frank', 'Gary', 'Guy', 'Joanna', 'Joey', 'Mark', 'Monica', 'Mrs. Geller', 'Phoebe', 'Rachel', 'Ross', 'Susan', 'Tag', 'The Dry Cleaner', 'Woman']; fraction of rows spoken by them per split: {'train': 0.862, 'dev': 0.935, 'test': 0.854}

### D7 (PASS)

No wav path contains its emotion label

paths containing the label: 0

### D8 (PASS)

Duration stats per split (duration_s); clips above 15 s counted



|       |   train |     dev |    test |
|:------|--------:|--------:|--------:|
| count | 9988    | 1108    | 2610    |
| mean  |    3.14 |    3.12 |    3.31 |
| 50%   |    2.47 |    2.44 |    2.47 |
| min   |    0.06 |    0.06 |    0.13 |
| max   |   41.05 |   28.54 |  304.96 |

clips above threshold: {'train': 25, 'dev': 6, 'test': 6}

### D9 (PASS)

Extracted audio length agrees with the CSV timestamps within 1 s for at least 98 % of clips (WARN only: MELD timestamps are known to be unreliable; the WAV length is the ground truth downstream)

clips differing by more than 1 s per split: {'train': 1, 'dev': 0, 'test': 2}

### P1 (PASS)

Data policy applied: train exclusions by reason; dev/test never excluded; clean_subset sizes

train exclusion reasons: {'kept': 9639, 'fragment_lt_0.5s': 306, 'shared_audio_group': 38, 'timestamp_duplicate_of': 5, 'no_audio': 1}; clean_subset per split: {'train': '9637/9989', 'dev': '1066/1109', 'test': '2504/2610'}; train class counts after exclusion: {'neutral': 4546, 'joy': 1666, 'surprise': 1156, 'anger': 1071, 'sadness': 671, 'disgust': 267, 'fear': 262}
