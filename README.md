# Prosodic features

Utterance-level prosodic features of speech recordings: speeches, sermons,
interviews, and debates. This is one of two parallel repositories of the same
project, to be used together with the crowd-reaction classifier in
[chymaera96/crowd_reaction_analysis](https://github.com/chymaera96/crowd_reaction_analysis).

The pipeline takes a recording (`.flac` recommended; `.wav` also accepted) and:

1. denoises the speech, separating the speaker's voice from the crowd and
   background;
2. transcribes the speech word by word and splits it into sentences, labelling
   who speaks when in multi-speaker recordings;
3. estimates the recording's speech quality (PESQ, by TorchAudio-Squim, on its
   speech only);
4. in recordings with an active crowd (`ACTIVE_CROWD = 1`), detects crowd
   reactions with the CNN of
   [crowd_reaction_analysis](https://github.com/chymaera96/crowd_reaction_analysis);
5. computes prosodic features for every speech chunk or every sentence (your
   choice, `FEATURES_UNIT`), on its voiced parts only: the eGeMAPS set, plus
   rhythm measures not in eGeMAPS (number of words, speech rate, and the
   duration of the gap after it).

For recordings with an active crowd, it also measures the crowd's loudness in
the gaps where the CNN detects a reaction: the loudness of the separated crowd
track in the gap after each chunk or sentence and, for sentences, summed over
the sentence's inner pauses and its end.

Before any feature is computed, the speech and crowd tracks are divided by the
99th percentile of the absolute amplitude of the original recording (one value
per recording, since both come from the same microphone), so features are
comparable across recordings with different gain.

The output is one table per recording, with one row per chunk or sentence,
and a summary table with one row per processed recording.

## Installation

Run commands from the repository root. The pipeline uses two Conda
environments: `crowd_audio` for steps 1 and 5, and `crowd_transcribe` for
steps 2 to 4.

```
conda env create -f envs/crowd_audio.yml
conda env create -f envs/crowd_transcribe.yml
```

Step 4 uses the crowd-reaction CNN from the parallel repository. Clone it next
to this one (or set `CROWD_CNN_REPO` to its path) and download its weights:

```
git clone https://github.com/chymaera96/crowd_reaction_analysis ../crowd_reaction_analysis
cd ../crowd_reaction_analysis && python scripts/download_weights.py --model cnn_tcn
```

## Models

Download `gtcrn_simple.onnx` into `models/`. It is the 16 kHz model in the
`16k/` folder of [SpeechDenoiser](https://github.com/yuyun2000/SpeechDenoiser),
an ONNX export of [GTCRN](https://github.com/Xiaobin-Rong/gtcrn). Silero VAD,
WhisperX, and pyannote download automatically on first use.

Diarization requires a Hugging Face token. Accept the terms of
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
then set the token in your shell rather than in the code:

```
export HF_TOKEN=hf_...
```

The token is only needed when at least one recording has
`DIARIZATION_NEEDED = 1`.

## Data

Arrange the data as follows:

```
data/
  recordings.xlsx
  audio/raw/*.flac
```

Name each recording with the speaker's identifier as a prefix, followed by an
underscore: `shelton_1965_kkk_meeting.wav`, `mlk_1963_dream.wav`. The prefix
(everything before the first `_`) is the speaker's name in the transcripts and
the features table: every word of a single-speaker recording gets it, and so
does the target speaker of a diarized recording (see the speaker marks below).

Recordings can be `.flac` (recommended) or `.wav`. The metadata file can be `.xlsx` or
`.csv`, with one row per recording. Every step processes the recordings listed
in it; recordings without audio are reported and skipped. The pipeline only
reads this file. It uses these columns:

| Column | Values | Step | Meaning |
|---|---|---|---|
| `FILENAME` | e.g. `mlk_1963_dream.wav` | all | Audio file name in `data/audio/raw/`. The name without extension names every output; its prefix up to the first `_` is the speaker's name. |
| `DIARIZATION_NEEDED` | 1 / 0 | 2 | 1: several people speak (interview, debate, Q&A). The recording is diarized, and each word gets the speaker whose turn overlaps it most (`SPEAKER_00`, `SPEAKER_01`, ...). 0: every word gets the speaker's name from the file name. |
| `ACTIVE_CROWD` | 1 / 0 | 4, 5 | 1: an audience reacts to the speaker, so the crowd is detected (step 4) and its loudness is measured in the gaps between chunks or sentences (step 5). 0: the gaps contain other speakers or silence, and the crowd columns stay empty. |
| `ONSET_S1`, `OFFSET_S1`, `ONSET_S2`, `OFFSET_S2` | seconds | 2 | Optional, for diarized recordings. One or two stretches where only the target speaker talks. The diarization speaker that talks most within them gets the speaker's name from the file name; the other voices keep their `SPEAKER_xx` labels. Leave both cells of a pair empty if unused. |

`1`, `yes`, `y`, and `true` count as 1; any other value, including an empty
cell, counts as 0. See `data/README.md` for how to choose the speaker marks.
Set `CROWD_DATA_DIR` and `CROWD_AUDIO_METADATA` to use another data folder or
metadata file. Column names and all other paths and
parameters are in `prosodic_features/config.py`.

## Running the pipeline

Run the five steps in order:

```
./run_all.sh
```

To run one step at a time, activate its environment first:

```
conda activate crowd_audio
python 01_separate.py
conda activate crowd_transcribe
python 02_transcribe.py
python 03_quality.py
python 04_crowd_detection.py
conda activate crowd_audio
python 05_features.py
```

| Step | Script | What it does |
|---|---|---|
| 1 | `01_separate.py` | Splits each recording into speech (GTCRN speech enhancement) and crowd (the original minus the speech), mono 16 kHz. |
| 2 | `02_transcribe.py` | Finds speech chunks (Silero VAD), joins them into one audio with 0.1 s of silence between them, transcribes and word-aligns it in one pass (WhisperX), keeps each chunk's first and last word inside the chunk, joins sentences WhisperX cut at a pause, and assigns each word a speaker. Recordings with `DIARIZATION_NEEDED = 1` are diarized first (pyannote), and the target speaker is identified from the speaker marks. |
| 3 | `03_quality.py` | Estimates the speech quality of the original recording, without a reference, on its speech chunks: PESQ from 1 (bad) to 4.5 (excellent), by TorchAudio-Squim, written to `recordings_done.csv`. |
| 4 | `04_crowd_detection.py` | In recordings with `ACTIVE_CROWD = 1`, runs the crowd-reaction CNN (CNN-TCN, crowd-event head) on the original recording and keeps the crowd regions with a mean probability above `CNN_MIN_MEAN_SCORE` (0.8). |
| 5 | `05_features.py` | Normalizes the gain of the speech and crowd tracks, then computes, per chunk or sentence (`FEATURES_UNIT`), eGeMAPS (openSMILE) and speech rate on its voiced parts, and the duration and crowd loudness of its gaps. |

Each step skips recordings whose outputs already exist, so an interrupted run
can be restarted. `SEPARATE_N_WORKERS` and `FEATURES_N_WORKERS` in `config.py`
set how many recordings (step 1) and rows (step 5) are processed at once;
adapt them to your machine. To test on the start of each recording, set
`SEPARATE_MAX_SECONDS` and `TRANSCRIBE_MAX_SECONDS`.

The pipeline was developed on macOS (Apple silicon) with the package versions
in `envs/`. Diarization runs on the Apple GPU (`DIARIZATION_DEVICE = "mps"`)
and falls back to the CPU when needed.

## Outputs

All outputs are written inside the data folder:

```
data/
  audio/separated/denoised/pass01/denoised_<name>.flac       speech (step 1)
  audio/separated/noise/pass01/noise_<name>.flac             crowd (step 1)
  derived/diarization/denoised_<name>_diarization.json, .rttm speaker turns (step 2)
  derived/transcripts/<name>.json                            sentences, words, times, speakers (step 2)
  derived/transcripts/<name>_*.txt                           label tracks (step 2)
  derived/crowd_cnn/<name>_crowd_cnn.txt                     crowd regions, label track (step 4)
  derived/features/<name>_<unit>_features.csv                one row per chunk or sentence (step 5)
  recordings_done.csv                                        one row per processed recording (steps 3, 5)
```

Diarization files exist only for diarized recordings. The label tracks are
`_silero_chunks.txt` (speech chunks; chunks less than `SILERO_MERGE_GAP_S` =
0.05 s apart are joined, and each chunk ends `CHUNK_END_PAD_S` = 0.03 s after
its last word), `_words.txt` (words after the edge-word check: stray
hallucinated words at a sentence's edges removed, and each chunk's first and
last word kept inside the chunk), and `_sentences.txt` (a sentence not ending in `.`, `?` or `!`, or
ending in `...` before a sentence starting in lowercase, is joined with the next one if the same speaker goes on within
`SENTENCE_JOIN_MAX_GAP_S` = 3 s; a sentence can span several chunks, but not by a
fragment shorter than `SENTENCE_FRAGMENT_MAX_S` = 0.5 s into a chunk it shares
with the neighbouring sentence; the first sentence of each chunk starts at the
chunk's start and the last one ends at the chunk's end). Load the separated audio in Audacity, then import a label
track to check it by listening.

`<name>_<unit>_features.csv` has one row per chunk or sentence, set by
`FEATURES_UNIT` in `config.py` (or the environment variable
`CROWD_FEATURES_UNIT`):

- `chunk` (default): one row per Silero chunk; a chunk where the speaker
  changes is split into one row per speaker. The row is its own voiced part.
- `sentence`: one row per sentence. A sentence's voiced parts are the Silero
  chunks it spans, cut to the sentence's own start and end (a sentence inside
  a longer chunk keeps only its part).

All speech features use only the voiced parts; the crowd features use the gaps.

- `start`, `end`, `speaker`, `multi_speaker`, `text`: the row. `multi_speaker`
  marks chunks cut where the speaker changes, or sentences with words of
  several speakers.
- `speech_quality`: the recording's speech quality (the same on every row),
  see below.
- `sentence_end` (chunks only): the chunk ends a sentence. `n_chunks`
  (sentences only): the sentence's number of voiced parts.
- `duration`, `n_words`, `word_rate`: total duration of the voiced parts, and
  words per second of it.
- `gap_duration`, `gap_duration_rob`, `reaction_end`: the end gap, from the
  row's end to the next row's start. The `_rob` versions leave out
  50 ms at each end of the gap.
- `crowd_energy_end`, `crowd_energy_end_rob`: crowd loudness in the end gap
  (eGeMAPS `loudness_sma3_amean` of the gain-normalized crowd track).
- `crowd_energy_around`, `crowd_energy_around_rob` (sentences only): the sum
  of the crowd loudness in the gaps between the sentence's voiced parts and in
  its end gap.
- In both, a gap counts only if a crowd region from step 4 overlaps it;
  otherwise, or if it is shorter than `MIN_SEGMENT_S` (0.25 s), it counts as 0.
  The `_rob` versions measure each gap without 50 ms at each end. They are
  computed only when `ACTIVE_CROWD = 1`, and never for a `multi_speaker`
  row (empty otherwise).
- `crowd_energy_end_rob_lag`: `crowd_energy_end_rob` of the previous row.
  For the first row, it is the crowd before the speech starts.
- `sp_<eGeMAPS feature>`: the 88 eGeMAPS v02 functionals of the voiced parts,
  on the gain-normalized speech track.

Gain normalization: both tracks are divided by the `AMPLITUDE_REF_PERCENTILE`
(99th) percentile of the absolute sample amplitude of the whole original
recording, before separation. A high percentile rather than the maximum, so a
single spike does not set the scale.

Speech quality (`speech_quality`): the perceived quality of the speech, PESQ
from 1 (bad) to 4.5 (excellent), covering noise, distortion and band-limiting.
It is estimated in step 3 by
[TorchAudio-Squim](https://pytorch.org/audio/stable/tutorials/squim_tutorial.html)
(objective model) from the original recording alone, without a clean
reference, on its voiced Silero chunks (in windows of at most 10 s, averaged
weighted by duration).

`recordings_done.csv`, next to the metadata file (`recordings.xlsx`), has one
row per processed recording: step 3 writes its `speech_quality`, and step 5
reads it from there and adds the rest: `file`,
`speaker` (the file name's prefix), `duration` (seconds), `diarized` (1 / 0),
and `speech_quality`.

## Repository structure

```
01_separate.py, 02_transcribe.py,
03_quality.py, 04_crowd_detection.py,
05_features.py                                     the five steps
run_all.sh                                         runs them in order
envs/                                              Conda environments
models/                                            GTCRN model (not included)
data/                                              recordings and outputs (not included)
prosodic_features/
  config.py            paths and parameters
  metadata.py          metadata file
  audio.py, labels.py, utils.py
  separation.py        step 1
  transcription.py     step 2: VAD, WhisperX, edge-word check
  diarization.py       step 2: pyannote
  speakers.py          step 2: speaker of each word and sentence
  speech_quality.py    step 3: speech quality (Squim)
  crowd_detection.py   step 4: crowd regions (CNN of crowd_reaction_analysis)
  segments.py          step 5: chunks or sentences and their voiced parts
  segment_features.py  step 5: all features of one chunk or sentence
  features/            step 5: one file per feature
```

## Acknowledgements

GTCRN ([Rong et al.](https://github.com/Xiaobin-Rong/gtcrn); ONNX export from
[SpeechDenoiser](https://github.com/yuyun2000/SpeechDenoiser)),
[Silero VAD](https://github.com/snakers4/silero-vad),
[WhisperX](https://github.com/m-bain/whisperX),
[pyannote.audio](https://github.com/pyannote/pyannote-audio),
[openSMILE](https://github.com/audeering/opensmile-python),
TorchAudio-Squim, and
[librosa](https://librosa.org).

## Citation

Hermida Carrillo, A. C., Bhattacharjee, A., & Benetos, E. (2026). [To the sound of
your applause: a signal processing approach to crowd-speaker communication in
political speeches.](https://spiral.imperial.ac.uk/entities/publication/29bbc09e-718a-4190-a838-fa276b371e9a) *UK and Ireland Speech (UKIS) Workshop*.
