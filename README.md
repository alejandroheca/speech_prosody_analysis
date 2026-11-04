# Prosodic features

Utterance-level prosodic features of speech recordings: speeches, sermons,
interviews, and debates. This is one of two parallel repositories of the same
project, to be used together with the crowd-reaction classifier in
[chymaera96/crowd_reaction_analysis](https://github.com/chymaera96/crowd_reaction_analysis).

The pipeline takes a recording (`.flac` recommended; `.wav` also accepted) and:

1. denoises the speech, separating the speaker's voice from the crowd and
   background;
2. transcribes the speech word by word and splits it into utterances, labelling
   who speaks when in multi-speaker recordings;
3. computes prosodic features for every utterance: the eGeMAPS set, plus
   rhythm measures not in eGeMAPS (number of words, speech rate, and the
   duration of the gap after the utterance).

For recordings with an active crowd (`ACTIVE_CROWD = 1`), it also computes a
simple crowd energy measure in each gap between utterances: the loudness of
the separated crowd track in the gap, relative to the whole crowd track.

The output is one table per recording, with one row per utterance.

## Installation

Run commands from the repository root. The pipeline uses two Conda
environments: `crowd_audio` for steps 1 and 3, and `crowd_transcribe` for
step 2.

```
conda env create -f envs/crowd_audio.yml
conda env create -f envs/crowd_transcribe.yml
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
| `ACTIVE_CROWD` | 1 / 0 | 3 | 1: an audience reacts to the speaker, so crowd loudness is measured in the gaps between segments. 0: the gaps contain other speakers or silence, and the crowd columns stay empty. |
| `ONSET_S1`, `OFFSET_S1`, `ONSET_S2`, `OFFSET_S2` | seconds | 2 | Optional, for diarized recordings. One or two stretches where only the target speaker talks. The diarization speaker that talks most within them gets the speaker's name from the file name; the other voices keep their `SPEAKER_xx` labels. Leave both cells of a pair empty if unused. |

`1`, `yes`, `y`, and `true` count as 1; any other value, including an empty
cell, counts as 0. See `data/README.md` for how to choose the speaker marks.
Set `CROWD_DATA_DIR` and `CROWD_AUDIO_METADATA` to use another data folder or
metadata file. Column names and all other paths and
parameters are in `prosodic_features/config.py`.

## Running the pipeline

Run the three steps in order:

```
./run_all.sh
```

To run one step at a time, activate its environment first:

```
conda activate crowd_audio
python 01_separate.py
conda activate crowd_transcribe
python 02_transcribe.py
conda activate crowd_audio
python 03_features.py
```

| Step | Script | What it does |
|---|---|---|
| 1 | `01_separate.py` | Splits each recording into speech (GTCRN speech enhancement) and crowd (the original minus the speech), mono 16 kHz. |
| 2 | `02_transcribe.py` | Finds speech chunks (Silero VAD), joins them into one audio with 0.1 s of silence between them, transcribes and word-aligns it in one pass (WhisperX), keeps each chunk's first and last word inside the chunk, and assigns each word a speaker. Recordings with `DIARIZATION_NEEDED = 1` are diarized first (pyannote), and the target speaker is identified from the speaker marks. |
| 3 | `03_features.py` | Builds single-speaker segments and computes, per segment, eGeMAPS (openSMILE), speech rate, and the duration and crowd loudness of the following gap. |

Each step skips recordings whose outputs already exist, so an interrupted run
can be restarted. `SEPARATE_N_WORKERS` and `FEATURES_N_WORKERS` in `config.py`
set how many recordings (step 1) and segments (step 3) are processed at once;
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
  derived/features/<name>_utterance_features.csv             one row per segment (step 3)
```

Diarization files exist only for diarized recordings. The label tracks are
`_silero_chunks.txt` (speech chunks; chunks less than `SILERO_MERGE_GAP_S` =
0.05 s apart are joined, and each chunk ends `CHUNK_END_PAD_S` = 0.03 s after
its last word; `_silero_chunks_original.txt` has the chunks as Silero found them), `_words.txt` and `_words_original.txt`
(words after and before the edge-word check: stray hallucinated words at a
sentence's edges removed, and each chunk's first and last word kept inside the
chunk), and `_sentences.txt` (a sentence can span several chunks, but not by a
fragment shorter than `SENTENCE_FRAGMENT_MAX_S` = 0.5 s into a chunk it shares
with the neighbouring sentence; the first sentence of each chunk starts at the
chunk's start and the last one ends at the chunk's end). Load the separated audio in Audacity, then import a label
track to check it by listening.

`<name>_utterance_features.csv` contains:

- `start`, `end`, `duration`, `speaker`, `multi_speaker`, `text`, `n_words`,
  `word_rate`, `sentence_end`: the segment. `multi_speaker` marks segments cut
  from a Silero chunk that contained a speaker change.
- `gap_duration`, `gap_duration_rob`, `reaction_end`: the gap until the next
  segment. The `_rob` versions leave out 50 ms at each end of the gap.
- `crowd_energy`, `crowd_energy_rob`: crowd loudness in the gap, relative to the
  loudness of the whole crowd track (eGeMAPS `loudness_sma3_amean`). These are
  computed only when `ACTIVE_CROWD = 1`, and never after a `multi_speaker`
  segment.
- `crowd_energy_rob_lag`: `crowd_energy_rob` of the previous gap. For the first
  segment, it is the crowd before the speech starts.
- `sp_<eGeMAPS feature>`: the 88 eGeMAPS v02 functionals. They are not
  normalized, so loudness and energy features depend on each recording's gain.
- `rec_loudness_p99`: the recording's loudness reference, the 99th percentile
  of the frame-level eGeMAPS loudness of the whole original recording, before
  separation (the same value on every row).
- `sp_loudness_sma3_amean_rel`: `sp_loudness_sma3_amean` divided by
  `rec_loudness_p99`, comparable across recordings with different gain.

## Repository structure

```
01_separate.py, 02_transcribe.py, 03_features.py   the three steps
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
  segments.py          step 3: segments from chunks and words
  segment_features.py  step 3: all features of one segment
  features/            step 3: one file per feature
```

## Acknowledgements

GTCRN ([Rong et al.](https://github.com/Xiaobin-Rong/gtcrn); ONNX export from
[SpeechDenoiser](https://github.com/yuyun2000/SpeechDenoiser)),
[Silero VAD](https://github.com/snakers4/silero-vad),
[WhisperX](https://github.com/m-bain/whisperX),
[pyannote.audio](https://github.com/pyannote/pyannote-audio),
[openSMILE](https://github.com/audeering/opensmile-python), and
[librosa](https://librosa.org).

## Citation

Hermida Carrillo, A. C., Bhattacharjee, A., & Benetos, E. (2026). [To the sound of
your applause: a signal processing approach to crowd-speaker communication in
political speeches.](https://spiral.imperial.ac.uk/entities/publication/29bbc09e-718a-4190-a838-fa276b371e9a) *UK and Ireland Speech (UKIS) Workshop*.
