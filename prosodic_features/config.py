"""
All paths and parameters of the pipeline, grouped by step.

Paths are relative to DATA_DIR, which defaults to ./data inside this
repository. Point it elsewhere with an environment variable:

    export CROWD_DATA_DIR=/path/to/data
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(os.environ.get("CROWD_DATA_DIR", REPO_DIR / "data"))
MODELS_DIR = Path(os.environ.get("CROWD_MODELS_DIR", REPO_DIR / "models"))

# ---------------------------------------------------------------------------
# Recordings metadata: one row per recording; every step processes the
# recordings listed here (see metadata.py). Only read, never modified.
# ---------------------------------------------------------------------------
AUDIO_METADATA = Path(os.environ.get("CROWD_AUDIO_METADATA", DATA_DIR / "recordings.xlsx"))
METADATA_FILENAME_COLUMN = "FILENAME"             # audio file name, e.g. mlk_1963_dream.wav
METADATA_DIARIZE_COLUMN  = "DIARIZATION_NEEDED"   # 1 / 0: diarize (step 02)
METADATA_CROWD_COLUMN    = "ACTIVE_CROWD"         # 1 / 0: measure crowd loudness (step 03)
# Optional, for diarized recordings: stretches (onset, offset in seconds)
# where only the target speaker talks. The diarization speaker that talks most
# within them gets the target speaker's name (step 02; see metadata.speaker_name).
METADATA_MARK_COLUMNS    = [("ONSET_S1", "OFFSET_S1"), ("ONSET_S2", "OFFSET_S2")]

RAW_AUDIO_DIR   = DATA_DIR / "audio" / "raw"                              # input of step 01
DENOISED_DIR    = DATA_DIR / "audio" / "separated" / "denoised" / "pass01"  # speech (step 01)
CROWD_DIR       = DATA_DIR / "audio" / "separated" / "noise" / "pass01"     # crowd = mix - speech (step 01)
DIARIZATION_DIR = DATA_DIR / "derived" / "diarization"                    # step 02 (diarized recordings)
TRANSCRIPT_DIR  = DATA_DIR / "derived" / "transcripts"                    # step 02
FEATURES_DIR    = DATA_DIR / "derived" / "features"                       # step 03

DENOISED_PREFIX = "denoised_"   # step 01 output names: denoised_<stem>.flac
CROWD_PREFIX    = "noise_"      #                       noise_<stem>.flac

DEFAULT_SPEAKER = "SPEAKER_00"  # step 03: label of words without a speaker (outside every diarization turn)

# ---------------------------------------------------------------------------
# Step 01: speech / crowd separation (GTCRN denoiser, 16 kHz)
# ---------------------------------------------------------------------------
GTCRN_MODEL_PATH = MODELS_DIR / "gtcrn_simple.onnx"
SEPARATE_SR          = 16000
SEPARATE_MAX_SECONDS = None     # e.g. 180 to test on the first 3 min; None = full file
SEPARATE_SKIP_EXISTING = True   # skip files whose two outputs already exist
# Recordings denoised in parallel. ADAPT TO YOUR MACHINE: each worker holds a
# whole recording in memory (a 1-hour file is ~0.5 GB as raw audio) plus its
# own copy of the model, so lower this if you run out of RAM, raise it if
# you have spare cores and memory. 8 was used on a 14-core, 24 GB laptop.
SEPARATE_N_WORKERS = 8

# ---------------------------------------------------------------------------
# Step 02, recordings with DIARIZATION_NEEDED: speaker diarization (pyannote)
# ---------------------------------------------------------------------------
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DIARIZATION_DEVICE   = "mps"    # falls back to CPU if an MPS op is unimplemented
DIARIZATION_MAX_GAP_S      = 0.25   # merge same-speaker turns closer than this
DIARIZATION_MIN_DURATION_S = 0.60   # then drop turns shorter than this

# ---------------------------------------------------------------------------
# Step 02: voice activity detection (Silero) + transcription (WhisperX)
# ---------------------------------------------------------------------------
SILERO_SR        = 16000
SILERO_THRESHOLD = 0.40
SILERO_MIN_SPEECH_MS  = 500
SILERO_MIN_SILENCE_MS = 50
SILERO_MERGE_GAP_S    = 0.05    # chunks closer than this (s) are joined into one

WHISPER_MODEL        = "large-v3"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE     = "en"

TRANSCRIBE_MAX_SECONDS = None   # e.g. 120 to test on the first 2 min; None = full file
TRIM_WORDS_SLACK = 3            # see transcription.handle_edge_words

# ---------------------------------------------------------------------------
# Step 03: acoustic features
# ---------------------------------------------------------------------------
FEATURES_SR        = 16000
# Segments processed in parallel. ADAPT TO YOUR MACHINE: default = all cores
# but one; lower it to keep the computer usable or on a shared server.
FEATURES_N_WORKERS = max(1, mp.cpu_count() - 1)

SEGMENT_MERGE_GAP_S = 0.2       # merge same-speaker segments closer than this
REVERB_LAG_S        = 0.05      # trimmed from both ends of a gap for the *_rob measures
MIN_SEGMENT_S       = 0.25      # shorter segments get NaN openSMILE features

SPEECH_PREFIX = "sp_"           # prefix of the speech-side feature columns
CROWD_LOUDNESS_KEY = "loudness_sma3_amean"   # eGeMAPS feature used as crowd loudness

# Recording-level loudness reference: this percentile of the frame-level
# eGeMAPS loudness (LLD) of the whole original recording (before separation). Utterance mean loudness
# is divided by it, so loudness is comparable across recordings with
# different gain. A high percentile rather than the max, so a single spike
# (a shout, a residual noise burst) does not set the scale.
LOUDNESS_LLD_KEY       = "Loudness_sma3"
LOUDNESS_REF_PERCENTILE = 99
