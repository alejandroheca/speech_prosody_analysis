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
METADATA_CROWD_COLUMN    = "ACTIVE_CROWD"         # 1 / 0: detect the crowd and measure its loudness (steps 04, 05)
# Optional, for diarized recordings: stretches (onset, offset in seconds)
# where only the target speaker talks. The diarization speaker that talks most
# within them gets the target speaker's name (step 02; see metadata.speaker_name).
METADATA_MARK_COLUMNS    = [("ONSET_S1", "OFFSET_S1"), ("ONSET_S2", "OFFSET_S2")]

RAW_AUDIO_DIR   = Path(os.environ.get("CROWD_RAW_AUDIO_DIR", DATA_DIR / "audio" / "raw"))  # input of steps 01, 03
DENOISED_DIR    = DATA_DIR / "audio" / "separated" / "denoised" / "pass01"  # speech (step 01)
CROWD_DIR       = DATA_DIR / "audio" / "separated" / "noise" / "pass01"     # crowd = mix - speech (step 01)
DIARIZATION_DIR = DATA_DIR / "derived" / "diarization"                    # step 02 (diarized recordings)
TRANSCRIPT_DIR  = DATA_DIR / "derived" / "transcripts"                    # step 02
CROWD_CNN_DIR   = DATA_DIR / "derived" / "crowd_cnn"                      # step 04
QUALITY_DIR     = DATA_DIR / "derived" / "quality"                        # step 03, per row
FEATURES_DIR    = DATA_DIR / "derived" / "features"                       # step 05
# One row per processed recording, next to the metadata file (steps 03, 05)
RECORDINGS_DONE_CSV = AUDIO_METADATA.parent / "recordings_done.csv"

DENOISED_PREFIX = "denoised_"   # step 01 output names: denoised_<stem>.flac
CROWD_PREFIX    = "noise_"      #                       noise_<stem>.flac

DEFAULT_SPEAKER = "SPEAKER_00"  # step 05: label of words without a speaker (outside every diarization turn)

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
SILERO_SPEECH_PAD_MS  = 30      # padding (ms) added to both ends of each chunk (Silero's default)
SILERO_MERGE_GAP_S    = 0.05    # chunks closer than this (s) are joined into one
CHUNK_END_PAD_S       = 0.03    # after transcription, a chunk ends this long after its last word
SENTENCE_FRAGMENT_MAX_S = 0.5   # a sentence piece shorter than this in a chunk shared with
                                # the neighbouring sentence is a fragment (see sentences.fit_sentences_to_chunks)
# Sentences are built after WhisperX, from the words, by prosodic_features/sentences.py
# (rules in its docstring): replacements first, then a sentence ends at . ? ! or at a
# change of speaker (an interrupted sentence gets a "."); no pause rule.
# Replacements, word by word. Entries that start with punctuation are replaced anywhere
# in a word (an ellipsis becomes a comma); the others replace a whole word.
ABBREVIATIONS = {
    "...": ",", "…": ",",
    "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz", "Dr.": "Doctor", "Rev.": "Reverend",
    "Gov.": "Governor", "Sen.": "Senator", "Rep.": "Representative", "Gen.": "General",
    "Prof.": "Professor", "St.": "Saint", "Jr.": "Junior", "Sr.": "Senior",
    "vs.": "versus", "etc.": "etcetera", "i.e.": "that is", "e.g.": "for example",
    "a.m.": "AM", "p.m.": "PM", "A.D.": "AD", "B.C.": "BC",
    "U.S.": "United States", "U.S.A.": "United States of America", "U.N.": "United Nations",
}
# ...of which these come before a name and never end a sentence (the others end one
# when the next word is capitalised)
ABBREVIATION_TITLES = {"Mr.", "Mrs.", "Ms.", "Dr.", "Rev.", "Gov.", "Sen.", "Rep.", "Gen.",
                       "Prof.", "St."}

WHISPER_MODEL        = "large-v3"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE     = "en"
# VAD WhisperX uses to cut the recording into its <= 30 s decoding windows.
# "silero" (not WhisperX's default, "pyannote"): with pyannote, a window at the
# start of a recording could decode as dots only and lose its speech.
WHISPER_VAD_METHOD   = "silero"

TRANSCRIBE_MAX_SECONDS = None   # e.g. 120 to test on the first 2 min; None = full file
TRIM_WORDS_SLACK = 3            # see transcription.handle_edge_words

# ---------------------------------------------------------------------------
# Step 03: speech quality (TorchAudio-Squim objective model, no reference) of
# the original recording, on a random set of its Silero chunks (see speech_quality.py)
# ---------------------------------------------------------------------------
SQUIM_SR           = 16000
SQUIM_WINDOW_S     = 10.0       # chunks are scored in windows of at most this long
SQUIM_MIN_WINDOW_S = 0.5        # shorter windows are skipped
SQUIM_N_CHUNKS     = None       # a number: that many chunks drawn at random, one score per
                                #   recording (recordings_done.csv); None: every chunk and
                                #   sentence scored on its own (derived/quality, then a
                                #   speech_quality column per row in the step-05 tables)
SQUIM_SEED         = 0          # fixed, so a rerun draws the same chunks

# ---------------------------------------------------------------------------
# Step 04, recordings with ACTIVE_CROWD: crowd detection (CNN-TCN of
# crowd_reaction_analysis, event head, on the original recording)
# ---------------------------------------------------------------------------
CNN_DEVICE       = "cpu"
CNN_BATCH_SIZE   = 1
CNN_MEDIAN_FILTER_S = 3.0       # smoothing of the detected regions (the CNN's default)
                                # Every region is saved, with its mean probability; which
                                # ones count as crowd reactions is decided in step 05
                                # (CROWD_MIN_MEAN_SCORE)

# ---------------------------------------------------------------------------
# Step 05: acoustic features
# ---------------------------------------------------------------------------
FEATURES_SR        = 16000
# Feature tables written per recording, one per unit of rows: "chunk" (one
# row per Silero chunk, split where the speaker changes) and "sentence" (one
# row per transcript sentence, on the chunks it spans; adds the
# crowd_energy_around measures). Both by default; the environment variable
# CROWD_FEATURES_UNIT=chunk (or =sentence) writes only that one.
FEATURES_UNITS     = tuple(u for u in ("chunk", "sentence")
                           if os.environ.get("CROWD_FEATURES_UNIT", "both") in (u, "both"))
# Segments processed in parallel. ADAPT TO YOUR MACHINE: default = all cores
# but one; lower it to keep the computer usable or on a shared server.
FEATURES_N_WORKERS = max(1, mp.cpu_count() - 1)

REVERB_LAG_S        = 0.08      # trimmed from both ends of a gap for the *_rob measures
MIN_SEGMENT_S       = 0.25      # shorter segments get NaN openSMILE features
CROWD_MIN_MEAN_SCORE = 0.8      # a step-04 CNN region counts as a crowd reaction if its mean
                                #   crowd probability is at least this; the others are ignored
LAG_REACTION_WINDOW_S = 3.0     # crowd_energy_lag_rob: a CNN reaction counts if it starts
                                #   at most this many seconds after the row's end
CROWD_FLOOR_PERCENTILE = 10     # crowd_energy_floor: this percentile of the crowd loudness
                                #   in the pauses without a crowd (static); subtracted
                                #   from every crowd energy (crowd_gap.crowd_energy_floor)

SPEECH_PREFIX = "sp_"           # prefix of the speech-side feature columns
CROWD_LOUDNESS_KEY = "loudness_sma3_amean"   # eGeMAPS feature used as crowd loudness

# Gain normalization: before any feature is computed, the speech and crowd
# signals are divided by this percentile of the absolute sample amplitude of
# the whole original recording (before separation). One reference per
# recording for both signals (same microphone), so features are comparable
# across recordings with different gain. A high percentile rather than the
# max, so a single spike (a shout, a click) does not set the scale.
AMPLITUDE_REF_PERCENTILE = 99
