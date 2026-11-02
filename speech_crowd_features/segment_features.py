"""Step 03: all features of one segment (run in parallel worker processes).

The recording's speech and crowd signals are put in shared memory once by
the step script, so the workers read them without each keeping a copy."""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from speech_crowd_features import config
from speech_crowd_features.features.crowd_gap import crowd_gap
from speech_crowd_features.features.opensmile_features import egemaps, ratio_to_file
from speech_crowd_features.features.pitch import pyin_pitch
from speech_crowd_features.features.spectral_entropy import spectral_entropy
from speech_crowd_features.features.spectral_profile import spectral_profile
from speech_crowd_features.features.speech_rate import speech_rate
from speech_crowd_features.utils import safe_float


def _round_or_nan(x, digits: int):
    return round(x, digits) if not np.isnan(x) else np.nan


def compute_segment_features(task: dict) -> Optional[dict]:
    """One output row for task["segment"]: timing, text, speech features of
    the segment and crowd features of the gap after it (until
    task["next_start"]).

    task also carries the shared-memory names / shapes / dtypes of the
    speech and crowd signals, the whole-file baselines used for the
    "_norm" and crowd_rms ratios, and is_speech (crowd loudness is only
    measured for single-speaker speeches, and not after multi_speaker
    segments). Returns None for a segment without valid times. The row
    also carries "_profile" and "_pitch", which the step script removes
    after using them."""
    seg = task["segment"]
    shm_s = shared_memory.SharedMemory(name=task["speech_shm"])
    shm_c = shared_memory.SharedMemory(name=task["crowd_shm"])
    y_speech = np.ndarray(task["shape"], dtype=task["dtype"], buffer=shm_s.buf)
    y_crowd = np.ndarray(task["shape"], dtype=task["dtype"], buffer=shm_c.buf)
    sr = config.FEATURES_SR
    try:
        start, end = safe_float(seg.get("start")), safe_float(seg.get("end"))
        if start is None or end is None or end <= start:
            return None
        multi_speaker = bool(seg.get("multi_speaker", False))
        words = seg.get("words", [])
        duration = round(end - start, 4)

        i0 = max(0, int(start * sr))
        y_seg = y_speech[i0:min(len(y_speech), int(end * sr))].copy()

        speech = egemaps(y_seg, sr)
        for key in config.NORMALIZED_SPEECH_KEYS:
            speech[f"{key}_norm"] = ratio_to_file(speech.get(key), task["file_speech"].get(key))
        pitch, t, f0, voiced_prob = pyin_pitch(y_seg, sr)
        gap = crowd_gap(y_crowd, sr, end, task["next_start"], task["file_crowd_loudness"],
                        measure_loudness=task["is_speech"] and not multi_speaker)

        row = {
            "seg_idx": task["seg_idx"],
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": seg.get("speaker"),
            "multi_speaker": multi_speaker,
            "reaction_end": _round_or_nan(gap["reaction_end"], 3),
            "duration": duration,
            "text": seg.get("text", "").strip(),
            **speech_rate(words, duration),
            "sentence_end": 1 if any(w.get("is_sentence_end") for w in words) else 0,
            "gap_duration": _round_or_nan(gap["gap_duration"], 3),
            "gap_duration_rob": _round_or_nan(gap["gap_duration_rob"], 3),
            "crowd_rms": _round_or_nan(gap["crowd_rms"], 6),
            "crowd_rms_rob": _round_or_nan(gap["crowd_rms_rob"], 6),
        }
        row.update(pitch)
        row.update(spectral_entropy(y_seg, sr))
        row.update(speech)
        row["_profile"] = spectral_profile(y_seg, sr)
        row["_pitch"] = (t + i0 / sr, f0, voiced_prob)
        return row
    finally:
        shm_s.close()
        shm_c.close()
