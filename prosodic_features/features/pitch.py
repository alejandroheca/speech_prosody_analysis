"""Pitch (f0) of the speech in a segment with pYIN (librosa)."""
from __future__ import annotations

import librosa
import numpy as np

from prosodic_features import config


def pyin_pitch(y: np.ndarray, sr: int):
    """Median and SD of f0 over the voiced frames, and the voiced fraction.

    Frame length is 1024 samples (shorter for very short segments), hop a
    quarter of it. If pYIN finds no voiced frame, the median / SD fall back
    to plain YIN over the frames inside [PYIN_FMIN, PYIN_FMAX].

    Returns (features, t, f0, voiced_prob): the frame track, with f0 NaN on
    frames that are neither voiced_flag nor voiced_prob >= PYIN_VOICED_PROB_MIN;
    t is relative to the start of y."""
    empty = {"pyin_pitch_hz": np.nan, "pyin_pitch_hz_sd": np.nan, "pyin_voiced_frac": np.nan}
    if len(y) < 16:
        return empty, np.array([]), np.array([]), np.array([])

    frame_length = min(1024, 2 ** int(np.log2(max(len(y), 256))))
    hop = frame_length // 4
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=config.PYIN_FMIN, fmax=config.PYIN_FMAX, sr=sr,
        frame_length=frame_length, hop_length=hop, fill_na=None,   # best guess everywhere
    )
    track_t = librosa.times_like(f0, sr=sr, hop_length=hop)
    track_f0 = np.where(voiced_flag | (voiced_prob >= config.PYIN_VOICED_PROB_MIN), f0, np.nan)

    voiced_frac = float(np.mean(voiced_flag)) if voiced_flag.size else np.nan
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced) == 0:
        f0_yin = librosa.yin(y, fmin=config.PYIN_FMIN, fmax=config.PYIN_FMAX, sr=sr,
                             frame_length=frame_length, hop_length=hop)
        voiced = f0_yin[(f0_yin > config.PYIN_FMIN) & (f0_yin < config.PYIN_FMAX)]

    frac = round(voiced_frac, 4) if not np.isnan(voiced_frac) else np.nan
    if len(voiced) == 0:
        feats = {**empty, "pyin_voiced_frac": frac}
    else:
        feats = {
            "pyin_pitch_hz": round(float(np.median(voiced)), 4),
            "pyin_pitch_hz_sd": round(float(np.std(voiced)), 4) if len(voiced) > 1 else np.nan,
            "pyin_voiced_frac": frac,
        }
    return feats, track_t, track_f0, voiced_prob
