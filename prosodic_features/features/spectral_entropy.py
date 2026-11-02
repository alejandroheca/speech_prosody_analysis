"""Spectral entropy of the speech in a segment."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.stats import entropy

from prosodic_features import config


def spectral_entropy(y: np.ndarray, sr: int) -> dict:
    """Mean and SD over frames of the Shannon entropy (nats) of the energy
    distribution across SPEC_ENT_BANDS mel bands (50-8000 Hz). NaN if the
    signal is shorter than one FFT frame (SPEC_N_FFT) or on error."""
    names = (f"{config.SPEECH_PREFIX}spectral_entropy",
             f"{config.SPEECH_PREFIX}spectral_entropy_sd")
    empty = {names[0]: np.nan, names[1]: np.nan}
    if len(y) < config.SPEC_N_FFT:
        return empty
    try:
        mel = librosa.feature.melspectrogram(
            y=y.astype(np.float64), sr=sr, n_mels=config.SPEC_ENT_BANDS,
            fmin=50, fmax=8000, power=2,
            n_fft=config.SPEC_N_FFT, hop_length=config.SPEC_HOP)
        props = mel / (mel.sum(axis=0, keepdims=True) + 1e-10)
        ent = entropy(props, axis=0)
        mean = float(np.mean(ent)) if ent.size else np.nan
        sd = float(np.std(ent)) if ent.size > 1 else np.nan
        return {names[0]: round(mean, 6) if not np.isnan(mean) else np.nan,
                names[1]: round(sd, 6) if not np.isnan(sd) else np.nan}
    except Exception as e:
        print(f"         [spectral_entropy] error: {e}")
        return empty
