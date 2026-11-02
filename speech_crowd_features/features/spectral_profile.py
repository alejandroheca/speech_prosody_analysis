"""Spectral profile (SPS) of a segment and its similarity to the previous
segment's: does the speaker keep the same timbre / register?"""
from __future__ import annotations

import librosa
import numpy as np

from speech_crowd_features import config


def _log_frequency_bins(sr: int):
    """FFT bins kept (SPS_FMIN_HZ-SPS_FMAX_HZ), the log-frequency bin
    (SPS_BINS_OCT per octave) of each, and the number of log bins."""
    fft_freqs = np.fft.rfftfreq(config.SPEC_N_FFT, d=1.0 / sr)
    mask = (fft_freqs >= config.SPS_FMIN_HZ) & (fft_freqs <= config.SPS_FMAX_HZ)
    cents = 1200.0 * np.log2(fft_freqs[mask] / config.SPS_FMIN_HZ)
    bin_width_cents = 1200.0 / config.SPS_BINS_OCT
    n_log_bins = int(np.ceil(1200.0 * np.log2(config.SPS_FMAX_HZ / config.SPS_FMIN_HZ)
                             / bin_width_cents)) + 1
    log_bin = np.clip(np.floor(cents / bin_width_cents).astype(int), 0, n_log_bins - 1)
    return mask, log_bin, n_log_bins


def _smooth(vec: np.ndarray) -> np.ndarray:
    """Gaussian smoothing with SD SPS_SIGMA_CENTS (in log bins)."""
    sigma_bins = config.SPS_SIGMA_CENTS / (1200.0 / config.SPS_BINS_OCT)
    if sigma_bins <= 0:
        return vec
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    return np.convolve(vec, kernel / kernel.sum(), mode="same")


def spectral_profile(y: np.ndarray, sr: int):
    """Long-term magnitude spectrum of y (mean STFT magnitude per FFT bin),
    summed into log-frequency bins, smoothed and scaled to unit length.
    None if y is shorter than one FFT frame, silent, or on error."""
    if len(y) < config.SPEC_N_FFT:
        return None
    try:
        S = np.abs(librosa.stft(y.astype(np.float64), n_fft=config.SPEC_N_FFT,
                                hop_length=config.SPEC_HOP))
        mask, log_bin, n_log_bins = _log_frequency_bins(sr)
        profile = np.zeros(n_log_bins, dtype=np.float64)
        np.add.at(profile, log_bin, S.mean(axis=1)[mask])
        profile = _smooth(profile)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else None
    except Exception as e:
        print(f"         [spectral_profile] error: {e}")
        return None


def similarity_to_previous(profiles: list) -> list[float]:
    """Cosine similarity of each segment's profile to the last available
    profile before it (segments without a profile are skipped over); NaN
    for the first segment and for segments without a profile."""
    out, prev = [], None
    for p in profiles:
        out.append(round(float(np.dot(p, prev)), 6) if p is not None and prev is not None
                   else np.nan)
        if p is not None:
            prev = p
    return out
