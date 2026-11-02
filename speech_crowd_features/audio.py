"""Reading audio and locating the separated audio files."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_audio_mono(path: Path, target_sr: int,
                    max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Read an audio file as float32 mono (mean of the channels), resampled
    to target_sr and optionally cut to its first max_seconds."""
    import librosa          # imported here: step 02's environment has no librosa
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if max_seconds is not None:
        data = data[: int(max_seconds * sr)]

    return data.astype(np.float32), sr


def find_audio_file(folder: Path, prefix: str, stem: str) -> Path:
    """The separated audio of a recording: <prefix><stem>.flac, else
    <prefix><stem>.wav, else <stem>.flac, else <stem>.wav. Returns the
    first candidate if none exists (the caller checks .exists())."""
    candidates = [folder / f"{prefix}{stem}.flac", folder / f"{prefix}{stem}.wav",
                  folder / f"{stem}.flac", folder / f"{stem}.wav"]
    return next((p for p in candidates if p.exists()), candidates[0])
