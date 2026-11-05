"""Step 03: speech quality of a recording, without a reference: PESQ
estimated by the TorchAudio-Squim objective model from the denoised speech
(step 01) alone, on its voiced Silero chunks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prosodic_features import config

_MODEL = None


def _squim():
    global _MODEL
    if _MODEL is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        _MODEL = SQUIM_OBJECTIVE.get_model().eval()
    return _MODEL


def _load_16k_mono(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = torch.from_numpy(data.mean(axis=1))
    if sr != config.SQUIM_SR:
        y = AF.resample(y, sr, config.SQUIM_SR)
    return y.numpy()


def speech_quality(speech_path: Path, chunks: list[tuple[float, float]]) -> float:
    """Perceived speech quality (PESQ, from 1 = bad to 4.5 = excellent: noise,
    distortion, band-limiting) of the denoised speech (speech_path),
    estimated by Squim on the Silero chunks only (speech, not silence or
    crowd-only stretches), each cut into windows of at most SQUIM_WINDOW_S;
    windows shorter than SQUIM_MIN_WINDOW_S are skipped. The mean over
    windows weighted by duration; NaN if no window could be scored."""
    import torch
    y = _load_16k_mono(speech_path)
    sr = config.SQUIM_SR
    win, min_win = int(config.SQUIM_WINDOW_S * sr), int(config.SQUIM_MIN_WINDOW_S * sr)
    scores, weights = [], []
    model = _squim()
    for cs, ce in chunks:
        a, b = max(0, int(cs * sr)), min(len(y), int(ce * sr))
        for s in range(a, b, win):
            piece = y[s:min(b, s + win)]
            if len(piece) < min_win:
                continue
            with torch.inference_mode():
                _, pesq, _ = model(torch.from_numpy(piece)[None, :])   # (stoi, pesq, si_sdr)
            scores.append(float(pesq))
            weights.append(len(piece))
    if not scores:
        return np.nan
    return round(float(np.average(scores, weights=weights)), 4)
