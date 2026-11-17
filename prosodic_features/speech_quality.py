"""Step 03: speech quality without a reference: PESQ estimated by the
TorchAudio-Squim objective model on the denoised speech, either of a whole
recording from a random set of its Silero chunks (speech_quality), or of one
chunk / sentence on its own voiced pieces (segment_quality)."""
from __future__ import annotations

import random

import numpy as np

from prosodic_features import config

_MODEL = None


def _squim():
    global _MODEL
    if _MODEL is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        _MODEL = SQUIM_OBJECTIVE.get_model().eval()
    return _MODEL


def speech_quality(y: np.ndarray, chunks: list[dict]) -> float:
    """Perceived speech quality (PESQ, from 1 = bad to 4.5 = excellent: noise,
    distortion, band-limiting) of a recording (y: mono, SQUIM_SR), estimated
    by Squim on its Silero chunks ([{"start", "end"}, ...]): all of them if
    SQUIM_N_CHUNKS is None, else SQUIM_N_CHUNKS drawn at random with
    SQUIM_SEED, so a rerun draws the same ones (all of them if there are
    fewer). Chunks shorter than SQUIM_MIN_WINDOW_S are left out. Each chunk is cut into windows of at most SQUIM_WINDOW_S; shorter
    leftovers than SQUIM_MIN_WINDOW_S are skipped. The mean over windows
    weighted by duration; NaN if no window could be scored."""
    import torch
    sr = config.SQUIM_SR
    win, min_win = int(config.SQUIM_WINDOW_S * sr), int(config.SQUIM_MIN_WINDOW_S * sr)
    usable = [c for c in chunks if c["end"] - c["start"] >= config.SQUIM_MIN_WINDOW_S]
    n = config.SQUIM_N_CHUNKS
    drawn = usable if n is None else random.Random(config.SQUIM_SEED).sample(usable, min(n, len(usable)))
    print(f"         {len(drawn)} of {len(chunks)} chunks scored: "
          f"{sum(c['end'] - c['start'] for c in drawn):.1f} s of speech")
    scores, weights = [], []
    model = _squim()
    for c in drawn:
        a, b = max(0, int(c["start"] * sr)), min(len(y), int(c["end"] * sr))
        for s in range(a, b, win):
            piece = np.ascontiguousarray(y[s:min(b, s + win)], dtype=np.float32)
            if len(piece) < min_win:
                continue
            with torch.inference_mode():
                _, pesq, _ = model(torch.from_numpy(piece)[None, :])   # (stoi, pesq, si_sdr)
            scores.append(float(pesq))
            weights.append(len(piece))
    if not scores:
        return float("nan")
    return round(float(np.average(scores, weights=weights)), 4)


def _score_windows(y: np.ndarray) -> tuple[list[float], list[int]]:
    """Squim PESQ of y (mono, SQUIM_SR) in windows of at most SQUIM_WINDOW_S;
    windows shorter than SQUIM_MIN_WINDOW_S are skipped. (scores, lengths)."""
    import torch
    sr = config.SQUIM_SR
    win, min_win = int(config.SQUIM_WINDOW_S * sr), int(config.SQUIM_MIN_WINDOW_S * sr)
    scores, weights = [], []
    model = _squim()
    for s in range(0, len(y), win):
        piece = np.ascontiguousarray(y[s:s + win], dtype=np.float32)
        if len(piece) < min_win:
            continue
        with torch.inference_mode():
            _, pesq, _ = model(torch.from_numpy(piece)[None, :])   # (stoi, pesq, si_sdr)
        scores.append(float(pesq))
        weights.append(len(piece))
    return scores, weights


def segment_quality(y: np.ndarray, pieces: list[tuple[float, float]]) -> float:
    """Speech quality (PESQ) of one row (chunk or sentence): its voiced
    pieces (start, end) of y (mono, SQUIM_SR) joined, as for its eGeMAPS
    features in step 05, scored in windows (see _score_windows); the mean
    over windows weighted by duration. NaN if the row is shorter than
    SQUIM_MIN_WINDOW_S."""
    sr = config.SQUIM_SR
    joined = np.concatenate([y[max(0, int(a * sr)):min(len(y), int(b * sr))] for a, b in pieces])
    scores, weights = _score_windows(joined)
    if not scores:
        return float("nan")
    return round(float(np.average(scores, weights=weights)), 4)
