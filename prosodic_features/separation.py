"""Step 01: split each recording into speech and crowd.

Speech is the output of GTCRN, a 16 kHz speech-enhancement model (ONNX
version from github.com/yuyun2000/SpeechDenoiser). Crowd is the residual:
the original mix minus the speech."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Tuple

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from prosodic_features import config
from prosodic_features.audio import load_audio_mono


def load_gtcrn_denoiser(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load the GTCRN model and return a function that maps a 16 kHz mono
    signal to its enhanced (speech-only) version.

    The model is streaming: the STFT (n_fft 512, hop 256) is fed to it one
    frame at a time, carrying its three internal caches between frames."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
    if not {"mix", "conv_cache", "tra_cache", "inter_cache"}.issubset(model_inputs):
        raise RuntimeError(f"Not a GTCRN model. Inputs: {list(model_inputs.keys())}")

    def denoise(audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=512, hop_length=256)
        real, imag = stft.real, stft.imag
        n_frames = stft.shape[1]
        enhanced_stft = np.zeros_like(stft, dtype=np.complex64)

        conv = np.zeros(model_inputs["conv_cache"], dtype=np.float32)
        tra = np.zeros(model_inputs["tra_cache"], dtype=np.float32)
        inter = np.zeros(model_inputs["inter_cache"], dtype=np.float32)

        for i in range(n_frames):
            frame = np.stack([real[:, i], imag[:, i]], axis=-1)[None, :, None, :]
            out, conv, tra, inter = sess.run(
                None,
                {
                    "mix": frame.astype(np.float32),
                    "conv_cache": conv,
                    "tra_cache": tra,
                    "inter_cache": inter,
                },
            )
            out = out.squeeze()
            enhanced_stft[:, i] = out[:, 0] + 1j * out[:, 1]

        enhanced = librosa.istft(enhanced_stft, hop_length=256)
        return enhanced.astype(np.float32)

    return denoise


def separate_speech_crowd(audio_path: Path, denoise: Callable[[np.ndarray], np.ndarray],
                          speech_path: Path, crowd_path: Path) -> Tuple[bool, float, float]:
    """Write the speech and crowd versions of one recording (mono 16 kHz).
    Skips the recording if both files exist and SEPARATE_SKIP_EXISTING.

    Returns (skipped, audio_duration_s, processing_time_s)."""
    t0 = time.time()
    if config.SEPARATE_SKIP_EXISTING and speech_path.exists() and crowd_path.exists():
        return True, 0.0, 0.0

    audio, sr = load_audio_mono(audio_path, config.SEPARATE_SR, config.SEPARATE_MAX_SECONDS)
    duration = len(audio) / sr

    speech = denoise(audio)
    n = min(len(audio), len(speech))
    audio, speech = audio[:n], speech[:n]
    crowd = audio - speech

    sf.write(str(speech_path), speech, config.SEPARATE_SR)
    sf.write(str(crowd_path), crowd, config.SEPARATE_SR)
    return False, duration, time.time() - t0


# ---- parallel workers: each loads its own model once ----
_DENOISE = None


def init_worker(model_path: Path) -> None:
    global _DENOISE
    _DENOISE = load_gtcrn_denoiser(model_path)


def separate_in_worker(paths: tuple[Path, Path, Path]):
    """separate_speech_crowd for (audio_path, speech_path, crowd_path) in a
    worker started with init_worker. Returns its result, or the error text."""
    try:
        return separate_speech_crowd(paths[0], _DENOISE, paths[1], paths[2])
    except Exception as e:
        return f"{type(e).__name__}: {e}"
