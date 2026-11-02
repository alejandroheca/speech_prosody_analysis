"""Step 02: speech chunks (Silero VAD) and their words (WhisperX)."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np

from speech_crowd_features import config
from speech_crowd_features.utils import safe_float


def detect_speech_chunks(audio_path: Path, vad_model):
    """Speech chunks of a recording with Silero VAD (settings in config.py).

    Returns (audio, chunks): audio is the 16 kHz signal Silero read (a
    torch tensor) and chunks is [{"start", "end"}, ...] in seconds (Silero
    rounds them to 0.1 s), cut to the first TRANSCRIBE_MAX_SECONDS if set."""
    from silero_vad import get_speech_timestamps, read_audio

    audio = read_audio(str(audio_path))
    chunks = get_speech_timestamps(
        audio, vad_model,
        threshold=config.SILERO_THRESHOLD,
        min_speech_duration_ms=config.SILERO_MIN_SPEECH_MS,
        min_silence_duration_ms=config.SILERO_MIN_SILENCE_MS,
        return_seconds=True,
    )
    total = sum(float(c["end"]) - float(c["start"]) for c in chunks)
    print(f"         Detected {len(chunks)} segments | Total speech: {total:.1f}s")

    if config.TRANSCRIBE_MAX_SECONDS is not None:
        chunks = [c for c in chunks if float(c["start"]) < config.TRANSCRIBE_MAX_SECONDS]
        print(f"         Truncated to first {config.TRANSCRIBE_MAX_SECONDS}s: "
              f"{len(chunks)} segments remaining")
    return audio, [{"start": float(c["start"]), "end": float(c["end"])} for c in chunks]


def transcribe_chunk(chunk: np.ndarray, chunk_start: float,
                     whisper_model, align_model, align_meta) -> list[dict]:
    """Transcribe ONE speech chunk on its own with WhisperX and align its
    words. Chunks are never concatenated, so a chunk's words and timestamps
    never depend on its neighbours.

    Returns the chunk's WhisperX segments, timestamps shifted by
    chunk_start onto the recording's timeline:
    [{start, end, text, words: [{start, end, word}, ...]}, ...]"""
    import whisperx

    result = whisper_model.transcribe(chunk, language=config.WHISPER_LANGUAGE)
    if not result.get("segments"):
        return []
    aligned = whisperx.align(result["segments"], align_model, align_meta,
                             chunk, config.WHISPER_DEVICE,
                             return_char_alignments=False)
    out = []
    for seg in aligned.get("segments", []):
        seg_start = safe_float(seg.get("start"))
        seg_end = safe_float(seg.get("end"))
        if seg_start is None or seg_end is None:
            continue
        words = []
        for w in seg.get("words", []):
            ws = safe_float(w.get("start"))
            we = safe_float(w.get("end"))
            txt = (w.get("word") or "").strip()
            if ws is None or we is None or not txt:
                continue
            words.append({"start": round(ws + chunk_start, 4),
                          "end": round(we + chunk_start, 4),
                          "word": txt})
        out.append({"start": round(seg_start + chunk_start, 4),
                    "end": round(seg_end + chunk_start, 4),
                    "text": (seg.get("text") or "").strip(),
                    "words": words})
    return out


_WORD_RE = re.compile(r"[A-Za-z0-9'’]+")


def _normalize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def trim_words_to_text(segment_text: str, words: list[dict],
                       slack: int = config.TRIM_WORDS_SLACK) -> list[dict]:
    """Drop stray words at the edges of a WhisperX segment that are not part
    of the segment's own text.

    Whisper sometimes hallucinates a few words on the crowd noise or silence
    at a segment's edge; they get timestamps inside the segment but do not
    appear in its text. This slides windows of the segment's own word count
    (+/- slack, to absorb tokenization differences such as "self-evident"
    being one word but two text tokens) over the words and keeps the
    contiguous span whose wording best matches segment_text."""
    expected = _normalize_words(segment_text)
    if not expected or not words:
        return words
    n = len(expected)
    if len(words) <= n:
        return words

    best_span, best_score = (0, len(words)), -1.0
    for length in range(max(1, n - slack), min(len(words), n + slack) + 1):
        for i in range(len(words) - length + 1):
            window_text = " ".join(w["word"] for w in words[i:i + length])
            score = difflib.SequenceMatcher(None, _normalize_words(window_text),
                                            expected).ratio()
            if score > best_score:
                best_score, best_span = score, (i, i + length)
    return words[best_span[0]:best_span[1]]
