"""Step 02: speech chunks (Silero VAD) and their words (WhisperX)."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np

from prosodic_features import config
from prosodic_features.utils import safe_float


def detect_speech_chunks(audio_path: Path, vad_model):
    """Speech chunks of a recording with Silero VAD (settings in config.py).

    Returns (audio, chunks): audio is the 16 kHz signal Silero read (a
    torch tensor) and chunks is [{"start", "end"}, ...] in seconds (Silero
    rounds them to 0.1 s), cut to the first TRANSCRIBE_MAX_SECONDS if set.
    Chunks separated by less than SILERO_MERGE_GAP_S (often touching, gap
    0) are joined into one."""
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
    n_detected = len(chunks)
    chunks = merge_close_chunks(chunks, config.SILERO_MERGE_GAP_S)
    print(f"         Detected {n_detected} segments, {len(chunks)} after joining those "
          f"< {config.SILERO_MERGE_GAP_S}s apart | Total speech: {total:.1f}s")

    if config.TRANSCRIBE_MAX_SECONDS is not None:
        chunks = [c for c in chunks if float(c["start"]) < config.TRANSCRIBE_MAX_SECONDS]
        print(f"         Truncated to first {config.TRANSCRIBE_MAX_SECONDS}s: "
              f"{len(chunks)} segments remaining")
    return audio, [{"start": float(c["start"]), "end": float(c["end"])} for c in chunks]


def merge_close_chunks(chunks: list[dict], max_gap: float) -> list[dict]:
    """Join consecutive chunks whose gap is shorter than max_gap seconds.
    chunks: [{"start", "end"}, ...] sorted by start."""
    merged: list[dict] = []
    for c in chunks:
        start, end = float(c["start"]), float(c["end"])
        if merged and start - merged[-1]["end"] < max_gap - 1e-9:
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"start": start, "end": end})
    return merged


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


def handle_edge_words(segment_text: str, words: list[dict], chunk_start: float,
                      chunk_end: float, slack: int = config.TRIM_WORDS_SLACK) -> list[dict]:
    """The edge words of a WhisperX segment, checked against its own text
    and its Silero chunk [chunk_start, chunk_end]:

    1. Stray words at the edges that are not part of the segment's text are
       dropped. Whisper sometimes hallucinates a few words on the crowd
       noise or silence at a segment's edge; they get timestamps inside the
       segment but do not appear in its text. Windows of the segment's own
       word count (+/- slack, to absorb tokenization differences such as
       "self-evident" being one word but two text tokens) slide over the
       words, and the contiguous span whose wording best matches
       segment_text is kept.
    2. The first word, if it starts before the chunk, starts at the chunk's
       start; the last word, if it ends after the chunk, ends at the chunk's
       end (the alignment can overshoot the audio it was given by a few
       tens of ms). Corrected words are copies, so the input words keep
       their original times."""
    expected = _normalize_words(segment_text)
    if expected and len(words) > len(expected):
        n = len(expected)
        best_span, best_score = (0, len(words)), -1.0
        for length in range(max(1, n - slack), min(len(words), n + slack) + 1):
            for i in range(len(words) - length + 1):
                window_text = " ".join(w["word"] for w in words[i:i + length])
                score = difflib.SequenceMatcher(None, _normalize_words(window_text),
                                                expected).ratio()
                if score > best_score:
                    best_score, best_span = score, (i, i + length)
        words = words[best_span[0]:best_span[1]]

    words = list(words)
    if words and words[0]["start"] < chunk_start:
        words[0] = {**words[0], "start": chunk_start}
    if words and words[-1]["end"] > chunk_end:
        words[-1] = {**words[-1], "end": chunk_end}
    return words
