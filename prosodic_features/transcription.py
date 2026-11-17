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
    torch tensor) and chunks is [{"start", "end"}, ...] in seconds, exact to
    the sample (Silero's return_seconds would round them to 0.1 s), cut to
    the first TRANSCRIBE_MAX_SECONDS if set.
    Chunks separated by less than SILERO_MERGE_GAP_S (often touching, gap
    0) are joined into one."""
    from silero_vad import get_speech_timestamps, read_audio

    audio = read_audio(str(audio_path))
    chunks = get_speech_timestamps(
        audio, vad_model,
        threshold=config.SILERO_THRESHOLD,
        min_speech_duration_ms=config.SILERO_MIN_SPEECH_MS,
        min_silence_duration_ms=config.SILERO_MIN_SILENCE_MS,
        speech_pad_ms=config.SILERO_SPEECH_PAD_MS,
    )
    chunks = [{"start": c["start"] / config.SILERO_SR, "end": c["end"] / config.SILERO_SR}
              for c in chunks]          # samples -> seconds
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


def transcribe(audio: np.ndarray, whisper_model, align_model, align_meta) -> list[dict]:
    """Transcribe the whole recording with WhisperX (it cuts it into its
    decoding windows with its own VAD, WHISPER_VAD_METHOD) and align the
    words on the recording itself, so each word lands where it was spoken.
    The Silero chunks are not used here: the words are fitted to them
    afterwards (handle_edge_words). audio: 16 kHz mono, cut to
    TRANSCRIBE_MAX_SECONDS if set.

    Returns [{start, end, text, words: [{start, end, word}, ...]}, ...]."""
    import whisperx

    audio = np.asarray(audio, dtype=np.float32)
    if config.TRANSCRIBE_MAX_SECONDS is not None:
        audio = audio[: int(config.TRANSCRIBE_MAX_SECONDS * config.SILERO_SR)]
    result = whisper_model.transcribe(audio, language=config.WHISPER_LANGUAGE)
    if not result.get("segments"):
        return []
    aligned = whisperx.align(result["segments"], align_model, align_meta,
                             audio, config.WHISPER_DEVICE,
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
            words.append({"start": ws, "end": we, "word": txt})
        out.append({"start": seg_start, "end": seg_end,
                    "text": (seg.get("text") or "").strip(), "words": words})
    return out


_WORD_RE = re.compile(r"[A-Za-z0-9'’]+")


def _normalize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _trim_stray_words(segment_text: str, words: list[dict], slack: int) -> list[dict]:
    """Drop stray words at the edges of a sentence that are not part of its
    text. Whisper sometimes hallucinates a few words on crowd noise or
    silence at a sentence's edge; they get timestamps but do not appear in
    its text. Windows of the sentence's own word count (+/- slack, to
    absorb tokenization differences such as "self-evident" being one word
    but two text tokens) slide over the words, and the contiguous span
    whose wording best matches segment_text is kept."""
    expected = _normalize_words(segment_text)
    if not expected or len(words) <= len(expected):
        return words
    n = len(expected)
    best_span, best_score = (0, len(words)), -1.0
    for length in range(max(1, n - slack), min(len(words), n + slack) + 1):
        for i in range(len(words) - length + 1):
            window_text = " ".join(w["word"] for w in words[i:i + length])
            score = difflib.SequenceMatcher(None, _normalize_words(window_text),
                                            expected).ratio()
            if score > best_score:
                best_score, best_span = score, (i, i + length)
    return words[best_span[0]:best_span[1]]


def handle_edge_words(sentences: list[dict], chunks: list[dict],
                      slack: int = config.TRIM_WORDS_SLACK) -> dict[str, int]:
    """Fit the sentences' words to the Silero chunks, IN PLACE:

    1. Stray words at each sentence's edges that are not in its text are
       dropped (_trim_stray_words).
    2. Each word belongs to the chunk it overlaps the most (word["chunk"],
       None if it lies in a gap). The first word of each chunk, if it starts
       before the chunk, starts at the chunk's start; the last word of each
       chunk, if it ends after the chunk, ends at the chunk's end. Other
       words are not changed. Words are copies, so the input word objects
       keep their original times.

    Sentences (and their times) are built later, in step 05, by
    prosodic_features/sentences.py.
    Returns counts: {"dropped", "moved", "in_gap"}."""
    counts = {"dropped": 0, "moved": 0, "in_gap": 0}

    def overlap(w, c):
        return min(w["end"], c["end"]) - max(w["start"], c["start"])

    members: dict[int, list[tuple[dict, int]]] = {}   # chunk -> [(sentence, word index)]
    for s in sentences:
        kept = _trim_stray_words(s["text"], s["words"], slack)
        counts["dropped"] += len(s["words"]) - len(kept)
        s["words"] = [dict(w) for w in kept]
        for wi, w in enumerate(s["words"]):
            k = max(range(len(chunks)), key=lambda i: overlap(w, chunks[i]))
            w["chunk"] = k if overlap(w, chunks[k]) > 0 else None
            if w["chunk"] is None:
                counts["in_gap"] += 1
            else:
                members.setdefault(k, []).append((s, wi))

    for k, ws in members.items():
        c = chunks[k]
        (s0, i0), (s1, i1) = ws[0], ws[-1]
        if s0["words"][i0]["start"] < c["start"]:
            s0["words"][i0]["start"] = c["start"]
            counts["moved"] += 1
        if s1["words"][i1]["end"] > c["end"]:
            s1["words"][i1]["end"] = c["end"]
            counts["moved"] += 1
    return counts


def trim_chunk_ends(chunks: list[dict], sentences: list[dict],
                    pad: float = config.CHUNK_END_PAD_S) -> list[dict]:
    """The chunks with each end moved back to the end of the chunk's last
    word plus pad seconds (after handle_edge_words, which sets
    word["chunk"]). Ends are only ever moved back, never forward; chunks
    without words keep their end. Returns new chunk dicts."""
    last_end: dict[int, float] = {}
    for s in sentences:
        for w in s["words"]:
            k = w.get("chunk")
            if k is not None:
                last_end[k] = max(last_end.get(k, w["end"]), w["end"])
    return [{"start": c["start"],
             "end": min(c["end"], round(last_end[k] + pad, 4)) if k in last_end else c["end"]}
            for k, c in enumerate(chunks)]
