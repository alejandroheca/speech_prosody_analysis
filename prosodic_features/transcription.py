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

    Sentence times are set afterwards by fit_sentences_to_chunks.
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


def fit_sentences_to_chunks(sentences: list[dict], chunks: list[dict],
                            max_fragment: float = config.SENTENCE_FRAGMENT_MAX_S) -> int:
    """Set each sentence's chunks and start / end, IN PLACE (after
    handle_edge_words, which sets word["chunk"]).

    A sentence can span several chunks. But where it shares its first (or
    last) chunk with the previous (or next) sentence and its own piece of
    that chunk is shorter than max_fragment seconds, the piece is an
    alignment fragment (e.g. the "100" of "100 years later." placed at the
    end of the chunk before): the sentence does not take that chunk, so the
    gap between the chunks stays the boundary between the two sentences.
    Word times are not changed.

    Then, per chunk, the first sentence that takes it starts at the chunk's
    start and the last one ends at the chunk's end; the other sentence
    boundaries inside a chunk follow their words. s["chunks"] lists the
    chunks each sentence takes. Returns how many fragments were found."""
    sents = [s for s in sentences if s["words"]]

    def piece(s, k):
        ws = [w for w in s["words"] if w.get("chunk") == k]
        return ws[-1]["end"] - ws[0]["start"] if ws else 0.0

    own = []
    for s in sents:
        ks = [w["chunk"] for w in s["words"] if w.get("chunk") is not None]
        own.append(sorted(set(ks)))
    n_fragments = 0
    for i, s in enumerate(sents):
        ks = own[i]
        if len(ks) < 2:
            continue
        first, last = ks[0], ks[-1]
        if i > 0 and first in own[i - 1] and piece(s, first) < max_fragment:
            ks = ks[1:]
            n_fragments += 1
        if i + 1 < len(sents) and last in own[i + 1] and len(ks) > 1 and piece(s, last) < max_fragment:
            ks = ks[:-1]
            n_fragments += 1
        own[i] = ks

    takers: dict[int, list[int]] = {}
    for i, ks in enumerate(own):
        for k in ks:
            takers.setdefault(k, []).append(i)
    for i, s in enumerate(sents):
        ks = own[i]
        s["chunks"] = ks
        in_own = [w for w in s["words"] if w.get("chunk") in ks] or s["words"]
        s["start"], s["end"] = in_own[0]["start"], in_own[-1]["end"]
        if ks:
            if takers[ks[0]][0] == i:
                s["start"] = chunks[ks[0]]["start"]
            if takers[ks[-1]][-1] == i:
                s["end"] = chunks[ks[-1]]["end"]
    return n_fragments


def split_at_long_pauses(sentences: list[dict],
                         min_gap: float = config.SENTENCE_SPLIT_MIN_GAP_S) -> tuple[list[dict], int]:
    """End a sentence at a pause of at least min_gap seconds between two of
    its words when the word before the pause has no punctuation: WhisperX
    sometimes stops punctuating for a while and runs several sentences into
    one ("...like a mighty stream [9.6 s] i am not unmindful..."). A comma
    before the pause means the sentence goes on ("my friends, [applause]
    so even though..."), so it is kept. The word before the pause gets a
    "." and the first word after it a capital; each part's text is its
    words. Run on WhisperX's sentences, before handle_edge_words.
    Returns (sentences, number of splits)."""
    out: list[dict] = []
    n_split = 0
    for s in sentences:
        words, start = s["words"], 0
        for i in range(1, len(words)):
            prev = words[i - 1]["word"]
            if (words[i]["start"] - words[i - 1]["end"] >= min_gap
                    and prev[-1:].isalnum()):
                words[i - 1] = {**words[i - 1], "word": prev + "."}
                nxt = words[i]["word"]
                words[i] = {**words[i], "word": nxt[:1].upper() + nxt[1:]}
                out.append(_sentence_of(words[start:i]))
                start = i
                n_split += 1
        out.append(_sentence_of(words[start:]) if start else s)
    return out, n_split


def _sentence_of(words: list[dict]) -> dict:
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words), "words": words}


def _main_speaker(sentence: dict):
    speakers = [w.get("speaker") for w in sentence["words"]]
    return max(set(speakers), key=speakers.count) if speakers else None


def join_unfinished_sentences(sentences: list[dict],
                              max_gap: float = config.SENTENCE_JOIN_MAX_GAP_S,
                              end_chars: str = config.SENTENCE_END_CHARS) -> tuple[list[dict], int]:
    """Join a sentence with the next one when it is unfinished: its text
    does not end in one of end_chars, or the next one starts in lowercase
    (whatever the punctuation before it: WhisperX puts a "." or "..." at
    a pause inside a sentence, as in "This is no time." / "to engage in
    ..."). Only when both have the same (main) speaker and the next one's
    first word starts less than max_gap seconds after its last word ends.
    Joined, the sentence's end is the real sentence end; before a lowercase
    start, the period(s) at the pause are dropped from the text.
    Repeats, so a sentence cut twice is joined whole. Run after
    handle_edge_words (sentences with words, speakers set) and before
    fit_sentences_to_chunks. Returns (sentences, number of joins)."""
    out: list[dict] = []
    n_joined = 0
    for s in sentences:
        if out:
            prev = out[-1]
            text, nxt = prev["text"].strip(), s["text"].strip()
            unfinished = not text.endswith(tuple(end_chars)) or nxt[:1].islower()
            gap = s["words"][0]["start"] - prev["words"][-1]["end"]
            if unfinished and gap < max_gap and _main_speaker(prev) == _main_speaker(s):
                if nxt[:1].islower():
                    text = text.rstrip(".")     # WhisperX's period at the pause
                prev["text"] = f"{text} {nxt}"
                prev["words"] = prev["words"] + s["words"]
                prev["end"] = s["end"]
                n_joined += 1
                continue
        out.append(dict(s))
    return out, n_joined
