"""Step 03: speaker-homogeneous segments from the Silero chunks and words."""
from __future__ import annotations

from speech_crowd_features import config
from speech_crowd_features.utils import safe_float


def flatten_words(transcript: dict) -> list[dict]:
    """All words of a step-02 transcript JSON as one time-sorted list of
    {start, end, word, speaker, is_sentence_end}. Words without a speaker
    get config.DEFAULT_SPEAKER. is_sentence_end marks the last word of each
    WhisperX sentence, so the segment containing it knows it ends one."""
    words: list[dict] = []
    for seg in transcript.get("segments", []):
        valid: list[dict] = []
        for w in seg.get("words", []):
            ws, we = safe_float(w.get("start")), safe_float(w.get("end"))
            txt = (w.get("word") or "").strip()
            if ws is None or we is None or not txt:
                continue
            valid.append({"start": ws, "end": we, "word": txt,
                          "speaker": w.get("speaker") or config.DEFAULT_SPEAKER})
        for i, wd in enumerate(valid):
            wd["is_sentence_end"] = (i == len(valid) - 1)
        words.extend(valid)
    words.sort(key=lambda w: w["start"])
    return words


def _words_in_span(words: list[dict], start: float, end: float) -> list[dict]:
    """Words whose midpoint falls within [start, end], so each word belongs
    to exactly one chunk."""
    return [w for w in words if start <= 0.5 * (w["start"] + w["end"]) <= end]


def split_chunks_by_speaker(chunks: list[tuple[float, float]],
                            words: list[dict]) -> list[dict]:
    """One segment per run of same-speaker words inside each Silero chunk.

    A chunk with one speaker gives one segment with the chunk's own start
    and end. A chunk spanning a speaker change gives one segment per run,
    cut at the runs' word edges (the outer edges stay the chunk's), and all
    of them get multi_speaker=True. Chunks without words give one segment
    with no speaker and no words.

    Returns [{start, end, speaker, multi_speaker, text, words}], by start."""
    segments: list[dict] = []
    for (cs, ce) in chunks:
        cw = _words_in_span(words, cs, ce)
        if not cw:
            segments.append({"start": cs, "end": ce, "speaker": None,
                             "multi_speaker": False, "text": "", "words": []})
            continue

        runs: list[list[dict]] = []
        cur: list[dict] = [cw[0]]
        for w in cw[1:]:
            if w["speaker"] == cur[-1]["speaker"]:
                cur.append(w)
            else:
                runs.append(cur)
                cur = [w]
        runs.append(cur)

        for k, run in enumerate(runs):
            segments.append({
                "start": cs if k == 0 else run[0]["start"],
                "end": ce if k == len(runs) - 1 else run[-1]["end"],
                "speaker": run[0]["speaker"],
                "multi_speaker": len(runs) > 1,
                "text": " ".join(w["word"] for w in run),
                "words": run,
            })
    segments.sort(key=lambda s: s["start"])
    return segments


def merge_same_speaker_segments(segments: list[dict],
                                max_gap: float = config.SEGMENT_MERGE_GAP_S) -> list[dict]:
    """Merge consecutive segments of the same speaker whose gap is shorter
    than max_gap seconds (never across speakers). A merged segment is
    multi_speaker if any of its parts was."""
    merged = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        cur["words"] = list(segments[i].get("words", []))
        cur_end = safe_float(cur.get("end"))
        j = i + 1
        while j < len(segments):
            nxt = segments[j]
            gap = (safe_float(nxt.get("start")) or 0) - (cur_end or 0)
            if gap < max_gap and nxt.get("speaker") == cur.get("speaker"):
                cur["end"] = nxt.get("end", cur["end"])
                cur["text"] = (cur.get("text", "") + " " + nxt.get("text", "")).strip()
                cur["words"] = cur["words"] + list(nxt.get("words", []))
                cur["multi_speaker"] = bool(cur.get("multi_speaker")) or bool(nxt.get("multi_speaker"))
                cur_end = safe_float(nxt.get("end"))
                j += 1
            else:
                break
        merged.append(cur)
        i = j
    return merged
