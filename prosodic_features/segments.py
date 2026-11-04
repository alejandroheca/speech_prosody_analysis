"""Step 05: the rows of the features table (config.FEATURES_UNIT), each with
its voiced pieces:
  chunk     one row per Silero chunk, split where the speaker changes
            inside it (load_chunks)
  sentence  one row per sentence of the transcript (load_sentences)"""
from __future__ import annotations

from prosodic_features import config
from prosodic_features.utils import safe_float


def load_sentences(transcript: dict, chunks: list[tuple[float, float]]) -> list[dict]:
    """The transcript's sentences as [{start, end, speaker, multi_speaker,
    text, words, pieces}], by start.

    pieces are the sentence's voiced parts: each Silero chunk the sentence
    takes (sentence["chunks"], indices into chunks), cut to the sentence's
    own start and end, so a sentence inside a longer chunk keeps only its
    own part. A sentence without chunks gets its own span as one piece.
    multi_speaker: the sentence's words are not all from one speaker.
    Words without a speaker, and sentences without one, get
    config.DEFAULT_SPEAKER. Sentences without valid words are dropped."""
    sentences: list[dict] = []
    for seg in transcript.get("segments", []):
        words = []
        for w in seg.get("words", []):
            ws, we = safe_float(w.get("start")), safe_float(w.get("end"))
            txt = (w.get("word") or "").strip()
            if ws is not None and we is not None and txt:
                words.append({"start": ws, "end": we, "word": txt,
                              "speaker": w.get("speaker") or config.DEFAULT_SPEAKER})
        start, end = safe_float(seg.get("start")), safe_float(seg.get("end"))
        if not words or start is None or end is None or end <= start:
            continue
        pieces = []
        for k in seg.get("chunks", []):
            if 0 <= k < len(chunks):
                ps, pe = max(start, chunks[k][0]), min(end, chunks[k][1])
                if pe > ps:
                    pieces.append((ps, pe))
        sentences.append({
            "start": start, "end": end,
            "speaker": seg.get("speaker") or config.DEFAULT_SPEAKER,
            "multi_speaker": not seg.get("speaker_pure", True),
            "text": (seg.get("text") or "").strip(),
            "words": words,
            "pieces": sorted(pieces) or [(start, end)],
        })
    sentences.sort(key=lambda s: s["start"])
    return sentences


def load_chunks(transcript: dict, chunks: list[tuple[float, float]]) -> list[dict]:
    """One row per Silero chunk, as [{start, end, speaker, multi_speaker,
    text, words, pieces, sentence_end}], by start.

    Each word belongs to the chunk containing its midpoint. A chunk with one
    speaker gives one row with the chunk's own start and end. A chunk with a
    speaker change gives one row per run of same-speaker words, cut at the
    runs' word edges (the outer edges stay the chunk's), all with
    multi_speaker=True. Chunks without words are dropped. pieces is the row
    itself; sentence_end: the row contains the last word of a sentence."""
    words: list[dict] = []
    for seg in transcript.get("segments", []):
        valid = []
        for w in seg.get("words", []):
            ws, we = safe_float(w.get("start")), safe_float(w.get("end"))
            txt = (w.get("word") or "").strip()
            if ws is not None and we is not None and txt:
                valid.append({"start": ws, "end": we, "word": txt,
                              "speaker": w.get("speaker") or config.DEFAULT_SPEAKER,
                              "is_sentence_end": False})
        if valid:
            valid[-1]["is_sentence_end"] = True
        words.extend(valid)
    words.sort(key=lambda w: w["start"])

    rows: list[dict] = []
    for cs, ce in chunks:
        cw = [w for w in words if cs <= 0.5 * (w["start"] + w["end"]) <= ce]
        if not cw:
            continue
        runs: list[list[dict]] = [[cw[0]]]
        for w in cw[1:]:
            if w["speaker"] == runs[-1][-1]["speaker"]:
                runs[-1].append(w)
            else:
                runs.append([w])
        for k, run in enumerate(runs):
            start = cs if k == 0 else run[0]["start"]
            end = ce if k == len(runs) - 1 else run[-1]["end"]
            rows.append({
                "start": start, "end": end,
                "speaker": run[0]["speaker"],
                "multi_speaker": len(runs) > 1,
                "text": " ".join(w["word"] for w in run),
                "words": run,
                "pieces": [(start, end)],
                "sentence_end": int(any(w["is_sentence_end"] for w in run)),
            })
    rows.sort(key=lambda r: r["start"])
    return rows
