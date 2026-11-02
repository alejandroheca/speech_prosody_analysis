"""Step 02: the speaker of each word and transcript segment."""
from __future__ import annotations

import json
from pathlib import Path

from prosodic_features import config


def load_speaker_turns(stem: str, diarization_dir: Path) -> list[dict] | None:
    """Speaker turns [{"start", "end", "speaker"}, ...] of a recording from
    its diarization JSON, sorted by start; None if there is no JSON.
    Looks for <stem>_diarization.json or <stem>.diarization.json, with and
    without the 'denoised_' prefix."""
    stems = [stem]
    if not stem.startswith(config.DENOISED_PREFIX):
        stems.append(f"{config.DENOISED_PREFIX}{stem}")
    for s in stems:
        for name in (f"{s}_diarization.json", f"{s}.diarization.json"):
            path = diarization_dir / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            turns = []
            for t in data.get("segments") or data.get("turns") or []:
                start, end = t.get("start"), t.get("end")
                if start is None or end is None:
                    continue
                turns.append({"start": float(start), "end": float(end),
                              "speaker": str(t.get("speaker"))})
            turns.sort(key=lambda t: t["start"])
            return turns
    return None


def _speaker_with_most_overlap(start: float, end: float, turns: list[dict]) -> str | None:
    """Speaker whose turn overlaps [start, end] the most (None if none does)."""
    best_speaker, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, turn["speaker"]
    return best_speaker


def assign_word_speakers(words: list[dict], turns: list[dict] | None,
                         constant: str | None) -> None:
    """Set each word's "speaker" IN PLACE:
      - constant given (single-speaker recording): every word gets it;
      - turns given: the speaker whose turn overlaps the word the most
        (None if no turn overlaps it);
      - neither: None for every word."""
    for w in words:
        if constant is not None:
            w["speaker"] = constant
        elif turns:
            w["speaker"] = _speaker_with_most_overlap(w["start"], w["end"], turns)
        else:
            w["speaker"] = None


def set_segment_speaker(segment: dict) -> None:
    """Set segment["speaker"] to the most common speaker among its words and
    segment["speaker_pure"] to whether all its (labelled) words agree,
    IN PLACE. No labelled words: speaker None, speaker_pure True."""
    speakers = [w.get("speaker") for w in segment.get("words", [])
                if w.get("speaker") is not None]
    if not speakers:
        segment["speaker"], segment["speaker_pure"] = None, True
        return
    unique = set(speakers)
    segment["speaker"] = max(unique, key=speakers.count)
    segment["speaker_pure"] = len(unique) == 1
