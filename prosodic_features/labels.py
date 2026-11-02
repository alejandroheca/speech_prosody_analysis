"""Audacity label tracks: text files with one 'start<TAB>end<TAB>label'
line per label, times in seconds. They open in Audacity or Sonic
Visualiser over the audio, so every step's output can be checked by ear."""
from __future__ import annotations

from pathlib import Path

from prosodic_features.utils import safe_float


def write_audacity_labels(path: Path, labels: list[tuple[float, float, str]]) -> None:
    """Write (start, end, label) tuples, times with 6 decimals."""
    lines = [f"{start:.6f}\t{end:.6f}\t{label}" for start, end, label in labels]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def read_silero_chunks(path: Path) -> list[tuple[float, float]]:
    """[(start, end), ...] of a <stem>_silero_chunks.txt label track, sorted
    by start; lines without valid times (or with end <= start) are skipped."""
    chunks: list[tuple[float, float]] = []
    if not path.exists():
        return chunks
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        start, end = safe_float(parts[0]), safe_float(parts[1])
        if start is None or end is None or end <= start:
            continue
        chunks.append((start, end))
    chunks.sort(key=lambda c: c[0])
    return chunks
