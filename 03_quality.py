"""
Step 03: speech quality of every transcribed recording.

    conda activate crowd_transcribe
    python 03_quality.py

For each recording in config.AUDIO_METADATA with step-02 speech chunks,
TorchAudio-Squim (objective model) estimates the perceived quality of the
speech (PESQ, from 1 = bad to 4.5 = excellent) from the original recording
alone, without a clean reference, on its voiced Silero chunks only (see
prosodic_features/speech_quality.py).

Output: the speech_quality column of config.RECORDINGS_DONE_CSV (next to the
metadata file); recordings that already have one are skipped.
"""
from __future__ import annotations

import time

from prosodic_features import config, recordings_done
from prosodic_features.audio import find_audio_file
from prosodic_features.labels import read_label_spans
from prosodic_features.metadata import load_recordings
from prosodic_features.speech_quality import speech_quality
from prosodic_features.utils import format_hms


def main() -> None:
    recordings = load_recordings()
    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        raw = find_audio_file(config.RAW_AUDIO_DIR, "", rec.stem)
        chunks_path = config.TRANSCRIPT_DIR / f"{rec.stem}_silero_chunks.txt"
        print(f"\n[{i}/{len(recordings)}] {rec.stem}")
        done = recordings_done.get(raw.name, "speech_quality")
        if done is not None:
            print(f"  [skip] speech quality already in {config.RECORDINGS_DONE_CSV.name}: {done:.2f}")
            continue
        missing = [p.name for p in (raw, chunks_path) if not p.exists()]
        if missing:
            print(f"  [skip] not found: {', '.join(missing)}")
            continue
        t0 = time.time()
        try:
            quality = speech_quality(raw, read_label_spans(chunks_path))
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        recordings_done.update(raw.name, speech_quality=quality)
        print(f"  [Saved] speech quality (PESQ) {quality:.2f} in {config.RECORDINGS_DONE_CSV.name}  "
              f"[{format_hms(time.time() - t0)}]")
    print(f"\n[Batch Done] {len(recordings)} recordings | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    main()
