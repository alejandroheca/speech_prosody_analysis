"""
Step 04: crowd reactions detected by the CNN, in recordings with an active
crowd.

    conda activate crowd_transcribe
    python 04_crowd_detection.py

Needs a clone of github.com/chymaera96/crowd_reaction_analysis (with its
cnn_tcn weights) next to this repository, or at CROWD_CNN_REPO.

For each recording in config.AUDIO_METADATA with ACTIVE_CROWD = 1, the
CNN-TCN runs on the original recording (data/audio/raw) and its crowd
regions with a mean probability above CNN_MIN_MEAN_SCORE are saved. Step 05
measures crowd loudness only in the gaps these regions overlap.

Output (existing outputs are skipped):
  config.CROWD_CNN_DIR/<name>_crowd_cnn.txt   Audacity labels: "crowd mean=<p>"
"""
from __future__ import annotations

import time

from prosodic_features import config
from prosodic_features.audio import find_audio_file
from prosodic_features.crowd_detection import detect_crowd
from prosodic_features.labels import write_audacity_labels
from prosodic_features.metadata import load_recordings
from prosodic_features.utils import format_hms


def main() -> None:
    config.CROWD_CNN_DIR.mkdir(parents=True, exist_ok=True)
    recordings = [r for r in load_recordings() if r.crowd]
    print(f"[Batch] {len(recordings)} recordings with ACTIVE_CROWD = 1")
    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        out = config.CROWD_CNN_DIR / f"{rec.stem}_crowd_cnn.txt"
        raw = find_audio_file(config.RAW_AUDIO_DIR, "", rec.stem)
        print(f"\n[{i}/{len(recordings)}] {rec.stem}")
        if out.exists():
            print(f"  [skip] already exists: {out.name}")
            continue
        if not raw.exists():
            print(f"  [missing] no audio for {rec.stem} in {config.RAW_AUDIO_DIR}")
            continue
        t0 = time.time()
        try:
            regions = detect_crowd(raw)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        write_audacity_labels(out, [(s, e, f"crowd mean={p:.2f}") for s, e, p in regions])
        print(f"  [Saved] {out.name}: {len(regions)} crowd regions "
              f"({sum(e - s for s, e, _ in regions):.1f} s)  [{format_hms(time.time() - t0)}]")
    print(f"\n[Batch Done] {len(recordings)} recordings | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    main()
