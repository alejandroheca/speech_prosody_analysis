"""
Step 03: speech quality of every recording.

    conda activate crowd_transcribe
    python 03_quality.py

Two modes, set by config.SQUIM_N_CHUNKS:

  a number (random sample): one score per recording. Silero VAD finds the
    speech chunks of the recording's denoised speech (step 01,
    config.DENOISED_DIR) and TorchAudio-Squim (objective model) estimates
    the perceived quality of the speech (PESQ, from 1 = bad to 4.5 =
    excellent) on that many of them, drawn at random, without a clean
    reference. Saved as recording metadata: the recording's row of
    config.RECORDINGS_DONE_CSV (next to the metadata file; speaker,
    duration, diarized and speech_quality); recordings that already have
    a speech_quality are skipped.

  None (no sample, every row): one score per chunk and per sentence, on
    its own voiced pieces of the denoised speech (the rows and pieces of
    step 05: segments.load_chunks / load_sentences on the step-02
    transcript and Silero chunks). Saved per row in
    config.QUALITY_DIR/<name>_<unit>_quality.csv (start, end,
    speech_quality), which step 05 adds to its tables as a speech_quality
    column (e.g. as a control). Existing files are skipped.

See prosodic_features/speech_quality.py. Needs the denoised speech (step
01); the per-row mode also the transcript (step 02). The original
recording's file name is the recording's key in RECORDINGS_DONE_CSV (it can
be in a subfolder of config.RAW_AUDIO_DIR).

Output:
  random sample  config.RECORDINGS_DONE_CSV: speech_quality per recording
  every row      config.QUALITY_DIR/<name>_chunk_quality.csv, <name>_sentence_quality.csv
"""
from __future__ import annotations

import json
import time

import pandas as pd

from prosodic_features import config, recordings_done
from prosodic_features.audio import find_audio_file, find_raw_audio
from prosodic_features.labels import read_label_spans
from prosodic_features.metadata import load_recordings
from prosodic_features.segments import load_chunks, load_sentences
from prosodic_features.speech_quality import segment_quality, speech_quality
from prosodic_features.transcription import detect_speech_chunks
from prosodic_features.utils import format_hms


def per_row_quality(stem: str, denoised) -> None:
    """Every-row mode: <stem>_<unit>_quality.csv for each unit of config.FEATURES_UNITS."""
    from silero_vad import read_audio
    transcript_path = config.TRANSCRIPT_DIR / f"{stem}.json"
    chunks_path = config.TRANSCRIPT_DIR / f"{stem}_silero_chunks.txt"
    if not (transcript_path.exists() and chunks_path.exists()):
        print(f"  [skip] no transcript (run step 02): {transcript_path.name}")
        return
    todo = {u: config.QUALITY_DIR / f"{stem}_{u}_quality.csv" for u in config.FEATURES_UNITS}
    todo = {u: p for u, p in todo.items() if not p.exists()}
    if not todo:
        print("  [skip] per-row speech quality already exists")
        return
    t0 = time.time()
    y = read_audio(str(denoised)).numpy()                     # 16 kHz = SQUIM_SR
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    chunks = read_label_spans(chunks_path)
    for unit, out in todo.items():
        rows = (load_sentences if unit == "sentence" else load_chunks)(transcript, chunks)
        rows = [r for r in rows if r["end"] > r["start"]]
        df = pd.DataFrame({"start": [round(r["start"], 3) for r in rows],
                           "end": [round(r["end"], 3) for r in rows],
                           "speech_quality": [segment_quality(y, r["pieces"]) for r in rows]})
        df.to_csv(out, index=False)
        print(f"  [Saved] {out.name}: {len(df)} {unit}s, {df.speech_quality.notna().sum()} scored "
              f"(mean PESQ {df.speech_quality.mean():.2f})  [{format_hms(time.time() - t0)}]")


def main() -> None:
    from silero_vad import load_silero_vad

    recordings = load_recordings()
    per_row = config.SQUIM_N_CHUNKS is None
    print("[Mode] " + ("every chunk and sentence scored on its own -> " + str(config.QUALITY_DIR)
                       if per_row else f"{config.SQUIM_N_CHUNKS} chunks drawn at random per recording -> "
                       f"{config.RECORDINGS_DONE_CSV.name}"))
    if per_row:
        config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    vad = None if per_row else load_silero_vad()
    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        print(f"\n[{i}/{len(recordings)}] {rec.stem}")
        raw = find_raw_audio(rec.stem)
        if raw is None:
            print(f"  [skip] not found in {config.RAW_AUDIO_DIR}: {rec.stem}")
            continue
        denoised = find_audio_file(config.DENOISED_DIR, config.DENOISED_PREFIX, rec.stem)
        if not denoised.exists() or denoised.stat().st_size == 0:
            print(f"  [skip] no denoised speech (run step 01): {denoised.name}")
            continue
        if per_row:
            try:
                per_row_quality(rec.stem, denoised)
            except Exception as e:
                print(f"  [ERROR] {e}")
            continue
        done = recordings_done.get(raw.name, "speech_quality")
        if done is not None:
            print(f"  [skip] speech quality already in {config.RECORDINGS_DONE_CSV.name}: {done:.2f}")
            continue
        t0 = time.time()
        try:
            audio, chunks = detect_speech_chunks(denoised, vad)
            audio = audio.numpy()
            quality = speech_quality(audio, chunks)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        recordings_done.update(raw.name, speaker=rec.speaker, diarized=int(rec.diarize),
                               duration=round(len(audio) / config.SILERO_SR, 3),
                               speech_quality=quality)
        print(f"  [Saved] speech quality (PESQ) {quality:.2f} in {config.RECORDINGS_DONE_CSV.name}  "
              f"[{format_hms(time.time() - t0)}]")
    print(f"\n[Batch Done] {len(recordings)} recordings | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    main()
