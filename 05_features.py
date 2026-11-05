"""
Step 05: prosodic and crowd-reaction features of every transcribed recording.

    conda activate crowd_audio
    python 05_features.py

For each recording in config.AUDIO_METADATA with a step-02 transcript (and,
if ACTIVE_CROWD = 1, its step-04 crowd regions):
  0. Speech quality: the step-03 estimate of the original recording, read
     from config.RECORDINGS_DONE_CSV (speech_quality, PESQ from 1 to 4.5;
     one value per recording). Gain: the speech and crowd signals are divided by the
     AMPLITUDE_REF_PERCENTILE (99th) percentile of the absolute amplitude of
     the original recording (one reference for both, same microphone), so
     all features are comparable across recordings with different gain.
  1. Rows: one table per unit (config.FEATURES_UNITS, both by default):
       chunk     one per Silero chunk, split where the speaker changes
                 inside it
       sentence  one per sentence of the transcript; its voiced pieces are
                 the Silero chunks it takes, cut to its own start and end
  2. Per row (in parallel): speech features of its voiced pieces only
     (eGeMAPS, duration, speech rate), and crowd features of its gaps:
     duration of the end gap (until the next row) and crowd_energy_end
     (crowd loudness in the end gap); for sentences also
     crowd_energy_around (sum over the gaps between its chunks and the end
     gap). A gap counts only if a CNN crowd region overlaps it (else 0).
     Crowd loudness only for recordings with ACTIVE_CROWD = 1.
  3. Per row: crowd_energy_end_rob of the previous row
     (crowd_energy_end_rob_lag; for the first one, the crowd before the
     speech starts).

Columns empty on every row (e.g. the crowd ones when ACTIVE_CROWD = 0) are
left out of the table.

Outputs (existing feature tables are skipped):
  config.FEATURES_DIR/<name>_chunk_features.csv     one row per chunk
  config.FEATURES_DIR/<name>_sentence_features.csv  one row per sentence
  config.RECORDINGS_DONE_CSV (next to the metadata)  speaker, duration, diarized
                                                   added to the recording's row
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from prosodic_features import config, recordings_done
from prosodic_features.audio import amplitude_percentile, find_audio_file, load_audio_mono
from prosodic_features.features.crowd_gap import cnn_gap_energy
from prosodic_features.labels import read_label_spans
from prosodic_features.metadata import Recording, load_recordings
from prosodic_features.segment_features import compute_segment_features
from prosodic_features.segments import load_chunks, load_sentences
from prosodic_features.utils import format_hms


def load_signals(speech_path: Path, crowd_path: Path):
    """Speech and crowd at FEATURES_SR, zero-padded to the same length."""
    sr = config.FEATURES_SR
    y_speech, _ = librosa.load(str(speech_path), sr=sr)
    y_crowd, _ = librosa.load(str(crowd_path), sr=sr)
    n = max(len(y_speech), len(y_crowd))
    return (np.pad(y_speech, (0, n - len(y_speech))).astype(np.float32),
            np.pad(y_crowd, (0, n - len(y_crowd))).astype(np.float32))


def run_segments_in_parallel(tasks: list[dict], t_start: float) -> list[dict]:
    rows = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=config.FEATURES_N_WORKERS) as executor:
        futures = {executor.submit(compute_segment_features, t): t["seg_idx"] for t in tasks}
        for n_done, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                print(f"    [seg {idx}] ERROR: {e}")
            if n_done % 20 == 0 or n_done == len(tasks):
                print(f"    {n_done}/{len(tasks)} rows done ({format_hms(time.time() - t_start)})")
    return [r for r in rows if r is not None]


def process_recording(rec: Recording, raw_path: Path, speech_path: Path, crowd_path: Path,
                      chunks_path: Path, transcript_path: Path, cnn_path: Path | None) -> None:
    stem, is_speech = rec.stem, rec.crowd
    todo = {}
    for unit in config.FEATURES_UNITS:
        csv = config.FEATURES_DIR / f"{stem}_{unit}_features.csv"
        if csv.exists():
            print(f"  [skip] features already exist: {csv.name}")
        else:
            todo[unit] = csv
    if not todo:
        return
    t_start = time.time()
    sr = config.FEATURES_SR

    print("  [1/4] Loading audio...")
    y_speech, y_crowd = load_signals(speech_path, crowd_path)
    y_raw = load_audio_mono(raw_path, sr)[0]
    sq = recordings_done.get(raw_path.name, "speech_quality")
    quality = np.nan if sq is None else float(sq)
    if sq is None:
        print(f"  [quality] WARNING: no speech quality for {raw_path.name} (run step 03)")
    q = config.AMPLITUDE_REF_PERCENTILE
    amplitude_ref = amplitude_percentile(y_raw, q)
    if not amplitude_ref > 0:
        raise ValueError(f"p{q} amplitude of {raw_path.name} is {amplitude_ref}: silent recording?")
    y_speech, y_crowd = y_speech / amplitude_ref, y_crowd / amplitude_ref   # gain normalization
    print(f"         Original amplitude p{q}: {amplitude_ref:.5f} (speech and crowd divided by it) | "
          f"Workers: {config.FEATURES_N_WORKERS}")

    chunks = read_label_spans(chunks_path)
    if not chunks:
        raise ValueError(f"No Silero chunks found in {chunks_path.name}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    cnn_regions = read_label_spans(cnn_path) if cnn_path is not None else []
    for unit, features_csv in todo.items():
        features_table(unit, features_csv, transcript, chunks, cnn_regions, is_speech,
                       y_speech, y_crowd, quality, t_start)
    recordings_done.update(raw_path.name, speaker=rec.speaker, duration=round(len(y_raw) / sr, 3),
                           diarized=int(rec.diarize))
    print(f"  [Saved] {config.RECORDINGS_DONE_CSV.name}")
    print(f"  [Done]  {format_hms(time.time() - t_start)}")


def features_table(unit: str, features_csv: Path, transcript: dict, chunks: list,
                   cnn_regions: list, is_speech: bool, y_speech: np.ndarray,
                   y_crowd: np.ndarray, quality: float, t_start: float) -> None:
    """Steps 1-3 of the module docstring for one unit; writes features_csv."""
    sr = config.FEATURES_SR
    print(f"  [2/4] Rows: one per {unit}, with their voiced pieces (Silero chunks)...")
    segments = (load_sentences if unit == "sentence" else load_chunks)(transcript, chunks)
    if not segments:
        raise ValueError(f"No {unit}s with words")
    print(f"         {len(segments)} {unit}s, "
          f"{sum(len(s['pieces']) for s in segments)} voiced pieces"
          + (f"; {len(cnn_regions)} CNN crowd regions" if is_speech else ""))

    if is_speech:
        print("  [3/4] Pre-speech crowd loudness...")
        pre_speech_crowd = cnn_gap_energy(y_crowd, sr, 0.0, segments[0]["start"],
                                          cnn_regions)["crowd_energy_rob"]
        print(f"         Pre-speech crowd loudness (rob): {pre_speech_crowd:.6f}")
    else:
        pre_speech_crowd = np.nan

    print(f"  [4/4] Extracting {unit} features ({config.FEATURES_N_WORKERS} workers)...")
    shm_speech = shared_memory.SharedMemory(create=True, size=y_speech.nbytes)
    shm_crowd = shared_memory.SharedMemory(create=True, size=y_crowd.nbytes)
    try:
        np.ndarray(y_speech.shape, dtype=y_speech.dtype, buffer=shm_speech.buf)[:] = y_speech
        np.ndarray(y_crowd.shape, dtype=y_crowd.dtype, buffer=shm_crowd.buf)[:] = y_crowd
        audio_end = len(y_speech) / sr
        tasks = [{
            "seg_idx": i, "segment": seg,
            "next_start": segments[i + 1].get("start") if i + 1 < len(segments) else audio_end,
            "is_speech": is_speech, "cnn_regions": cnn_regions, "unit": unit,
            "speech_shm": shm_speech.name, "crowd_shm": shm_crowd.name,
            "shape": y_speech.shape, "dtype": y_speech.dtype,
        } for i, seg in enumerate(segments)]
        rows = run_segments_in_parallel(tasks, t_start)
    finally:
        shm_speech.close(); shm_speech.unlink()
        shm_crowd.close(); shm_crowd.unlink()
    rows.sort(key=lambda r: r.pop("seg_idx"))

    df = pd.DataFrame(rows)
    df["crowd_energy_end_rob_lag"] = df["crowd_energy_end_rob"].shift(1)
    df.loc[df.index[0], "crowd_energy_end_rob_lag"] = pre_speech_crowd
    df.insert(df.columns.get_loc("speaker") + 1, "speech_quality", quality)
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    df.to_csv(features_csv, index=False)
    print(f"  [Saved] {features_csv.name} ({len(df)} rows, {len(df.columns)} cols)"
          + (f"; left out, empty: {', '.join(empty)}" if empty else ""))


def main() -> None:
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings()
    if not config.FEATURES_UNITS:
        raise ValueError("CROWD_FEATURES_UNIT must be 'chunk', 'sentence' or 'both'")
    print(f"[Batch] Tables: {', '.join(config.FEATURES_UNITS)} | Workers = {config.FEATURES_N_WORKERS}")

    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        transcript = config.TRANSCRIPT_DIR / f"{rec.stem}.json"
        chunks = config.TRANSCRIPT_DIR / f"{rec.stem}_silero_chunks.txt"
        speech = find_audio_file(config.DENOISED_DIR, config.DENOISED_PREFIX, rec.stem)
        crowd = find_audio_file(config.CROWD_DIR, config.CROWD_PREFIX, rec.stem)
        raw = find_audio_file(config.RAW_AUDIO_DIR, "", rec.stem)
        print(f"\n[{i}/{len(recordings)}] {rec.stem}  (crowd features={rec.crowd})")
        cnn = config.CROWD_CNN_DIR / f"{rec.stem}_crowd_cnn.txt" if rec.crowd else None
        missing = [p.name for p in (transcript, chunks, speech, crowd, raw, cnn)
                   if p is not None and not p.exists()]
        if missing:
            print(f"  [skip] not found: {', '.join(missing)}")
            continue
        try:
            process_recording(rec, raw, speech, crowd, chunks, transcript, cnn)
        except Exception as e:
            print(f"  [ERROR] {e}")
    print(f"\n[Batch Done] {len(recordings)} recordings | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
