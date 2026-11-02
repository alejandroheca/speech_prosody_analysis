"""
Step 03: acoustic and crowd-reaction features of every transcribed recording.

    conda activate crowd_audio
    python 03_features.py

For each recording in config.AUDIO_METADATA with a step-02 transcript:
  1. Segments: the Silero chunks, split where the speaker changes inside a
     chunk, wordless pieces dropped, then same-speaker segments less than
     SEGMENT_MERGE_GAP_S apart merged.
  2. Per segment (in parallel): speech features of the segment (eGeMAPS,
     pYIN pitch, spectral entropy, speech rate) and crowd features of the
     gap after it (duration; crowd loudness relative to the whole file,
     only for recordings with crowd features, see metadata.py).
  3. Per segment, sequential: spectral similarity to the previous segment
     and the crowd loudness of the previous gap (crowd_rms_rob_lag; for the
     first segment, the crowd before the speech starts).
  4. The metadata columns in METADATA_COPY_COLUMNS are added to every row.

Outputs (recordings with both are skipped):
  config.FEATURES_DIR/<name>_utterance_features.csv   one row per segment
  config.PITCH_DIR/<name>_pitch.csv                   pYIN frames of every segment
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

from speech_crowd_features import config
from speech_crowd_features.audio import find_audio_file
from speech_crowd_features.features.opensmile_features import crowd_loudness, egemaps, ratio_to_file
from speech_crowd_features.features.spectral_profile import similarity_to_previous
from speech_crowd_features.labels import read_silero_chunks
from speech_crowd_features.metadata import Recording, load_recordings
from speech_crowd_features.segment_features import compute_segment_features
from speech_crowd_features.segments import (flatten_words, merge_same_speaker_segments,
                                     split_chunks_by_speaker)
from speech_crowd_features.utils import format_hms

PITCH_COLUMNS = ["seg_start", "seg_end", "speaker", "sentence_end", "t", "f0_hz", "voiced_prob"]


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
                print(f"    {n_done}/{len(tasks)} segments done ({format_hms(time.time() - t_start)})")
    return [r for r in rows if r is not None]


def process_recording(rec: Recording, speech_path: Path, crowd_path: Path,
                      chunks_path: Path, transcript_path: Path) -> None:
    stem, is_speech = rec.stem, rec.crowd
    features_csv = config.FEATURES_DIR / f"{stem}_utterance_features.csv"
    pitch_csv = config.PITCH_DIR / f"{stem}_pitch.csv"
    if features_csv.exists() and pitch_csv.exists():
        print(f"  [skip] features + pitch already exist: {features_csv.name}")
        return
    t_start = time.time()
    sr = config.FEATURES_SR

    print("  [1/4] Loading audio...")
    y_speech, y_crowd = load_signals(speech_path, crowd_path)
    file_crowd_loudness = crowd_loudness(y_crowd, sr)   # whole-file baselines
    file_speech = egemaps(y_speech, sr)
    print(f"         Crowd loudness (file): {file_crowd_loudness:.4f} | "
          f"Workers: {config.FEATURES_N_WORKERS}")

    print("  [2/4] Building speaker-homogeneous segments from Silero chunks...")
    chunks = read_silero_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No Silero chunks found in {chunks_path.name}")
    words = flatten_words(json.loads(transcript_path.read_text(encoding="utf-8")))
    segments = [s for s in split_chunks_by_speaker(chunks, words) if s.get("words")]
    segments = merge_same_speaker_segments(segments)
    print(f"         {len(segments)} segments (speaker-homogeneous)")

    print("  [3/4] Pre-speech crowd loudness...")
    first_start = segments[0].get("start") if segments else 0.0
    if first_start and first_start > 0:
        i_end = min(len(y_crowd), int(first_start * sr))
        i_start = min(int(config.REVERB_LAG_S * sr), i_end)
        pre_speech_crowd = ratio_to_file(crowd_loudness(y_crowd[i_start:i_end], sr),
                                         file_crowd_loudness)
    else:
        pre_speech_crowd = np.nan
    print(f"         Pre-speech crowd loudness (rob): {pre_speech_crowd:.6f}")

    print(f"  [4/4] Extracting features ({config.FEATURES_N_WORKERS} workers)...")
    shm_speech = shared_memory.SharedMemory(create=True, size=y_speech.nbytes)
    shm_crowd = shared_memory.SharedMemory(create=True, size=y_crowd.nbytes)
    try:
        np.ndarray(y_speech.shape, dtype=y_speech.dtype, buffer=shm_speech.buf)[:] = y_speech
        np.ndarray(y_crowd.shape, dtype=y_crowd.dtype, buffer=shm_crowd.buf)[:] = y_crowd
        audio_end = len(y_speech) / sr
        tasks = [{
            "seg_idx": i, "segment": seg,
            "next_start": segments[i + 1].get("start") if i + 1 < len(segments) else audio_end,
            "file_crowd_loudness": file_crowd_loudness, "file_speech": file_speech,
            "is_speech": is_speech,
            "speech_shm": shm_speech.name, "crowd_shm": shm_crowd.name,
            "shape": y_speech.shape, "dtype": y_speech.dtype,
        } for i, seg in enumerate(segments)]
        rows = run_segments_in_parallel(tasks, t_start)
    finally:
        shm_speech.close(); shm_speech.unlink()
        shm_crowd.close(); shm_crowd.unlink()
    rows.sort(key=lambda r: r.pop("seg_idx"))

    # sequential features
    similarity = similarity_to_previous([r.pop("_profile") for r in rows])
    pitch_frames = []
    for r, sim in zip(rows, similarity):
        r[f"{config.SPEECH_PREFIX}spectral_similarity_prev"] = sim
        t, f0, voiced_prob = r.pop("_pitch")
        if len(t):
            pitch_frames.append(pd.DataFrame({
                "seg_start": r["start"], "seg_end": r["end"], "speaker": r["speaker"],
                "sentence_end": r["sentence_end"], "t": np.round(t, 4),
                "f0_hz": np.round(f0, 3), "voiced_prob": np.round(voiced_prob, 4)}))

    config.PITCH_DIR.mkdir(parents=True, exist_ok=True)
    (pd.concat(pitch_frames, ignore_index=True) if pitch_frames
     else pd.DataFrame(columns=PITCH_COLUMNS)).to_csv(pitch_csv, index=False)
    print(f"  [Saved] {pitch_csv.name}")

    df = pd.DataFrame(rows)
    df["crowd_rms_rob_lag"] = df["crowd_rms_rob"].shift(1)
    df.loc[df.index[0], "crowd_rms_rob_lag"] = pre_speech_crowd
    for sheet_col, csv_col in config.METADATA_COPY_COLUMNS.items():
        df[csv_col] = rec.row.get(sheet_col, "")
    df.to_csv(features_csv, index=False)
    print(f"  [Saved] {features_csv.name} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"  [Done]  {format_hms(time.time() - t_start)}")


def main() -> None:
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings()
    print(f"[Batch] Workers = {config.FEATURES_N_WORKERS}")

    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        transcript = config.TRANSCRIPT_DIR / f"{rec.stem}.json"
        chunks = config.TRANSCRIPT_DIR / f"{rec.stem}_silero_chunks.txt"
        speech = find_audio_file(config.DENOISED_DIR, config.DENOISED_PREFIX, rec.stem)
        crowd = find_audio_file(config.CROWD_DIR, config.CROWD_PREFIX, rec.stem)
        print(f"\n[{i}/{len(recordings)}] {rec.stem}  (crowd features={rec.crowd})")
        missing = [p.name for p in (transcript, chunks, speech, crowd) if not p.exists()]
        if missing:
            print(f"  [skip] not found: {', '.join(missing)}")
            continue
        try:
            process_recording(rec, speech, crowd, chunks, transcript)
        except Exception as e:
            print(f"  [ERROR] {e}")
    print(f"\n[Batch Done] {len(recordings)} recordings | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
