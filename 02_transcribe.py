"""
Step 02: speech chunks, words and speakers of every recording.

    conda activate crowd_transcribe
    export HF_TOKEN=hf_...      # only needed if a recording needs diarization
    python 02_transcribe.py

For each recording in config.AUDIO_METADATA (its speech file from step 01):
  1. Silero VAD finds the speech chunks.
  2. WhisperX transcribes and word-aligns each chunk on its own.
  3. Speakers: single-speaker recordings get SPEAKER_00 for every word.
     Recordings with DIARIZATION_NEEDED = 1 in the metadata are
     diarized with pyannote (once; the result is saved and reused) and
     each word gets the speaker whose turn overlaps it the most.
  4. Stray words at the edges of each WhisperX sentence that are not in
     its text are dropped (trim_words_to_text).

Outputs, per recording (config.TRANSCRIPT_DIR; recordings with a JSON are
skipped):
  <name>.json                 sentences with their words and speakers
  <name>_silero_chunks.txt    Audacity labels: VAD chunks
  <name>_words.txt            Audacity labels: words (after step 4)
  <name>_words_original.txt   Audacity labels: words (before step 4)
  <name>_sentences.txt        Audacity labels: sentences
and, for diarized recordings (config.DIARIZATION_DIR):
  denoised_<name>_diarization.json, denoised_<name>.rttm
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from speech_crowd_features import config
from speech_crowd_features.labels import write_audacity_labels
from speech_crowd_features.audio import find_audio_file
from speech_crowd_features.metadata import clean_stem, load_recordings
from speech_crowd_features.speakers import (assign_word_speakers, load_speaker_turns,
                                     set_segment_speaker)
from speech_crowd_features.transcription import (detect_speech_chunks, transcribe_chunk,
                                          trim_words_to_text)
from speech_crowd_features.utils import format_hms


def _speaker_label(item: dict) -> str:
    return f" [{item['speaker']}]" if item.get("speaker") else ""


def diarize_if_missing(audio_path: Path, diarizer: dict) -> None:
    """Run pyannote on a recording unless its diarization JSON exists.
    The pipeline is loaded on first use and kept in diarizer."""
    if load_speaker_turns(audio_path.stem, config.DIARIZATION_DIR) is not None:
        return
    from speech_crowd_features.diarization import diarize_file, load_diarization_pipeline

    if "pipeline" not in diarizer:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not set. Run:  export HF_TOKEN=hf_...  (and accept "
                             f"the terms of {config.DIARIZATION_MODEL_ID} on Hugging Face)")
        diarizer["pipeline"], diarizer["device"] = load_diarization_pipeline(token)
    _, diarizer["device"] = diarize_file(
        diarizer["pipeline"], diarizer["device"], audio_path,
        out_json=config.DIARIZATION_DIR / f"{audio_path.stem}_diarization.json",
        out_rttm=config.DIARIZATION_DIR / f"{audio_path.stem}.rttm")


def transcribe_recording(audio_path: Path, models: dict, is_diarized: bool) -> None:
    stem = clean_stem(audio_path.stem)
    out = {name: config.TRANSCRIPT_DIR / f"{stem}{suffix}" for name, suffix in (
        ("json", ".json"), ("chunks", "_silero_chunks.txt"), ("words", "_words.txt"),
        ("words_original", "_words_original.txt"), ("sentences", "_sentences.txt"))}
    print(f"\n{'=' * 60}\n[File] {audio_path.name}  (diarize={is_diarized})")
    if out["json"].exists():
        print(f"  [skip] JSON already exists: {out['json'].name}")
        return
    t_start = time.time()

    # 1. speech chunks (saved right away, in case a later step fails)
    print("  [1/3] Running Silero VAD...")
    audio, chunks = detect_speech_chunks(audio_path, models["vad"])
    write_audacity_labels(out["chunks"], [
        (c["start"], c["end"], f"chunk_{i:04d} ({c['end'] - c['start']:.2f}s)")
        for i, c in enumerate(chunks)])
    print(f"         [saved] {out['chunks'].name}")

    # 2. transcribe each chunk on its own
    print(f"  [2/3] Transcribing {len(chunks)} chunks independently...")
    sentences: list[dict] = []
    for i, c in enumerate(chunks):
        piece = audio[int(c["start"] * config.SILERO_SR):int(c["end"] * config.SILERO_SR)]
        piece = piece.numpy() if isinstance(piece, torch.Tensor) else np.array(piece)
        sentences += transcribe_chunk(piece, c["start"], models["whisper"],
                                      models["align"], models["align_meta"])
        if (i + 1) % 20 == 0 or i + 1 == len(chunks):
            print(f"         {i + 1}/{len(chunks)} chunks transcribed", flush=True)
    words_original = [w for s in sentences for w in s["words"]]

    # 3. speakers, on all words (trimming below keeps these same word objects)
    if not is_diarized:
        assign_word_speakers(words_original, turns=None, constant=config.DEFAULT_SPEAKER)
        speaker_source, diarized = config.DEFAULT_SPEAKER, False
    else:
        diarize_if_missing(audio_path, models["diarizer"])
        turns = load_speaker_turns(audio_path.stem, config.DIARIZATION_DIR)
        assign_word_speakers(words_original, turns=turns, constant=None)
        speaker_source = f"pyannote ({len({t['speaker'] for t in turns})} speakers)"
        diarized = True
    print(f"  [speaker] {speaker_source}")

    # 4. drop stray edge words; sentence times follow the words that remain
    words: list[dict] = []
    for s in sentences:
        s["words"] = trim_words_to_text(s["text"], s["words"])
        if s["words"]:
            s["start"], s["end"] = s["words"][0]["start"], s["words"][-1]["end"]
        set_segment_speaker(s)
        words += s["words"]

    out["json"].write_text(json.dumps({
        "audio_path": str(audio_path),
        "language": config.WHISPER_LANGUAGE,
        "model_name": config.WHISPER_MODEL,
        "device": config.WHISPER_DEVICE,
        "compute_type": config.WHISPER_COMPUTE_TYPE,
        "silero_threshold": config.SILERO_THRESHOLD,
        "n_silero_chunks": len(chunks),
        "needs_diarization": is_diarized,
        "diarized": diarized,
        "segments": sentences,   # [{start, end, text, words: [{start, end, word, speaker}], speaker, speaker_pure}]
    }, indent=2), encoding="utf-8")
    write_audacity_labels(out["words"], [
        (w["start"], w["end"], w["word"] + _speaker_label(w)) for w in words])
    write_audacity_labels(out["words_original"], [
        (w["start"], w["end"], w["word"] + _speaker_label(w)) for w in words_original])
    write_audacity_labels(out["sentences"], [
        (s["start"], s["end"], s["text"].replace("\t", " ").replace("\n", " ") + _speaker_label(s))
        for s in sentences])
    print(f"  [Saved] {out['json'].name}  (+ silero_chunks.txt, words.txt, "
          "words_original.txt, sentences.txt)")
    print(f"  [Done]  Elapsed: {format_hms(time.time() - t_start)}")


def main() -> None:
    import whisperx
    from silero_vad import load_silero_vad

    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rec in load_recordings():
        path = find_audio_file(config.DENOISED_DIR, config.DENOISED_PREFIX, rec.stem)
        if path.exists():
            jobs.append((path, rec.diarize))
        else:
            print(f"  [missing] no speech file for {rec.stem} in {config.DENOISED_DIR} (run step 01)")
    print(f"[Batch] {len(jobs)} recordings, {sum(d for _, d in jobs)} to diarize")
    print(f"[Batch] Output dir: {config.TRANSCRIPT_DIR}")
    if config.TRANSCRIBE_MAX_SECONDS is not None:
        print(f"[TEST] Duration cap: first {config.TRANSCRIBE_MAX_SECONDS}s only")

    print("\n[Loading] Silero VAD + WhisperX...")
    models = {"vad": load_silero_vad(),
              "whisper": whisperx.load_model(config.WHISPER_MODEL, config.WHISPER_DEVICE,
                                             compute_type=config.WHISPER_COMPUTE_TYPE,
                                             language=config.WHISPER_LANGUAGE),
              "diarizer": {}}
    models["align"], models["align_meta"] = whisperx.load_align_model(
        language_code=config.WHISPER_LANGUAGE, device=config.WHISPER_DEVICE)

    t_batch = time.time()
    for i, (path, diarize) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] Processing: {path.name}")
        try:
            transcribe_recording(path, models, diarize)
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")
    print(f"\n[Batch Done] {len(jobs)} files | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    main()
