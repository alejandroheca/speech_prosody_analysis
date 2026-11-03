"""
Step 02: speech chunks, words and speakers of every recording.

    conda activate crowd_transcribe
    export HF_TOKEN=hf_...      # only needed if a recording needs diarization
    python 02_transcribe.py

For each recording in config.AUDIO_METADATA (its speech file from step 01):
  1. Silero VAD finds the speech chunks.
  2. WhisperX transcribes and word-aligns each chunk on its own.
  3. Speakers. The target speaker's name is the file name's prefix (up to
     the first underscore, e.g. "shelton"). Single-speaker recordings get
     it for every word. Recordings with DIARIZATION_NEEDED = 1 in the
     metadata are diarized with pyannote (once; the result is saved and
     reused) and each word gets the speaker whose turn overlaps it the
     most; if the metadata marks stretches where only the target speaker
     talks (ONSET_S1/OFFSET_S1, ONSET_S2/OFFSET_S2), the diarization
     speaker that talks most within them gets the target's name, and the
     others keep their SPEAKER_xx labels.
  4. Edge words of each WhisperX sentence (handle_edge_words): stray words
     that are not in its text are dropped, and a first / last word that
     crosses its Silero chunk is moved inside it. Sentence times follow
     their words, except that the first sentence of each chunk starts at
     the chunk's start and the last one ends at the chunk's end.

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

from prosodic_features import config
from prosodic_features.labels import write_audacity_labels
from prosodic_features.audio import find_audio_file
from prosodic_features.metadata import Recording, clean_stem, load_recordings
from prosodic_features.speakers import (assign_word_speakers, identify_target_speaker,
                                 load_speaker_turns, rename_speaker, set_segment_speaker)
from prosodic_features.transcription import (detect_speech_chunks, handle_edge_words,
                                      transcribe_chunk)
from prosodic_features.utils import format_hms


def _speaker_label(item: dict) -> str:
    return f" [{item['speaker']}]" if item.get("speaker") else ""


def diarize_if_missing(audio_path: Path, diarizer: dict) -> None:
    """Run pyannote on a recording unless its diarization JSON exists.
    The pipeline is loaded on first use and kept in diarizer."""
    if load_speaker_turns(audio_path.stem, config.DIARIZATION_DIR) is not None:
        return
    from prosodic_features.diarization import diarize_file, load_diarization_pipeline

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


def transcribe_recording(audio_path: Path, models: dict, rec: Recording) -> None:
    is_diarized = rec.diarize
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
        for s in transcribe_chunk(piece, c["start"], models["whisper"],
                                  models["align"], models["align_meta"]):
            s["chunk"] = i      # index of the sentence's Silero chunk
            sentences.append(s)
        if (i + 1) % 20 == 0 or i + 1 == len(chunks):
            print(f"         {i + 1}/{len(chunks)} chunks transcribed", flush=True)
    words_original = [w for s in sentences for w in s["words"]]

    # 3. speakers, on all words (step 4 keeps these word objects, or copies them)
    target = None       # target speaker identified from the metadata marks
    if not is_diarized:
        assign_word_speakers(words_original, turns=None, constant=rec.speaker)
        speaker_source, diarized = rec.speaker, False
    else:
        diarize_if_missing(audio_path, models["diarizer"])
        turns = load_speaker_turns(audio_path.stem, config.DIARIZATION_DIR)
        speaker_source = f"pyannote ({len({t['speaker'] for t in turns})} speakers)"
        if rec.marks:
            label, talk = identify_target_speaker(turns, rec.marks)
            marked = sum(off - on for on, off in rec.marks)
            target = {"name": rec.speaker, "diarization_speaker": label,
                      "marks": rec.marks, "marked_s": round(marked, 3),
                      "talk_in_marks_s": {k: round(v, 3) for k, v in talk.items()}}
            if label is None:
                print(f"  [speaker] WARNING: no diarization speaker talks in the marks of {rec.speaker}")
            else:
                rename_speaker(turns, label, rec.speaker)
                speaker_source += (f"; {label} = {rec.speaker} (talks {talk[label]:.1f} s "
                                   f"of the {marked:.1f} s marked)")
        assign_word_speakers(words_original, turns=turns, constant=None)
        diarized = True
    print(f"  [speaker] {speaker_source}")

    # 4. edge words: drop stray ones, move chunk-crossing ones inside the chunk;
    #    sentence times follow the words that remain
    words: list[dict] = []
    for s in sentences:
        c = chunks[s["chunk"]]
        s["words"] = handle_edge_words(s["text"], s["words"], c["start"], c["end"])
        if s["words"]:
            s["start"], s["end"] = s["words"][0]["start"], s["words"][-1]["end"]
        set_segment_speaker(s)
        words += s["words"]
    # the first sentence of each chunk starts at the chunk's start, the last
    # one ends at the chunk's end (word times are not changed)
    for i, c in enumerate(chunks):
        in_chunk = [s for s in sentences if s["chunk"] == i and s["words"]]
        if in_chunk:
            in_chunk[0]["start"], in_chunk[-1]["end"] = c["start"], c["end"]

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
        "target_speaker": target,
        "segments": sentences,   # [{start, end, text, words: [{start, end, word, speaker}], chunk, speaker, speaker_pure}]
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
            jobs.append((path, rec))
        else:
            print(f"  [missing] no speech file for {rec.stem} in {config.DENOISED_DIR} (run step 01)")
    print(f"[Batch] {len(jobs)} recordings, {sum(r.diarize for _, r in jobs)} to diarize")
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
    for i, (path, rec) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] Processing: {path.name}")
        try:
            transcribe_recording(path, models, rec)
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")
    print(f"\n[Batch Done] {len(jobs)} files | Total: {format_hms(time.time() - t_batch)}")


if __name__ == "__main__":
    main()
