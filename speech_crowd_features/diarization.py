"""Speaker turns with pyannote (used by step 02 for recordings flagged
needs_diarization). Needs a Hugging Face token (HF_TOKEN) with the terms of
the pyannote model accepted."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from speech_crowd_features import config


def load_diarization_pipeline(hf_token: str):
    """Load the pyannote pipeline (config.DIARIZATION_MODEL_ID) onto
    config.DIARIZATION_DEVICE. Returns (pipeline, torch_device)."""
    from pyannote.audio import Pipeline   # heavy import, only when needed

    print("[Init] Loading pipeline:", config.DIARIZATION_MODEL_ID)
    pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL_ID, token=hf_token)
    torch_device = torch.device(config.DIARIZATION_DEVICE)
    print("[Init] Moving pipeline to:", torch_device)
    pipeline.to(torch_device)
    return pipeline, torch_device


def merge_speaker_turns(turns: list[dict], max_gap: float,
                        min_duration: float) -> list[dict]:
    """Merge consecutive turns of the same speaker that are at most max_gap
    seconds apart, then drop turns shorter than min_duration seconds.
    turns: [{"start", "end", "speaker", ...}, ...]"""
    print(f"[Merge] Input segments: {len(turns)}")
    if not turns:
        return []

    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
    merged = [dict(turns[0])]
    for turn in turns[1:]:
        prev = merged[-1]
        if turn["speaker"] == prev["speaker"] and turn["start"] - prev["end"] <= max_gap:
            prev["end"] = max(prev["end"], turn["end"])
        else:
            merged.append(dict(turn))

    kept = [t for t in merged if (t["end"] - t["start"]) >= min_duration]
    print(f"[Merge] After merge: {len(merged)}")
    print(f"[Merge] After filtering short segments: {len(kept)}")
    return kept


def _annotation_of(output: Any):
    """The pyannote Annotation inside a pipeline output: older pyannote
    versions return it directly, newer ones wrap it (DiarizeOutput)."""
    if hasattr(output, "itertracks"):
        return output
    for attr in ("speaker_diarization", "annotation", "diarization"):
        inner = getattr(output, attr, None)
        if hasattr(inner, "itertracks"):
            return inner
    if isinstance(output, dict):
        for key in ("speaker_diarization", "annotation", "diarization"):
            if key in output and hasattr(output[key], "itertracks"):
                return output[key]
    raise TypeError(f"Unsupported diarization output type: {type(output)}")


def diarize_file(pipeline, torch_device, audio_path: Path,
                 out_json: Path, out_rttm: Path):
    """Speaker turns of one recording, merged with merge_speaker_turns
    (DIARIZATION_MAX_GAP_S, DIARIZATION_MIN_DURATION_S) and saved as
    out_json ({"audio_path", "segments": [{start, end, speaker,
    confidence}]}) plus the unmerged pyannote turns as out_rttm.

    If the pipeline fails on Apple's MPS GPU, it is moved to the CPU and
    the file rerun. Returns (turns, torch_device), as the device may change."""
    print(f"\n[Diarize] Processing: {audio_path}")
    # audio is read with soundfile instead of pyannote's own decoder
    audio, sr = sf.read(str(audio_path), always_2d=True)       # (time, channels)
    print(f"[Audio] sr={sr}, shape={audio.shape}")
    waveform = torch.from_numpy(audio.mean(axis=1)).float().unsqueeze(0)  # (1, time)
    try:
        waveform = waveform.to(torch_device)
    except Exception:
        pass
    file_dict = {"waveform": waveform, "sample_rate": int(sr)}

    print(f"[Diarize] Running pyannote pipeline (device={torch_device})")
    t0 = time.time()
    try:
        output = pipeline(file_dict)
    except RuntimeError as e:
        msg = str(e).lower()
        if "mps" in msg or "metal" in msg or "not implemented" in msg:
            print("[Warn] MPS failed; falling back to CPU for this file.")
            torch_device = torch.device("cpu")
            pipeline.to(torch_device)
            file_dict["waveform"] = file_dict["waveform"].to(torch_device)
            output = pipeline(file_dict)
        else:
            raise
    print(f"[Diarize] Completed in {time.time() - t0:.2f}s")

    annotation = _annotation_of(output)
    turns = [{"start": float(turn.start), "end": float(turn.end),
              "speaker": str(speaker), "confidence": None}
             for turn, _, speaker in annotation.itertracks(yield_label=True)]
    print(f"[Diarize] Raw segments: {len(turns)}")
    turns = merge_speaker_turns(turns, max_gap=config.DIARIZATION_MAX_GAP_S,
                                min_duration=config.DIARIZATION_MIN_DURATION_S)

    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    with out_rttm.open("w", encoding="utf-8") as f:
        annotation.write_rttm(f)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"audio_path": str(audio_path), "segments": turns},
                                   indent=2), encoding="utf-8")
    print(f"[Output] {out_json.name} + {out_rttm.name} ({len(turns)} turns)")
    return turns, torch_device
