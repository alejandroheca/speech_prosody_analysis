"""Step 05: all features of one chunk or sentence (run in parallel worker processes).

The recording's speech and crowd signals are put in shared memory once by
the step script, so the workers read them without each keeping a copy."""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from prosodic_features import config
from prosodic_features.features.crowd_gap import cnn_gap_energy, crowd_gap
from prosodic_features.features.opensmile_features import egemaps
from prosodic_features.features.speech_rate import speech_rate


def _round_or_nan(x, digits: int):
    return round(x, digits) if not np.isnan(x) else np.nan


def compute_segment_features(task: dict) -> Optional[dict]:
    """One output row for task["segment"], a chunk or a sentence
    (task["unit"]; see segments.load_chunks / load_sentences).

    Speech features (eGeMAPS, duration, speech rate) use only the row's
    voiced pieces (a chunk: itself; a sentence: its Silero chunks, cut to
    the sentence's edges), joined. Crowd features use the gaps: the end
    gap, from the row's end to the next row's start (task["next_start"]),
    and, for sentences, the inner gaps, between the sentence's pieces.
      crowd_energy_end     crowd loudness in the end gap
      crowd_energy_around  sentences only: sum of the crowd loudness in the
                           inner gaps and the end gap
    A gap counts only if a CNN crowd region (task["cnn_regions"]) overlaps
    it; otherwise, or if it is too short to measure, it counts as 0. Crowd
    loudness is NaN for recordings without an active crowd
    (task["is_speech"] False) and for multi_speaker sentences.

    The speech and crowd signals are already gain-normalized (divided by the
    recording's AMPLITUDE_REF_PERCENTILE amplitude); task carries their
    shared-memory names / shapes / dtypes. Returns None for a row without
    valid times."""
    seg = task["segment"]
    shm_s = shared_memory.SharedMemory(name=task["speech_shm"])
    shm_c = shared_memory.SharedMemory(name=task["crowd_shm"])
    y_speech = np.ndarray(task["shape"], dtype=task["dtype"], buffer=shm_s.buf)
    y_crowd = np.ndarray(task["shape"], dtype=task["dtype"], buffer=shm_c.buf)
    sr = config.FEATURES_SR
    try:
        start, end, pieces = seg["start"], seg["end"], seg["pieces"]
        if end <= start:
            return None
        multi_speaker = bool(seg.get("multi_speaker", False))
        words = seg.get("words", [])

        y_voiced = np.concatenate([y_speech[max(0, int(ps * sr)):min(len(y_speech), int(pe * sr))]
                                   for ps, pe in pieces])
        duration = round(sum(pe - ps for ps, pe in pieces), 4)
        speech = egemaps(y_voiced, sr)

        end_gap = crowd_gap(y_crowd, sr, end, task["next_start"], measure_loudness=False)
        sentence = task["unit"] == "sentence"
        keys = ["crowd_energy_end", "crowd_energy_end_rob"]
        if sentence:
            keys += ["crowd_energy_around", "crowd_energy_around_rob"]
        crowd = {k: np.nan for k in keys}
        if task["is_speech"] and not multi_speaker:
            gaps = [(pieces[i][1], pieces[i + 1][0]) for i in range(len(pieces) - 1)]
            gaps.append((end, task["next_start"]))
            energies = [cnn_gap_energy(y_crowd, sr, gs, ge, task["cnn_regions"])
                        for gs, ge in gaps]
            crowd["crowd_energy_end"] = energies[-1]["crowd_energy"]
            crowd["crowd_energy_end_rob"] = energies[-1]["crowd_energy_rob"]
            if sentence:
                crowd["crowd_energy_around"] = sum(e["crowd_energy"] for e in energies)
                crowd["crowd_energy_around_rob"] = sum(e["crowd_energy_rob"] for e in energies)

        row = {
            "seg_idx": task["seg_idx"],
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": seg.get("speaker"),
            "multi_speaker": multi_speaker,
            "reaction_end": _round_or_nan(end_gap["reaction_end"], 3),
            "duration": duration,
            **({"n_chunks": len(pieces)} if sentence else {}),
            "text": seg.get("text", "").strip(),
            **speech_rate(words, duration),
            **({} if sentence else {"sentence_end": seg.get("sentence_end", 0)}),
            "gap_duration": _round_or_nan(end_gap["gap_duration"], 3),
            "gap_duration_rob": _round_or_nan(end_gap["gap_duration_rob"], 3),
            **{k: _round_or_nan(v, 6) for k, v in crowd.items()},
        }
        row.update(speech)
        return row
    finally:
        shm_s.close()
        shm_c.close()
