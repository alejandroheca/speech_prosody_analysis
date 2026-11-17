"""Gaps around a sentence (between its chunks, and after it until the next
sentence starts) and the crowd's loudness in them: the crowd's reaction."""
from __future__ import annotations

import numpy as np

from prosodic_features import config
from prosodic_features.features.opensmile_features import crowd_loudness


def crowd_gap(y_crowd: np.ndarray, sr: int, gap_start: float, gap_end: float,
              measure_loudness: bool) -> dict:
    """Gap duration and the crowd loudness in it (mean eGeMAPS loudness of
    the gain-normalized crowd signal, see AMPLITUDE_REF_PERCENTILE).

    The *_rob versions leave out REVERB_LAG_S at both ends of the gap, so
    the speech's own reverberation and the next segment's onset do not
    count as crowd. Loudness is NaN unless measure_loudness (False for
    interviews, whose gaps are another speaker's turn, not a crowd)."""
    nan = {"gap_duration": np.nan, "gap_duration_rob": np.nan, "reaction_end": np.nan,
           "crowd_energy": np.nan, "crowd_energy_rob": np.nan}
    if gap_start is None or gap_end is None or gap_end <= gap_start:
        return nan

    lag = int(config.REVERB_LAG_S * sr)
    i_start = max(0, int(gap_start * sr))
    i_end = min(len(y_crowd), int(gap_end * sr))
    i_rob_start = min(i_start + lag, i_end)
    i_rob_end = max(i_rob_start, i_end - lag)

    out = {"gap_duration": gap_end - gap_start,
           "gap_duration_rob": (i_rob_end - i_rob_start) / sr,
           "reaction_end": gap_end,
           "crowd_energy": np.nan, "crowd_energy_rob": np.nan}
    if measure_loudness:
        out["crowd_energy"] = crowd_loudness(y_crowd[i_start:i_end], sr)
        out["crowd_energy_rob"] = crowd_loudness(y_crowd[i_rob_start:i_rob_end], sr)
    return out


def overlaps_any(start: float, end: float, regions: list[tuple[float, float]]) -> bool:
    """Whether [start, end] overlaps any of the (start, end) regions."""
    return any(rs < end and re > start for rs, re in regions)


def crowd_energy_floor(y_crowd: np.ndarray, sr: int,
                       speech_chunks: list[tuple[float, float]],
                       cnn_regions: list[tuple[float, float]]) -> float:
    """The recording's static: CROWD_FLOOR_PERCENTILE of the crowd loudness
    in the pauses without a crowd. Pauses are the stretches between Silero
    speech chunks (any speaker) that no CNN crowd region overlaps; each
    loses REVERB_LAG_S at both ends (as the *_rob measures), and pauses then
    shorter than MIN_SEGMENT_S are left out. If no such pause is left (a
    recording with crowd in every pause), the parts of the pauses outside
    the CNN regions are used instead. NaN if still nothing is left."""
    lag = config.REVERB_LAG_S
    chunks = sorted(speech_chunks)
    pauses = [(gs, ge) for (_, gs), (ge, _) in zip(chunks[:-1], chunks[1:]) if ge > gs]

    def loudness(spans):
        values = []
        for s0, s1 in spans:
            i0, i1 = max(0, int((s0 + lag) * sr)), min(len(y_crowd), int((s1 - lag) * sr))
            if i1 - i0 >= int(config.MIN_SEGMENT_S * sr):
                v = crowd_loudness(y_crowd[i0:i1], sr)
                if not np.isnan(v):
                    values.append(v)
        return values

    values = loudness([p for p in pauses if not overlaps_any(*p, cnn_regions)])
    if not values:
        values = loudness([piece for gs, ge in pauses for piece in subtract_spans(gs, ge, cnn_regions)])
    return float(np.percentile(values, config.CROWD_FLOOR_PERCENTILE)) if values else np.nan


def above_floor(value: float, floor: float) -> float:
    """value - floor, not below 0 (a crowd no louder than the static is 0);
    value as it is if floor is NaN."""
    return value if np.isnan(floor) else max(0.0, value - floor)


def cnn_gap_energy(y_crowd: np.ndarray, sr: int, gap_start: float, gap_end: float,
                   cnn_regions: list[tuple[float, float]], floor: float = np.nan) -> dict:
    """crowd_gap of a gap, with crowd_energy / crowd_energy_rob set to 0 when
    no CNN crowd region overlaps the gap, or when the gap is too short to
    measure (shorter than MIN_SEGMENT_S, after the trimming for _rob);
    otherwise minus the recording's floor (crowd_energy_floor), not below 0."""
    gap = crowd_gap(y_crowd, sr, gap_start, gap_end, measure_loudness=True)
    fires = (gap_start is not None and gap_end is not None and gap_end > gap_start
             and overlaps_any(gap_start, gap_end, cnn_regions))
    for key in ("crowd_energy", "crowd_energy_rob"):
        if not fires or np.isnan(gap[key]):
            gap[key] = 0.0
        else:
            gap[key] = above_floor(gap[key], floor)
    return gap


def subtract_spans(start: float, end: float,
                   spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of [start, end] not covered by any of the (start, end) spans."""
    out, cur = [], start
    for s, e in sorted(spans):
        if e <= cur or s >= end:
            continue
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out


def lagged_reaction_energy(y_crowd: np.ndarray, sr: int, onset_from: float, offset: float,
                           cnn_regions: list[tuple[float, float]],
                           speech_chunks: list[tuple[float, float]],
                           floor: float = np.nan) -> float:
    """Loudness of the crowd reaction after a row's end (offset):
    crowd_energy_end_lag_rob, with onset_from = -inf (any onset up to the
    window's end, also one before the row started).

      1. Reaction: the first CNN crowd region whose onset is in
         [onset_from, offset + LAG_REACTION_WINDOW_S] and that is still going
         on at offset; none -> 0. A region that started before onset_from
         does not count, nor one that ends before offset.
      2. Measured span: from the later of its onset and offset to its end,
         so only the part after the row counts (the row's own pauses are
         crowd_energy_around's inner gaps).
      3. Speech removed: every Silero chunk (speech, any speaker) inside the
         span is cut out, leaving its speech-free pieces. (Speech between
         offset and a later onset is never measured.)
      4. Each piece loses REVERB_LAG_S at both ends (as the other *_rob
         measures), and pieces then shorter than MIN_SEGMENT_S are dropped.
      5. Energy: mean eGeMAPS loudness of the crowd signal in each piece,
         averaged over the pieces weighted by their duration, minus the
         recording's floor (crowd_energy_floor), not below 0; 0 if no piece
         is left (the reaction is all speech)."""
    window_end = offset + config.LAG_REACTION_WINDOW_S
    onsets = [(rs, re) for rs, re in cnn_regions
              if onset_from <= rs <= window_end and re > offset]
    if not onsets:
        return 0.0
    r_start, r_end = min(onsets)
    lag = config.REVERB_LAG_S
    values, weights = [], []
    for ps, pe in subtract_spans(max(r_start, offset), r_end, speech_chunks):
        ps, pe = ps + lag, pe - lag
        i0, i1 = max(0, int(ps * sr)), min(len(y_crowd), int(pe * sr))
        if i1 - i0 < int(config.MIN_SEGMENT_S * sr):
            continue
        v = crowd_loudness(y_crowd[i0:i1], sr)
        if not np.isnan(v):
            values.append(v)
            weights.append(i1 - i0)
    if not values:
        return 0.0
    return above_floor(float(np.average(values, weights=weights)), floor)
