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


def cnn_gap_energy(y_crowd: np.ndarray, sr: int, gap_start: float, gap_end: float,
                   cnn_regions: list[tuple[float, float]]) -> dict:
    """crowd_gap of a gap, with crowd_energy / crowd_energy_rob set to 0 when
    no CNN crowd region overlaps the gap, or when the gap is too short to
    measure (shorter than MIN_SEGMENT_S, after the trimming for _rob)."""
    gap = crowd_gap(y_crowd, sr, gap_start, gap_end, measure_loudness=True)
    fires = (gap_start is not None and gap_end is not None and gap_end > gap_start
             and overlaps_any(gap_start, gap_end, cnn_regions))
    for key in ("crowd_energy", "crowd_energy_rob"):
        if not fires or np.isnan(gap[key]):
            gap[key] = 0.0
    return gap
