"""The gap after a segment (until the next segment starts) and the crowd's
loudness in it: the crowd's reaction to that segment."""
from __future__ import annotations

import numpy as np

from prosodic_features import config
from prosodic_features.features.opensmile_features import crowd_loudness, ratio_to_file


def crowd_gap(y_crowd: np.ndarray, sr: int, gap_start: float, gap_end: float,
              file_crowd_loudness: float, measure_loudness: bool) -> dict:
    """Gap duration and the crowd loudness in it, relative to the whole
    crowd track's loudness (ratio_to_file).

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
        out["crowd_energy"] = ratio_to_file(crowd_loudness(y_crowd[i_start:i_end], sr),
                                         file_crowd_loudness)
        out["crowd_energy_rob"] = ratio_to_file(crowd_loudness(y_crowd[i_rob_start:i_rob_end], sr),
                                             file_crowd_loudness)
    return out
