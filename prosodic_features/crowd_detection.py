"""Step 04: crowd detection with the CNN-TCN of crowd_reaction_analysis.

The model's event head gives, every 40 ms, the probability that the crowd
reacts (whatever the polarity). The package thresholds and smooths it
into regions; here only regions whose mean probability is above
CNN_MIN_MEAN_SCORE are kept."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from prosodic_features import config


@lru_cache(maxsize=1)
def _detector():
    from crowd_reaction import CrowdDetector

    return CrowdDetector(model="cnn_tcn", device=config.CNN_DEVICE)


def detect_crowd(audio_path: Path) -> list[tuple[float, float, float]]:
    """[(start, end, mean probability)] of the crowd regions of a recording."""
    result = _detector().predict(
        audio_path=audio_path,
        median_filter_sec=config.CNN_MEDIAN_FILTER_S,
        batch_size=config.CNN_BATCH_SIZE,
        min_mean_score=config.CNN_MIN_MEAN_SCORE,
        mode="event",
        show_progress=False,
    )
    return [
        (float(onset), float(offset), result.region_mean_score(onset, offset, label))
        for onset, offset, label in result.predicted_regions
    ]
