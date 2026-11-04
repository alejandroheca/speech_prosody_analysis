"""Step 04: crowd detection with the CNN-TCN of crowd_reaction_analysis.

The model's event head gives, every 40 ms, the probability that the crowd
reacts (whatever the polarity). Its scripts/api.py thresholds and smooths it
into regions; here only regions whose mean probability is above
CNN_MIN_MEAN_SCORE are kept."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from prosodic_features import config

_API = None


def load_cnn_api():
    """scripts/api.py of the crowd_reaction_analysis clone in CNN_REPO_DIR."""
    global _API
    if _API is None:
        api_path = config.CNN_REPO_DIR / "scripts" / "api.py"
        if not api_path.exists():
            raise FileNotFoundError(
                f"crowd_reaction_analysis not found at {config.CNN_REPO_DIR}. Clone "
                "github.com/chymaera96/crowd_reaction_analysis next to this repository, "
                "or set CROWD_CNN_REPO to its path")
        spec = importlib.util.spec_from_file_location("crowd_cnn_api", api_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module      # needed by its dataclasses
        spec.loader.exec_module(module)
        _API = module
    return _API


def detect_crowd(audio_path: Path) -> list[tuple[float, float, float]]:
    """[(start, end, mean probability)] of the crowd regions of a recording."""
    api = load_cnn_api()
    checkpoint = config.CNN_REPO_DIR / config.CNN_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} not found: run python scripts/download_weights.py "
                                f"--model cnn_tcn in {config.CNN_REPO_DIR}")
    result = api.run_audio_inference(
        audio_path=audio_path,
        config_path=config.CNN_REPO_DIR / config.CNN_CONFIG,
        checkpoint_path=checkpoint,
        median_filter_sec=config.CNN_MEDIAN_FILTER_S,
        batch_size=config.CNN_BATCH_SIZE,
        device=config.CNN_DEVICE,
        mode="event",
        show_progress=False,
    )
    regions = []
    for onset, offset, label in result.predicted_regions:
        mean = api.region_mean_score(result, onset, offset, label)
        if mean > config.CNN_MIN_MEAN_SCORE:
            regions.append((float(onset), float(offset), mean))
    return regions
