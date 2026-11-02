"""Speech rate of a segment."""
from __future__ import annotations

import numpy as np


def speech_rate(words: list[dict], duration: float) -> dict:
    """Number of (non-empty) words and words per second of the segment."""
    n_words = len([w for w in words if w.get("word", "").strip()])
    return {"n_words": n_words,
            "word_rate": round(n_words / duration, 4) if duration > 0 else np.nan}
