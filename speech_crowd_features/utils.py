"""Small shared utilities."""
from __future__ import annotations

from typing import Any, Optional


def safe_float(x: Any) -> Optional[float]:
    """float(x), or None if x is None or not convertible."""
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def format_hms(seconds: float) -> str:
    """Seconds as HH:MM:SS (for progress messages)."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
