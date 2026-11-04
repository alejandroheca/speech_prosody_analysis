"""The summary table of processed recordings (config.RECORDINGS_DONE_CSV,
next to the metadata file): one row per recording, keyed by "file".
Step 03 writes speech_quality; step 05 adds speaker, duration and diarized."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prosodic_features import config

COLUMNS = ["file", "speaker", "duration", "diarized", "speech_quality"]


def read() -> pd.DataFrame:
    path = config.RECORDINGS_DONE_CSV
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def get(file: str, column: str):
    """The value of column for file, or None if the file has no row / value."""
    df = read()
    values = df.loc[df["file"] == file, column]
    return None if values.empty or pd.isna(values.iloc[0]) else values.iloc[0]


def update(file: str, **fields) -> None:
    """Set fields in file's row (created if missing), keeping its other values."""
    df = read()
    if not (df["file"] == file).any():
        df = pd.concat([df, pd.DataFrame([{"file": file}])], ignore_index=True)
    for col, value in fields.items():
        df.loc[df["file"] == file, col] = value
    df = df[COLUMNS + [c for c in df.columns if c not in COLUMNS]].sort_values("file")
    df["diarized"] = pd.to_numeric(df["diarized"]).astype("Int64")
    config.RECORDINGS_DONE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RECORDINGS_DONE_CSV, index=False)
