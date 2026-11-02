"""Reading the recordings' metadata file (config.AUDIO_METADATA, .xlsx or .csv).

One row per recording. Columns used (names set in config.py):
  METADATA_FILENAME_COLUMN  the recording's audio file name, e.g. "mlk_1963_dream.wav"
  METADATA_DIARIZE_COLUMN   1 / 0: the recording has several speakers
                            (interview, debate, ...) and is diarized in step 02
  METADATA_CROWD_COLUMN     1 / 0: an audience reacts to the speaker, so the
                            crowd's loudness in the gaps is measured in step 03
                            (0: the gaps are other speakers or silence; the
                            crowd columns are left empty)
1 is any of 1, yes, y, true (any case); anything else, including an empty
cell, is 0. The file is only read, never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from speech_crowd_features import config


@dataclass
class Recording:
    stem: str          # file name without extension: the name of every output
    diarize: bool      # run the diarization pathway in step 02
    crowd: bool        # compute crowd features in step 03
    row: dict          # the whole metadata row (for METADATA_COPY_COLUMNS)


def clean_stem(stem: str) -> str:
    """Recording name without the denoiser's 'denoised_' prefix."""
    prefix = config.DENOISED_PREFIX
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def is_yes(value) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1", "1.0"}


def load_recordings(path: Path = config.AUDIO_METADATA) -> list[Recording]:
    """The recordings listed in the metadata file, in its row order. Rows
    with an empty file name are skipped; a missing column or a file name
    listed twice is an error."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path} (set AUDIO_METADATA "
                                "in speech_crowd_features/config.py)")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df = df.fillna("")

    for col in (config.METADATA_FILENAME_COLUMN, config.METADATA_DIARIZE_COLUMN,
                config.METADATA_CROWD_COLUMN):
        if col not in df.columns:
            raise ValueError(f"{path.name} has no column '{col}'")

    recordings, seen = [], set()
    for row in df.to_dict("records"):
        name = row[config.METADATA_FILENAME_COLUMN].strip()
        if not name:
            continue
        stem = Path(name).stem
        if stem in seen:
            raise ValueError(f"{path.name}: '{name}' is listed twice")
        seen.add(stem)
        diarize = is_yes(row[config.METADATA_DIARIZE_COLUMN])
        recordings.append(Recording(stem=stem, diarize=diarize,
                                    crowd=is_yes(row[config.METADATA_CROWD_COLUMN]), row=row))
    print(f"[meta] {len(recordings)} recordings in {path.name} "
          f"({sum(r.diarize for r in recordings)} to diarize, "
          f"{sum(r.crowd for r in recordings)} with crowd features)")
    return recordings
