"""openSMILE features: the eGeMAPS v02 functionals of the speech, and the
crowd's loudness (eGeMAPS mean loudness, loudness_sma3_amean: openSMILE's
auditory-model loudness, the closest eGeMAPS has to an energy measure).
Both use the same openSMILE session."""

from __future__ import annotations

import numpy as np
import opensmile

from prosodic_features import config

_SMILE = None       # one openSMILE session per process
_SMILE_LLD = None   # frame-level (LLD) session, only used in the main process


def get_smile() -> "opensmile.Smile":
    """The process's openSMILE session (eGeMAPS v02, functionals)."""
    global _SMILE
    if _SMILE is None:
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                 feature_level=opensmile.FeatureLevel.Functionals)
    return _SMILE


def loudness_percentile(y: np.ndarray, sr: int, q: float) -> float:
    """The q-th percentile of the frame-level eGeMAPS loudness of y (10 ms
    frames); NaN if openSMILE fails."""
    global _SMILE_LLD
    if _SMILE_LLD is None:
        _SMILE_LLD = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    try:
        frames = _SMILE_LLD.process_signal(y.astype(np.float32), sr)[config.LOUDNESS_LLD_KEY]
        return float(np.percentile(frames.to_numpy(), q))
    except Exception as e:
        print(f"         [loudness_percentile] error: {e}")
        return np.nan


def egemaps(y: np.ndarray, sr: int) -> dict:
    """The 88 eGeMAPS functionals, named with config.SPEECH_PREFIX.
    All NaN if the signal is shorter than MIN_SEGMENT_S or openSMILE fails."""
    smile = get_smile()
    nan_row = {f"{config.SPEECH_PREFIX}{n}": np.nan for n in smile.feature_names}
    if len(y) < int(config.MIN_SEGMENT_S * sr):
        return nan_row
    try:
        values = smile.process_signal(y.astype(np.float32), sr).iloc[0].to_dict()
        return {f"{config.SPEECH_PREFIX}{k}": float(v) for k, v in values.items()}
    except Exception as e:
        print(f"         [egemaps] error: {e}")
        return nan_row


def crowd_loudness(y: np.ndarray, sr: int) -> float:
    """Mean eGeMAPS loudness of y; NaN if shorter than MIN_SEGMENT_S or if
    openSMILE fails."""
    if len(y) < int(config.MIN_SEGMENT_S * sr):
        return np.nan
    try:
        df = get_smile().process_signal(y.astype(np.float32), sr)
        return float(df.iloc[0][config.CROWD_LOUDNESS_KEY])
    except Exception as e:
        print(f"         [crowd_energy] error: {e}")
        return np.nan


def ratio_to_file(value, file_value) -> float:
    """value divided by the same measure on the whole recording, so loudness
    is comparable across recordings with different gain, microphones and
    noise floors. NaN if either is missing or the file value is 0."""
    if value is None or file_value is None or np.isnan(value) or not file_value or np.isnan(file_value):
        return np.nan
    return value / file_value
