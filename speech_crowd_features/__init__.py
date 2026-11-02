"""Speech and crowd-reaction features of recorded speeches.

Modules, in pipeline order:
    separation        step 1  speech / crowd separation (GTCRN denoiser)
    transcription     step 2  speech chunks (Silero VAD) and words (WhisperX)
    diarization       step 2  speaker turns (pyannote), multi-speaker recordings only
    speakers          step 2  speaker of each word / sentence
    segments          step 3  single-speaker segments from the chunks
    segment_features  step 3  all features of one segment
    features/         step 3  one acoustic feature per file
    config, metadata, audio, labels, utils: shared
"""
