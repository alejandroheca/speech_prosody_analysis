"""Speech and crowd-reaction features of recorded speeches.

Modules, in pipeline order:
    separation        step 1  speech / crowd separation (GTCRN denoiser)
    transcription     step 2  speech chunks (Silero VAD) and words (WhisperX)
    diarization       step 2  speaker turns (pyannote), multi-speaker recordings only
    speakers          step 2  speaker of each word / sentence
    speech_quality    step 3  speech quality of the recording (Squim)
    crowd_detection   step 4  crowd regions (CNN of crowd_reaction_analysis)
    segments          step 5  one row per chunk or sentence, with its voiced pieces
    segment_features  step 5  all features of one chunk or sentence
    features/         step 5  one acoustic feature per file
    config, metadata, audio, labels, utils: shared
"""
