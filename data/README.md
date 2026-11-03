Data folder (its contents are not part of the repository). Expected layout:

    recordings.xlsx        metadata file, one row per recording (see the main README)
    audio/raw/             the recordings (.wav or .flac)

Every other folder is created by the pipeline (see "Outputs" in the main README).

File names
----------
Name each recording with the speaker's identifier as a prefix, followed by an
underscore, e.g.

    shelton_1965_kkk_meeting_p3_interview.wav
    mlk_1963_i_have_a_dream.wav

The prefix (everything before the first "_") is used as the speaker's name in
the transcripts and in the features table:
  - single-speaker recordings (DIARIZATION_NEEDED = 0): every word gets it;
  - diarized recordings (DIARIZATION_NEEDED = 1): the target speaker,
    identified with the speaker marks below, gets it.
Use the same prefix for every recording of the same person.

Speaker marks (recordings.xlsx)
-------------------------------
In a diarized recording, pyannote labels the voices SPEAKER_00, SPEAKER_01,
... in no particular order. To know which one is the person you study, mark
one or two stretches where only they talk:

    FILENAME                                   ONSET_S1  OFFSET_S1  ONSET_S2  OFFSET_S2
    shelton_1965_kkk_meeting_p3_interview.wav  36        56         110       120

Times are in seconds from the start of the audio file listed in FILENAME (if
the file is a cut of a longer recording, use times in the cut). Step 02 sums,
for each diarization speaker, how long they talk inside the marks; the one
that talks the most is renamed to the file name's prefix ("shelton") in the
transcript JSON, the _words.txt / _sentences.txt label tracks, and the
features table. The other voices keep their SPEAKER_xx labels. The
diarization files themselves are left unchanged.

Choosing the marks:
  - Take stretches where only the target speaker talks (no interviewer, no
    overlapping voices). Several seconds each is enough.
  - Use two stretches far apart (e.g. one near the start, one near the end),
    so a speaker that pyannote splits or merges differently over time is
    still caught.
  - The transcript JSON records the result under "target_speaker": the
    diarization speaker chosen and how many seconds each speaker talks inside
    the marks. If the chosen one talks for much less than the marked time,
    the marks or the diarization need checking.

The marks are optional. Without them, the voices of a diarized recording keep
their SPEAKER_xx labels. Single-speaker recordings need no marks.
