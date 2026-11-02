#!/usr/bin/env bash
# Runs the three steps in order. Needs the two conda environments (envs/)
# and, if any recording needs diarization, HF_TOKEN in the environment.
set -euo pipefail
cd "$(dirname "$0")"

conda run --no-capture-output -n crowd_audio      python 01_separate.py
conda run --no-capture-output -n crowd_transcribe python 02_transcribe.py
conda run --no-capture-output -n crowd_audio      python 03_features.py
