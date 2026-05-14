#!/usr/bin/env bash
# HuggingFace Spaces runs this once per image build. Downloads the spaCy
# model into the Space's persistent venv so `app.py`'s `spacy.load()` works.
set -euo pipefail
python -m spacy download en_core_web_sm
