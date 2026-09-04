#!/usr/bin/env bash
# Anonymiser Studio — the pipeline as an app, running the real fine-tune.
#
# Uses legal-anon-env, which already has torch, transformers, fastapi and
# uvicorn. It deliberately does NOT use demo/venv-gradio: that venv has no
# torch, so the Gradio demo has never been able to load the fine-tuned model
# and silently falls back to spaCy.
#
#   ./demo/run_studio.sh                  # CPU (safe while a training run holds the GPU)
#   ANON_DEVICE=mps ./demo/run_studio.sh  # once the GPU is free
#   ANON_BACKEND=legalbert ./demo/run_studio.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=legal-anon-env/bin/python
[ -x "$PY" ] || { echo "legal-anon-env not found — see CLAUDE.md for setup."; exit 1; }

exec "$PY" demo/studio/server.py
