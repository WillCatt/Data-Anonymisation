#!/usr/bin/env bash
# Bootstrap + launch the Gradio demo in its own isolated venv.
#
# First run creates demo/venv-gradio/ and installs the pinned dependencies
# from demo/requirements-gradio.txt (which includes the en_core_web_sm wheel).
# Subsequent runs just activate the venv and launch.
#
# Why a separate venv?  The repo's main venv has spaCy 3.7 + the Phase 2
# transformers stack. Gradio 4.x has dep-tree conflicts with those (typer,
# huggingface_hub).  Isolating the demo here means we can pin
# Gradio-friendly versions without touching the main venv.

set -euo pipefail

# Resolve paths relative to the script regardless of cwd
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV="$HERE/venv-gradio"
REQS="$HERE/requirements-gradio.txt"
APP="$HERE/app.py"

# Find a Python 3.11 binary. Prefer pyenv shim, fall back to system.
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON not found. Set PYTHON=/path/to/python3.11 or install via pyenv."
    exit 1
fi

# First-run bootstrap
if [[ ! -d "$VENV" ]]; then
    echo "── First run: creating $VENV ──"
    "$PYTHON" -m venv "$VENV"

    # Hard-set the CA bundle for pip — sidesteps the cacert.pem corner
    # we hit with the main venv during pip self-upgrade on macOS.
    export SSL_CERT_FILE="$(${PYTHON} -c 'import ssl; print(ssl.get_default_verify_paths().cafile or "")')"

    "$VENV/bin/python" -m pip install --upgrade pip
    # requirements-gradio.txt pins the en_core_web_sm wheel directly, so the
    # model installs here too — no separate (and flaky) `spacy download` step.
    "$VENV/bin/python" -m pip install -r "$REQS"

    echo "── Bootstrap complete ──"
fi

# Make the repo's src/ importable. demo/app.py adds src/ to sys.path on its
# own, but this lets `python -m anonymisation.cli ...` work too if you want.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Optional override — if you've installed en_core_web_trf and prefer that.
export SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"

echo "── Launching Gradio demo (Ctrl-C to stop) ──"
echo "   spaCy model: $SPACY_MODEL"
echo "   venv:        $VENV"
echo
exec "$VENV/bin/python" "$APP"
