#!/usr/bin/env bash
# The spaCy model (en_core_web_sm) is now pinned as a direct wheel in
# requirements.txt, which HF Spaces installs automatically — so no model
# download is needed here. This script is kept as a safe no-op so older deploy
# notes that reference it don't break.
set -euo pipefail
echo "en_core_web_sm installs via requirements.txt — nothing to pre-build."
