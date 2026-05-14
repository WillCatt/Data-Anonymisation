#!/usr/bin/env bash
# Refresh spaces/anonymisation/ and spaces/examples/ from the main repo source.
# Run this whenever you've changed code in src/ or examples in demo/examples/
# and want to push the changes to your HuggingFace Space.
#
# Usage:  ./spaces/sync.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

SRC_PKG="$REPO/src/anonymisation"
SRC_EX="$REPO/demo/examples"

DST_PKG="$HERE/anonymisation"
DST_EX="$HERE/examples"

if [[ ! -d "$SRC_PKG" ]]; then
    echo "ERROR: $SRC_PKG not found"
    exit 1
fi

echo "── Syncing anonymisation package ──"
rm -rf "$DST_PKG"
cp -r "$SRC_PKG" "$DST_PKG"
# Remove __pycache__ if it sneaked in
find "$DST_PKG" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "  → $DST_PKG ($(find "$DST_PKG" -name "*.py" | wc -l | tr -d ' ') python files)"

echo "── Syncing examples ──"
rm -rf "$DST_EX"
cp -r "$SRC_EX" "$DST_EX"
echo "  → $DST_EX ($(ls "$DST_EX" | wc -l | tr -d ' ') files)"

echo
echo "Sync complete. To deploy:"
echo "  1. cp -r $HERE/* /path/to/your/space-clone/"
echo "  2. cd /path/to/your/space-clone/"
echo "  3. git add . && git commit -m 'sync' && git push"
