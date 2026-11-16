#!/usr/bin/env bash
# Clones upstream NeuroSeg3 at the commit this work was run against and applies
# the NumPy 2 overflow fix. Then: export NEUROSEG_REPO="$PWD/NeuroSeg3"
set -euo pipefail
DEST="${1:-NeuroSeg3}"
SHA=b5b53617676fdf6f7cb9b0167794a17334e0ec21   # zimo-5kk/NeuroSeg3 main, 2024-07-05
HERE="$(cd "$(dirname "$0")" && pwd)"
git clone https://github.com/zimo-5kk/NeuroSeg3 "$DEST"
git -C "$DEST" checkout -q "$SHA"
git -C "$DEST" apply "$HERE/patches/neuroseg3-numpy2-overflow.patch"
echo "done. export NEUROSEG_REPO=\"$(cd "$DEST" && pwd)\""
