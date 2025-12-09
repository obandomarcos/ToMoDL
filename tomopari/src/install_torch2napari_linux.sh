#!/usr/bin/env bash
# ================================================
# PyTorch Installer for Napari – Auto-Detect ANY version (Linux/Mac)
# Automatically finds ~/.local/napari-* and installs torch using pip
# ================================================

set -euo pipefail

PYTORCH_VERSION="2.5.0"

echo
echo "========================================"
echo " PyTorch Installer for Napari (Auto-Detect)"
echo "========================================"
echo

# -------------------------------------------------
# 1. Define Napari root path (~/.local/)
# -------------------------------------------------
APPDATA_ROOT="$HOME/.local"
echo "Scanning for Napari folder in:"
echo "  $APPDATA_ROOT"
echo

# -------------------------------------------------
# 2. Look for napari-* folders
# -------------------------------------------------
NAPARI_ENV=""
PYTHON_EXE=""

for folder in "$APPDATA_ROOT"/napari-*; do
    [ -d "$folder" ] || continue

    echo "  [CHECK] $folder"

    TEST_PYTHON="$folder/envs/$(basename "$folder")/bin/python"

    if [ -x "$TEST_PYTHON" ]; then
        NAPARI_ENV="$folder"
        PYTHON_EXE="$TEST_PYTHON"
        echo "  [FOUND] Napari environment: $(basename "$folder")"
        break
    fi
done

# -------------------------------------------------
# 3. Handle no environment found
# -------------------------------------------------
if [ -z "$NAPARI_ENV" ]; then
    echo
    echo "[ERROR] No Napari installation found!"
    echo "Expected folder pattern: napari-x.x.x  (e.g. napari-0.6.5)"
    echo
    echo "Make sure Napari was installed with the official Linux installer."
    exit 1
fi

# -------------------------------------------------
# 4. Install PyTorch with pip
# -------------------------------------------------
echo
echo "[INFO] Installing torch==$PYTORCH_VERSION into $NAPARI_ENV ..."
"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install torch=="$PYTORCH_VERSION"

echo
echo "✅ Installation complete!"
echo "Environment: $NAPARI_ENV"
echo "Python used: $PYTHON_EXE"
echo "PyTorch version: $PYTORCH_VERSION"
echo
