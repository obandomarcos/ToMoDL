#!/usr/bin/env bash
# ================================================
# PyTorch Installer for Napari – Auto-Detect ANY version (Linux/Mac)
# Searches a user-selected path (default: ~/.local) and installs torch using pip
# ================================================

set -euo pipefail

PYTORCH_VERSION="2.5.0"

echo
echo "========================================"
echo " PyTorch Installer for Napari (Auto-Detect)"
echo "========================================"
echo

# -------------------------------------------------
# 1. Ask for the Napari installation path
# -------------------------------------------------
DEFAULT_APPDATA_ROOT="$HOME/.local"
NAPARI_PATH_INPUT=""

read -r -p "Napari installation directory [$DEFAULT_APPDATA_ROOT]: " NAPARI_PATH_INPUT || true

if [ -z "$NAPARI_PATH_INPUT" ]; then
    APPDATA_ROOT="$DEFAULT_APPDATA_ROOT"
elif [ "$NAPARI_PATH_INPUT" = "~" ]; then
    APPDATA_ROOT="$HOME"
elif [[ "$NAPARI_PATH_INPUT" == "~/"* ]]; then
    APPDATA_ROOT="$HOME/${NAPARI_PATH_INPUT:2}"
else
    APPDATA_ROOT="$NAPARI_PATH_INPUT"
fi

echo "Scanning for Napari folder in:"
echo "  $APPDATA_ROOT"
echo

# -------------------------------------------------
# 2. Look for napari-* folders
# -------------------------------------------------
NAPARI_ENV=""
PYTHON_EXE=""

if [[ "$(basename "$APPDATA_ROOT")" == napari-* ]]; then
    NAPARI_FOLDERS=("$APPDATA_ROOT")
else
    NAPARI_FOLDERS=("$APPDATA_ROOT"/napari-*)
fi

for folder in "${NAPARI_FOLDERS[@]}"; do
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
