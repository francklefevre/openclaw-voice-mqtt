#!/usr/bin/env bash
# Setup virtual environment and install dependencies.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Installing dependencies ..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install openwakeword --no-deps
"$VENV/bin/pip" install -r requirements.txt

echo ""
echo "Done. Activate with:  source $VENV/bin/activate"
