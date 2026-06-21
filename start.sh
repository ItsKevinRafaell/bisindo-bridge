#!/usr/bin/env bash
# Start the BISINDO Bridge laptop server (meeting + capture ingest + REST).
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r meeting/requirements.txt
exec python3 meeting/app.py
