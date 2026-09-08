#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "node_modules" ]; then
  echo "[sfu] node_modules missing — running npm install…"
  npm install
fi

echo "[sfu] starting mediasoup SFU server on port 4501"
exec node server.js
