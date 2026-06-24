#!/usr/bin/env bash
# BISINDO Meeting Server - VPS-ready runner
# Usage: ./run-server.sh [--daemon] [--tunnel]
#
# VPS deploy notes:
#   - Uses nohup + background for daemon mode (no tmux needed)
#   - Auto-installs requirements
#   - Starts Cloudflare tunnel if --tunnel
#
set -e
cd "$(dirname "$0")"

DAEMON=false
TUNNEL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon) DAEMON=true; shift ;;
        --tunnel) TUNNEL=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Config ----
PIDFILE=".server.pid"
LOGFILE="meeting-server.log"
TUNNEL_LOGFILE="meeting-tunnel.log"
REPO_DIR="$(pwd)"
CSV_PATH="dataset/landmarks_captured_v2.csv"

# Ensure dataset dir exists
touch "$CSV_PATH" || mkdir -p dataset

# ---- Install deps ----
echo "📦 Checking dependencies..."
python3 -m pip install -q -r meeting/requirements.txt 2>&1 | grep -v "already satisfied" || true

# ---- Kill existing server ----
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "🔄 Killing existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

# ---- Cloudflare tunnel ----
start_tunnel() {
    if ! command -v cloudflared &>/dev/null; then
        echo "❌ cloudflared not found. Quick install:"
        echo "   wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared"
        echo "   chmod +x /usr/local/bin/cloudflared"
        exit 1
    fi

    echo "🌐 Starting Cloudflare Quick Tunnel..."
    cloudflared tunnel --url http://localhost:5000 --no-autoupdate > "$TUNNEL_LOGFILE" 2>&1 &
    TUNNEL_PID=$!

    echo "⏳ Waiting for tunnel URL..."
    TUNNEL_URL=""
    for i in $(seq 1 20); do
        TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOGFILE" 2>/dev/null | head -1 || true)
        [[ -n "$TUNNEL_URL" ]] && break
        sleep 1
    done

    if [[ -n "$TUNNEL_URL" ]]; then
        echo ""
        echo "✅ Cloudflare Tunnel URL: $TUNNEL_URL"
        echo "  👉 $TUNNEL_URL/capture  (Capture mode)"
        echo "  👉 $TUNNEL_URL/train    (Train mode)"
        echo ""
        echo "$TUNNEL_URL" > .tunnel-url
    else
        echo "⚠️  Could not detect tunnel URL. Check $TUNNEL_LOGFILE"
    fi
}

# ---- Start server ----
start_server() {
    echo "🚀 Starting BISINDO Meeting Server..."
    echo "   http://localhost:5000"

    if $DAEMON; then
        nohup python3 -u meeting/app.py >> "$LOGFILE" 2>&1 &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PIDFILE"
        echo "📋 Daemon PID: $SERVER_PID"
        echo "   logs: $LOGFILE"
    else
        python3 -u meeting/app.py &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PIDFILE"
        echo "📋 Server PID: $SERVER_PID"
    fi
}

# ---- Run ----
start_server

if $TUNNEL; then
    sleep 2
    start_tunnel
fi

if $DAEMON; then
    echo "✅ Daemon running. Stop with: kill \$(cat $PIDFILE)"
    exit 0
fi

# Wait for foreground server
wait $SERVER_PID 2>/dev/null || true
rm -f "$PIDFILE"
