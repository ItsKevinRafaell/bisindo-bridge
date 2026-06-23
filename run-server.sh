#!/usr/bin/env bash
# BISINDO Meeting Server - Auto-restart + periodic git push + Cloudflare tunnel
# Usage: ./run-server.sh [--daemon] [--tunnel]
set -e
cd "$(dirname "$0")"

DAEMON=false
TUNNEL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon) DAEMON=true; shift ;;
        --tunnel) TUNNEL=true; shift ;;
        *) shift ;;
    esac
done

if $DAEMON; then
    echo "🚀 Starting in daemon mode (logs: meeting-server.log)"
fi

# ---- Config ----
PIDFILE=".server.pid"
LOGFILE="meeting-server.log"
TUNNEL_LOGFILE="meeting-tunnel.log"
REPO_DIR="$(pwd)"
CSV_PATH="dataset/landmarks_captured_v2.csv"
PUSH_INTERVAL=300  # git push every 5 minutes

cleanup() {
    echo "🛑 Stopping everything..."
    kill $TUNNEL_PID 2>/dev/null
    kill $PUSHER_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    rm -f "$PIDFILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- Install deps ----
echo "📦 Checking dependencies..."
python3 -m pip install -q -r meeting/requirements.txt 2>&1 | grep -v "already satisfied" || true

# ---- Kill existing server ----
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "🔄 Killing existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

# ---- Periodic git push ----
periodic_push() {
    while true; do
        sleep "$PUSH_INTERVAL"
        if git diff --quiet "$CSV_PATH" 2>/dev/null; then
            continue
        fi
        echo "📤 Periodic push: $(wc -l < "$CSV_PATH") rows..."
        git add "$CSV_PATH" 2>/dev/null && \
            git commit -m "data: auto-commit $(date +%H:%M)" 2>/dev/null && \
            git push origin main --quiet 2>/dev/null && \
            echo "  ✅ Pushed" || \
            echo "  ⚠️  Push failed (will retry)"
    done
}

# ---- Start Cloudflare tunnel ----
start_tunnel() {
    # Check cloudflared
    if ! command -v cloudflared &>/dev/null; then
        echo "❌ cloudflared not found! Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        echo "   Or run: brew install cloudflared / sudo apt install cloudflared"
        exit 1
    fi

    echo "🌐 Starting Cloudflare Quick Tunnel..."
    if $DAEMON; then
        cloudflared tunnel --url http://localhost:5000 --no-autoupdate > "$TUNNEL_LOGFILE" 2>&1 &
        TUNNEL_PID=$!
    else
        cloudflared tunnel --url http://localhost:5000 --no-autoupdate &
        TUNNEL_PID=$!
    fi

    # Wait for tunnel URL
    echo "⏳ Waiting for tunnel URL..."
    TUNNEL_URL=""
    for i in $(seq 1 15); do
        TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOGFILE" 2>/dev/null | head -1 || true)
        if [[ -n "$TUNNEL_URL" ]]; then
            break
        fi
        sleep 1
    done

    if [[ -n "$TUNNEL_URL" ]]; then
        echo ""
        echo "✅ Cloudflare Tunnel URL: $TUNNEL_URL"
        echo ""
        echo "📋 Share these links:"
        echo "  👉 $TUNNEL_URL/train    (Train mode)"
        echo "  👉 $TUNNEL_URL/capture  (Capture mode)"
        echo ""
    else
        echo "⚠️  Could not detect tunnel URL. Check $TUNNEL_LOGFILE"
        echo "💡 Cloudflare Quick Tunnel URL juga bisa dilihat dari console output di atas"
        echo ""
    fi
}

# ---- Start server with auto-restart ----
start_server() {
    echo "🚀 Starting BISINDO Meeting Server..."
    echo "   http://localhost:5000"
    echo "   http://$(hostname -I | awk '{print $1}'):5000"
    echo ""

    while true; do
        START_TIME=$(date +%s)

        if $DAEMON; then
            python3 meeting/app.py >> "$LOGFILE" 2>&1 &
            SERVER_PID=$!
        else
            python3 meeting/app.py &
            SERVER_PID=$!
        fi

        echo "$SERVER_PID" > "$PIDFILE"
        echo "📋 Server PID: $SERVER_PID"

        # Wait for exit
        wait $SERVER_PID
        EXIT_CODE=$?

        ELAPSED=$(( $(date +%s) - START_TIME ))

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "✅ Server exited cleanly"
            break
        fi

        # Don't restart too fast (crash loop protection)
        if [[ $ELAPSED -lt 5 ]]; then
            echo "⚠️  Server crashed after ${ELAPSED}s, waiting 3s before restart..."
            sleep 3
        else
            echo "🔄 Server exited (code $EXIT_CODE, ran ${ELAPSED}s), restarting in 2s..."
            sleep 2
        fi
    done
}

# ---- Run ----
if $TUNNEL; then
    # Server must start first so tunnel has something to connect to
    python3 meeting/app.py &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE"
    sleep 2

    start_tunnel

    periodic_push &
    PUSHER_PID=$!

    # Wait for everything
    wait $SERVER_PID 2>/dev/null
    cleanup
else
    periodic_push &
    PUSHER_PID=$!
    start_server
fi
