#!/usr/bin/env bash
# Start ngrok tunnel + auto-update web/config.js dengan URL tunnel.
#
# Prereq: ngrok login (sekali saja)
#   ngrok config add-authtoken <YOUR_TOKEN>
#
# Setelah jalan, URL ngrok akan otomatis ditulis ke web/config.js.
# Commit + push untuk deploy ke Vercel, atau teman bisa buka langsung.
set -e
cd "$(dirname "$0")"

# Start ngrok in background
ngrok http 5000 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!
trap "kill $NGROK_PID 2>/dev/null" EXIT

echo "⏳ Waiting for ngrok tunnel URL..."
for i in $(seq 1 15); do
  URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data['tunnels']:
        if t['proto'] == 'https':
            print(t['public_url'])
            break
except: pass
" 2>/dev/null)
  if [ -n "$URL" ]; then
    break
  fi
  sleep 1
done

if [ -z "$URL" ]; then
  echo "❌ Failed to get ngrok URL. Pastikan: ngrok config add-authtoken <TOKEN>"
  exit 1
fi

echo "✅ Tunnel URL: $URL"
echo ""
echo "📋 Share this link dengan teman:"
echo "  👉 $URL/capture"
echo ""
echo "💡 Capture page served langsung dari Flask (same-origin, tanpa CORS/ngrok interstitial)"
echo "💡 Tekan Ctrl+C untuk stop tunnel"

# Keep running
wait $NGROK_PID
