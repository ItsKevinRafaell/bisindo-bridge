#!/bin/bash
# Setup BISINDO systemd services (run once)
# Note: ngrok free tier doesn't work well with systemd (needs TUI).
# Flask runs via systemd, ngrok via start-ngrok.sh (manual).
set -e
cd "$(dirname "$0")"

echo "📋 Installing BISINDO systemd services..."

# Only Flask via systemd (ngrok runs manually)
ln -sf "$PWD/systemd/bisindo-server.service" ~/.config/systemd/user/bisindo-server.service

# Reload systemd
systemctl --user daemon-reload

# Enable (auto-start on login)
systemctl --user enable bisindo-server.service

# Start now
systemctl --user start bisindo-server.service

echo ""
echo "✅ Flask server auto-starts on login!"
echo ""
echo "📊 Check status:"
echo "  systemctl --user status bisindo-server"
echo ""
echo "🔗 Start ngrok tunnel:"
echo "  ./start-ngrok.sh"
echo ""
echo "🛑 Stop Flask:"
echo "  systemctl --user stop bisindo-server"
