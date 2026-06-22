#!/bin/bash
# Setup BISINDO systemd services (run once)
set -e
cd "$(dirname "$0")"

echo "📋 Installing BISINDO systemd services..."

# Link services
ln -sf "$PWD/systemd/bisindo-server.service" ~/.config/systemd/user/bisindo-server.service
ln -sf "$PWD/systemd/ngrok.service" ~/.config/systemd/user/ngrok.service

# Reload systemd
systemctl --user daemon-reload

# Enable (auto-start on login)
systemctl --user enable bisindo-server.service
systemctl --user enable ngrok.service

# Start now
systemctl --user start bisindo-server.service
systemctl --user start ngrok.service

echo ""
echo "✅ Services installed and started!"
echo ""
echo "📊 Check status:"
echo "  systemctl --user status bisindo-server"
echo "  systemctl --user status ngrok"
echo ""
echo "🔗 Get ngrok URL:"
echo "  curl -s http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[0].public_url'"
echo ""
echo "🛑 Stop services:"
echo "  systemctl --user stop bisindo-server ngrok"
