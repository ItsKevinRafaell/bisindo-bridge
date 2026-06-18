#!/bin/bash
# Tunnel script untuk BISINDO Meeting
# Usage: ./start_tunnel.sh

PORT=5000

echo "🚀 Starting BISINDO Meeting Tunnel..."

# Kill existing tunnels
pkill -f "serveo.net" 2>/dev/null
pkill -f cloudflared 2>/dev/null

# Try serveo first
echo "Trying serveo.net..."
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -R 80:localhost:$PORT serveo.net 2>&1 &
SSH_PID=$!

# Wait for tunnel URL
sleep 8

# Check if running
if ps -p $SSH_PID > /dev/null 2>&1; then
    echo "✅ Tunnel running! PID: $SSH_PID"
    echo "📋 Share this URL with friends: http://$PORT.serveo.net"
    echo ""
    echo "Press Ctrl+C to stop tunnel"
    wait $SSH_PID
else
    echo "❌ Tunnel failed to start"
fi
