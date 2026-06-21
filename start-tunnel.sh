#!/usr/bin/env bash
# Run a Cloudflare named tunnel to expose the laptop server publicly.
# Prereq: `cloudflared tunnel login` + `cloudflared tunnel create bisindo`,
# then update ~/cloudflared/config.yml with hostname bisindo-bridge.duckdns.org.
exec cloudflared tunnel run bisindo
