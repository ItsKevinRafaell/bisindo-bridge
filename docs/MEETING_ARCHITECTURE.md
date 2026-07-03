# Meeting Architecture — MVP untuk 10+ Peserta

## Problem

Current impl relay video via SocketIO (`base64 JPEG`, 5fps, 320x240).
Server bandwidth = O(n²). With 10 users: ~90 frames/s relayed. Unusable.
Audio captured tapi **ga dikirim** sama sekali.

## Solution: WebRTC Full Mesh

```
┌─────────────────────────────────────────────────┐
│              Flask-SocketIO Server               │
│         (signaling only, ZERO media relay)       │
│                                                   │
│  Handles:                                         │
│  • Room management (join/leave/user list)        │
│  • WebRTC signaling (offer/answer/ICE)           │
│  • Chat messages                                   │
│  • Gesture/sentence data relay                     │
│                                                   │
│  STUN: stun.l.google.com:19302 (free)            │
│  TURN: optional (only if symmetric NAT)          │
└──────────────┬──────────────┬────────────────────┘
               │              │
      ┌────────▼──┐      ┌───▼────────┐
      │  Peer A   │◄────►│   Peer B   │
      │ (browser) │ P2P  │ (browser)  │
      └─────┬─────┘      └──────┬─────┘
            │                    │
            └────── P2P ────────┘
           audio + video streams
```

### Why Full Mesh?

| Topology | Pros | Cons | For 10p? |
|----------|------|------|----------|
| Full Mesh | Low latency, no server cost | O(n²) connections | ✅ 45 conns, fine |
| SFU (mediasoup) | Scales to 50+ | Complex server, $$$ | ❌ Overkill |
| MCU | Single stream out | Heavy server CPU | ❌ Way overkill |

10 peers = 45 PeerConnections. Each browser handles it fine.
Google Meet uses mesh for small calls too.

### Mesh Connection Budget (10 users)

```
Each peer: 9 outbound + 9 inbound = 18 PeerConnections
Total unique connections: n(n-1)/2 = 45
Bandwidth per peer: ~500kbps video + ~40kbps audio = ~540kbps
Each peer uploads: 9 × 540kbps ≈ 4.8 Mbps (fine for most connections)
Server bandwidth: ~0 (signaling only, few KB/s)
```

## Implementation Plan

### Phase 1: WebRTC Core (must-have)

#### 1.1 Signaling Protocol (SocketIO events)

```
Client → Server:
  webrtc_offer    { targetSid, sdp, room }
  webrtc_answer   { targetSid, sdp, room }
  webrtc_ice      { targetSid, candidate, room }

Server → Client:
  webrtc_offer    { fromSid, fromUsername, sdp }
  webrtc_answer   { fromSid, sdp }
  webrtc_ice      { fromSid, candidate }
```

Flow when new peer joins:
```
1. New peer → server: join { room, username }
2. Server → room: user_joined { username, sid }
3. EXISTING peers initiate offers to new peer
   (avoids glare — only one side creates offer)
4. New peer answers each offer
5. ICE candidates exchanged until connected
```

#### 1.2 Client-side WebRTC Manager

```js
class PeerManager {
  peers = new Map();  // sid → RTCPeerConnection

  async createPeerConnection(remoteSid, remoteUsername) {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    // Add local tracks
    this.localStream.getTracks().forEach(track => {
      pc.addTrack(track, this.localStream);
    });

    // Receive remote tracks
    pc.ontrack = (event) => {
      this.showRemoteStream(remoteSid, remoteUsername, event.streams[0]);
    };

    // ICE candidate → signal to peer via server
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        socket.emit('webrtc_ice', {
          targetSid: remoteSid,
          candidate: event.candidate,
          room: roomId
        });
      }
    };

    // Connection state monitoring
    pc.onconnectionstatechange = () => {
      console.log(`${remoteUsername}: ${pc.connectionState}`);
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        this.handleDisconnect(remoteSid);
      }
    };

    this.peers.set(remoteSid, pc);
    return pc;
  }
}
```

#### 1.3 Remove Old Video Relay

Delete:
- `startVideoSharing()` — base64 frame capture
- `showRemoteVideo()` — `<img>` tag update
- `socket.on('video_frame')` handler
- Server `on_video_frame` event

Replace with:
- WebRTC `<video>` elements with real MediaStream
- Remote peer cards use `<video autoplay playsinline>` not `<img>`

### Phase 2: Stability & Quality

#### 2.1 Connection Quality

```js
// Adaptive bitrate — reduce quality if connection struggles
pc.onconnectionstatechange = () => {
  if (pc.connectionState === 'disconnected') {
    // Wait 5s, if still disconnected → reconnect
    setTimeout(() => this.maybeReconnect(sid), 5000);
  }
};

// Sender bitrate constraints (important for mesh with 10 peers)
const sender = pc.getSenders().find(s => s.track?.kind === 'video');
if (sender) {
  sender.setParameters({
    ...sender.getParameters(),
    encodings: [{
      maxBitrate: 300_000,   // 300kbps cap per stream
      maxFramerate: 15,      // 15fps is enough for gestures
    }]
  });
}
```

Key constraint: **15fps is enough** — we're doing sign language, not watching movies.
Lower fps = less bandwidth = more stable with 10 peers.

#### 2.2 Reconnection Logic

```
Peer disconnected →
  1. Wait 3s (might be temporary network blip)
  2. If still disconnected → close old PC
  3. Re-create offer (existing peer initiates)
  4. Re-negotiate via signaling server
  5. Log reconnection attempts (max 3)
```

#### 2.3 Audio

Already captured via `getUserMedia({audio: true})`.
WebRTC `addTrack` sends it automatically. **Zero extra code needed.**

Toggle mic/camera already works (disables track, WebRTC sends black frames / silence).

### Phase 3: Nice-to-Have (skip if MVP works)

| Feature | Effort | Value |
|---------|--------|-------|
| TURN server | Medium | Needed if >20% fail to connect (NAT issues) |
| Simulcast | Low | Send 2 quality levels, receiver picks |
| Active speaker detection | Medium | Highlight who's talking |
| Screen sharing | Low | `getDisplayMedia()` + replaceTrack |
| Recording | High | Server-side SFU needed, skip for MVP |

## File Changes

```
meeting/
├── app.py                          # Add WebRTC signaling events
│   ├── + webrtc_offer              # Relay offer SDP
│   ├── + webrtc_answer             # Relay answer SDP
│   └── + webrtc_ice                # Relay ICE candidates
│
├── static/js/
│   ├── peer-manager.js             # NEW: WebRTC peer connection manager
│   ├── meeting.js                  # MODIFY: remove base64 relay, use PeerManager
│   └── sentence-builder.js         # NO CHANGE
│
└── templates/
    └── index.html                  # MODIFY: add peer-manager.js script tag
```

### app.py — New Socket Events

```python
# WebRTC signaling relay
@socketio.on('webrtc_offer')
def on_offer(data):
    emit('webrtc_offer', {
        'fromSid': request.sid,
        'fromUsername': data.get('username'),
        'sdp': data['sdp']
    }, to=data['targetSid'])

@socketio.on('webrtc_answer')
def on_answer(data):
    emit('webrtc_answer', {
        'fromSid': request.sid,
        'sdp': data['sdp']
    }, to=data['targetSid'])

@socketio.on('webrtc_ice')
def on_ice(data):
    emit('webrtc_ice', {
        'fromSid': request.sid,
        'candidate': data['candidate']
    }, to=data['targetSid'])
```

### meeting.js — Key Changes

```diff
- let videoShareInterval = null;
+ let peerManager = null;

  async function joinMeeting() {
    // ... getUserMedia ...
-   startVideoSharing();
+   peerManager = new PeerManager(socket, localStream, roomId, username);
  }

- function startVideoSharing() { /* DELETE: base64 frame relay */ }
- function showRemoteVideo(name, frameData) { /* DELETE */ }
- socket.on('video_frame', ...) { /* DELETE */ }

+ function showRemoteStream(sid, username, stream) {
+   let card = document.getElementById('card_' + sid);
+   if (!card) {
+     card = createVideoCard(sid, username);
+     document.getElementById('videoGrid').appendChild(card);
+   }
+   card.querySelector('video').srcObject = stream;
+   updateVideoGrid();
+ }
```

### index.html — Add Script

```diff
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js"></script>
+ <script src="/static/js/peer-manager.js"></script>
  <script src="/static/js/sentence-builder.js"></script>
  <script src="/static/js/meeting.js"></script>
```

## Deployment

### Local Dev
```bash
cd meeting && python app.py
# Open http://localhost:4500 in 2+ tabs/browsers
```

### Production (VPS)
- STUN works behind most NATs (home/office routers)
- If >20% peers can't connect → deploy TURN server (coturn, ~$5/mo VPS)
- Cloudflare Tunnel for HTTPS (required for getUserMedia on non-localhost)

### Bandwidth Budget (10 users, 15fps, 300kbps)
```
Per peer upload:  9 peers × 300kbps = 2.7 Mbps
Per peer download: 9 peers × 300kbps = 2.7 Mbps
Server:           ~5 KB/s (signaling only)
```

## Testing Checklist

- [ ] 2 peers: audio + video works
- [ ] 5 peers: all video tiles show, audio works
- [ ] 10 peers: stable, no freezing
- [ ] Toggle mic/camera works for all peers
- [ ] Peer leaves gracefully (no ghost tiles)
- [ ] Peer disconnects → auto-reconnect within 5s
- [ ] Gesture recognition still works during call
- [ ] Chat + sentence builder still works
- [ ] NAT traversal: test from different networks (home + mobile hotspot)

## Known Limitations (MVP)

1. **No TURN** — will fail behind symmetric NAT (~10-20% of users)
2. **No recording** — needs server-side media processing (SFU)
3. **No screen share** — easy add later, skip for MVP
4. **15fps video** — fine for gestures, not for watching movies together
5. **Full mesh** — caps at ~10-12 peers, then need SFU upgrade

## Future (Post-MVP)

If need >10 peers → migrate to SFU:
- **mediasoup** (Node.js, battle-tested) or **ion-sfu** (Go)
- Server receives 1 stream per peer, distributes to all
- Bandwidth: O(n) instead of O(n²)
- Same signaling server (Flask-SocketIO), just add media server

```
meeting/
├── app.py                     # Add WebRTC signaling events (~50 lines)
├── static/js/
│   ├── meeting.js             # Replace video relay with PeerManager (~200 lines rewritten)
│   ├── peer-manager.js        # NEW: WebRTC connection manager (~150 lines)
│   └── sentence-builder.js    # NO CHANGE
└── templates/
    └── index.html             # Add peer-manager.js script tag
```

### Server Changes (app.py)

Add 3 socket events, delete `on_video_frame`:

```python
@socketio.on('webrtc_offer')
def on_offer(data):
    target_sid = data['targetSid']
    emit('webrtc_offer', {
        'fromSid': request.sid,
        'fromUsername': data.get('username'),
        'sdp': data['sdp']
    }, to=target_sid)

@socketio.on('webrtc_answer')
def on_answer(data):
    target_sid = data['targetSid']
    emit('webrtc_answer', {
        'fromSid': request.sid,
        'sdp': data['sdp']
    }, to=target_sid)

@socketio.on('webrtc_ice')
def on_ice(data):
    target_sid = data['targetSid']
    emit('webrtc_ice', {
        'fromSid': request.sid,
        'candidate': data['candidate']
    }, to=target_sid)
```

### Client Changes (meeting.js)

1. Remove `startVideoSharing()`, `showRemoteVideo()`, `on('video_frame')`
2. Add `PeerManager` class
3. `joinMeeting()` → after `getUserMedia`, create offers to existing peers
4. `on('user_joined')` → create offer to new peer
5. `on('user_left')` → close PeerConnection, remove video card
6. Video cards: `<video autoplay playsinline>` instead of `<img>`

## Deployment

### Dev (local)

```bash
cd meeting && python app.py
# Works on localhost, no TURN needed
```

### Production (VPS)

```bash
# Same Flask-SocketIO, behind nginx + HTTPS (required for WebRTC)
# Free STUN from Google is sufficient
# Only add TURN if users report connection failures

# nginx config snippet:
# server {
#     listen 443 ssl;
#     location / {
#         proxy_pass http://127.0.0.1:4500;
#         proxy_http_version 1.1;
#         proxy_set_header Upgrade $http_upgrade;
#         proxy_set_header Connection "upgrade";
#     }
# }
```

**HTTPS is required** — browsers block `getUserMedia` on non-secure origins (except localhost).

## Testing Checklist

- [ ] 2 users: video + audio works
- [ ] 2 users: mic/camera toggle works
- [ ] 5 users: all see each other, audio clear
- [ ] 10 users: stable for 10+ minutes
- [ ] Peer disconnect/reconnect (unplug wifi, plug back)
- [ ] Gesture recognition still works with WebRTC video
- [ ] Mobile browser (Android Chrome)

## Summary

| Metric | Before (current) | After (WebRTC mesh) |
|--------|------------------|---------------------|
| Video transport | Server relay (base64) | P2P WebRTC |
| Audio | ❌ Not streamed | ✅ P2P WebRTC |
| Server bandwidth | O(n²) frames | ~0 (signaling only) |
| Max users (practical) | 3-4 | 10-15 |
| Latency | ~500ms (server hop) | ~50ms (P2P) |
| Code changes | - | ~400 lines |
