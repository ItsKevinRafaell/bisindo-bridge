// ICE Server configuration for WebRTC
//
// STUN (Google) — basic NAT traversal (no media relay).
// TURN (Metered openrelay, free public) — CGNAT / campus WiFi / cross-network.
// meeting.js may also append servers from /api/turn when the server sets env vars.

const ICE_SERVERS = [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },

    // Public free TURN — https://www.metered.ca/tools/openrelay/
    {
        urls: 'turn:openrelay.metered.ca:80',
        username: 'openrelayproject',
        credential: 'openrelayproject',
    },
    {
        urls: 'turn:openrelay.metered.ca:443',
        username: 'openrelayproject',
        credential: 'openrelayproject',
    },
    {
        urls: 'turn:openrelay.metered.ca:443?transport=tcp',
        username: 'openrelayproject',
        credential: 'openrelayproject',
    },
];
