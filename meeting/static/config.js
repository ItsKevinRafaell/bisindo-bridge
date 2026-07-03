// ICE Server configuration for WebRTC
// Get free TURN credentials from Cloudflare Calls (zero cost)

const ICE_SERVERS = [
    // Google public STUN (always works for basic NAT traversal)
    { urls: 'stun:stun.l.google.com:19302' },

    // Cloudflare TURN (free tier)
    // 1. Go to https://dash.cloudflare.com
    // 2. Workers & Pages → Calls
    // 3. Create TURN credentials
    // 4. Replace the placeholder values below
    {
        urls: 'turn:turn.cloudflare.com:3478?transport=udp',
        username: 'YOUR_CLOUDFLARE_TURN_USERNAME',
        credential: 'YOUR_CLOUDFLARE_TURN_CREDENTIAL'
    },
    {
        urls: 'turn:turn.cloudflare.com:3478?transport=tcp',
        username: 'YOUR_CLOUDFLARE_TURN_USERNAME',
        credential: 'YOUR_CLOUDFLARE_TURN_CREDENTIAL'
    }
];
