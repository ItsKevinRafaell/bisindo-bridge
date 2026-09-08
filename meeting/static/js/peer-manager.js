/**
 * PeerManager - WebRTC mesh P2P (full mesh, ≤10 peers).
 * Each peer connects to every other peer directly. Server only relays signaling.
 *
 * Topology:
 *   N peers → N(N-1)/2 connections. Fine for ≤10. For >10, switch to SFU.
 *
 * Signaling (relayed via server SocketIO):
 *   webrtc_offer: {targetSid, sdp}
 *   webrtc_answer: {targetSid, sdp}
 *   webrtc_ice:    {targetSid, candidate}
 *
 * Reliability fixes vs. previous version:
 *   - Loads ICE servers (incl. TURN) from a runtime array instead of hardcoding
 *     only Google STUN. Without TURN, peers behind symmetric NAT / CGNAT can
 *     never connect → "only 2 visible / all black" on some laptops.
 *   - Auto ICE-restart on `disconnected` and full reconnect on `failed`
 *     (previously only `failed`/`closed` were handled, and nothing reconnected
 *     → "video disappears and never comes back").
 *   - Join race: when WE join an existing room we proactively connect to every
 *     peer already present (not only to peers that join later), so all N peers
 *     actually link up instead of a subset.
 *   - Tracks are attached as soon as they arrive; the <video> element is told to
 *     play on both `track` and `loadedmetadata` to avoid a black frame.
 */

class PeerManager {
    constructor(socket, localStream, roomId, username, iceServers) {
        this.socket = socket;
        this.localStream = localStream;
        this.roomId = roomId;
        this.username = username;
        // ICE servers: STUN + (public) TURN. Passed in from meeting.js after
        // fetching /api/turn so credentials never live in the client bundle.
        this.iceServers = iceServers && iceServers.length
            ? iceServers
            : [{ urls: 'stun:stun.l.google.com:19302' }, { urls: 'stun:stun1.l.google.com:19302' }];

        // Callbacks (set by meeting.js)
        this.onRemoteStream = null;
        this.onRemoveStream = null;
        this.onPeerConnected = null;   // (sid) -> void, for UI feedback

        // Map<sid, RTCPeerConnection>
        this.peerConnections = new Map();
        // Map<sid, MediaStream> — for meeting.js consumers
        this.remoteStreams = new Map();
        // Map<sid, {username}> — metadata for compatibility
        this.peers = new Map();
        // Map<sid, boolean> — re-entrancy guard for reconnect storms
        this._reconnecting = new Map();

        this._setupSocketListeners();
    }

    _setupSocketListeners() {
        this.socket.on('webrtc_offer', async (data) => {
            const fromSid = data.fromSid;
            const pc = this._getOrCreatePC(fromSid, data.fromUsername || 'Peer');
            try {
                await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
                const answer = await pc.createAnswer();
                await pc.setLocalDescription(answer);
                this.socket.emit('webrtc_answer', {
                    targetSid: fromSid,
                    sdp: answer,
                    username: this.username,
                });
            } catch (e) {
                console.error('[PeerManager] offer handling failed:', e);
            }
        });

        this.socket.on('webrtc_answer', async (data) => {
            const fromSid = data.fromSid;
            const pc = this.peerConnections.get(fromSid);
            if (!pc) return;
            try {
                await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
            } catch (e) {
                console.error('[PeerManager] answer handling failed:', e);
            }
        });

        this.socket.on('webrtc_ice', async (data) => {
            const fromSid = data.fromSid;
            const pc = this.peerConnections.get(fromSid);
            if (!pc || !data.candidate) return;
            try {
                await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
            } catch (e) {
                // Some candidates are duplicates — ignore those
                if (!String(e).includes('duplicate')) {
                    console.warn('[PeerManager] ICE add failed:', e);
                }
            }
        });

        // ICE restart requested by remote (network change / recovery)
        this.socket.on('webrtc_restart', async (data) => {
            const pc = this.peerConnections.get(data.fromSid);
            if (pc) {
                try {
                    const offer = await pc.createOffer({ iceRestart: true });
                    await pc.setLocalDescription(offer);
                    this.socket.emit('webrtc_offer', {
                        targetSid: data.fromSid,
                        sdp: offer,
                        username: this.username,
                    });
                } catch (e) {
                    console.error('[PeerManager] ICE restart failed:', e);
                }
            }
        });
    }

    _getOrCreatePC(remoteSid, remoteUsername) {
        if (this.peerConnections.has(remoteSid)) {
            return this.peerConnections.get(remoteSid);
        }

        const pc = new RTCPeerConnection({
            iceServers: this.iceServers,
            // Pre-gather candidates so connection is faster once offer/answer flows.
            iceCandidatePoolSize: 1,
        });

        // Add local tracks
        if (this.localStream) {
            for (const track of this.localStream.getTracks()) {
                pc.addTrack(track, this.localStream);
            }
        }

        // ICE candidate → server
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.socket.emit('webrtc_ice', {
                    targetSid: remoteSid,
                    candidate: event.candidate,
                });
            }
        };

        // Track received → always use native stream, addTrack as backup
        pc.ontrack = (event) => {
            let stream;
            if (event.streams && event.streams[0]) {
                stream = event.streams[0];
            } else {
                stream = this.remoteStreams.get(remoteSid);
                if (!stream) {
                    stream = new MediaStream();
                }
            }
            this.remoteStreams.set(remoteSid, stream);
            // Always add track manually in case native stream didn't carry it
            try { stream.addTrack(event.track); } catch(_){}
            console.log(`[PeerManager] ontrack ${event.track.kind} from ${remoteUsername} (${remoteSid.slice(0,6)}) ice=${pc.iceConnectionState} conn=${pc.connectionState} tracks=${stream.getTracks().length} audioTracks=${stream.getAudioTracks().length} videoTracks=${stream.getVideoTracks().length}`);
            if (this.onRemoteStream) {
                this.onRemoteStream(remoteSid, remoteUsername, stream);
            }
        };

        pc.onconnectionstatechange = () => {
            const state = pc.connectionState;
            console.log(`[PeerManager] ${remoteUsername} (${remoteSid.slice(0,6)}): ${state}`);
            if (state === 'connected') {
                this._reconnecting.delete(remoteSid);
                if (this.onPeerConnected) this.onPeerConnected(remoteSid);
            } else if (state === 'failed' || state === 'closed') {
                this.handlePeerDisconnect(remoteSid);
            }
        };

        // ICE state logging + recovery
        pc.oniceconnectionstatechange = () => {
            const state = pc.iceConnectionState;
            const conn = pc.connectionState;
            console.log(`[PeerManager] ICE ${remoteUsername} (${remoteSid.slice(0,6)}): ice=${state} conn=${conn} candidates=${pc.localDescription ? (pc.localDescription.sdp.match(/candidate/g)||[]).length : 0}`);
            if (state === 'disconnected') {
                this._tryIceRestart(remoteSid);
            } else if (state === 'failed') {
                this._tryReconnect(remoteSid);
            }
        };

        this.peerConnections.set(remoteSid, pc);
        this.peers.set(remoteSid, { username: remoteUsername });
        return pc;
    }

    /** Attempt an ICE restart (cheap) when the ICE connection drops. */
    async _tryIceRestart(remoteSid) {
        if (this._reconnecting.get(remoteSid)) return;
        this._reconnecting.set(remoteSid, true);
        const pc = this.peerConnections.get(remoteSid);
        if (!pc) return;
        try {
            const offer = await pc.createOffer({ iceRestart: true });
            await pc.setLocalDescription(offer);
            this.socket.emit('webrtc_offer', {
                targetSid: remoteSid,
                sdp: offer,
                username: this.username,
            });
            console.log(`[PeerManager] ICE restart sent to ${remoteSid.slice(0,6)}`);
        } catch (e) {
            console.error('[PeerManager] ICE restart failed:', e);
            this._tryReconnect(remoteSid);
        }
    }

    /** Full reconnect (close + re-offer) when ICE restart didn't recover.
     *  CRITICAL FIX: force offer regardless of SID polarity to avoid deadlock where
     *  both sides wait. Guard still prevents double-offer storm.
     */
    async _tryReconnect(remoteSid) {
        if (this._reconnecting.get(remoteSid)) return;
        this._reconnecting.set(remoteSid, true);
        const pc = this.peerConnections.get(remoteSid);
        const username = this.peers.get(remoteSid)?.username || 'Peer';
        try { if (pc) pc.close(); } catch (e) { /* ignore */ }
        this.peerConnections.delete(remoteSid);
        this.remoteStreams.delete(remoteSid);
        this.peers.delete(remoteSid);
        // Force offer on reconnect (ignore polarity) — otherwise both sides may wait → black forever
        const mySid = this.socket && this.socket.id;
        const newPc = this._getOrCreatePC(remoteSid, username);
        try {
            const offer = await newPc.createOffer({ iceRestart: true });
            await newPc.setLocalDescription(offer);
            this.socket.emit('webrtc_offer', {
                targetSid: remoteSid,
                sdp: offer,
                username: this.username,
            });
            console.log(`[PeerManager] reconnect offer → ${username} (${remoteSid.slice(0,6)}) (forced)`);
        } catch (e) {
            console.error('[PeerManager] reconnect offer failed:', e);
            // Fallback to polarity-based path
            this.connectToPeer(remoteSid, username);
        }
        setTimeout(() => this._reconnecting.delete(remoteSid), 15000);
    }

    /** Called when a new peer joins the room. We initiate the offer if our SID is lower. */
    async connectToPeer(remoteSid, remoteUsername) {
        if (!remoteSid || remoteSid === this.socket.id) return;
        if (this.peerConnections.has(remoteSid)) return;
        const mySid = this.socket.id;
        if (!mySid) {
            // Socket not fully connected yet — retry shortly
            setTimeout(() => this.connectToPeer(remoteSid, remoteUsername), 200);
            return;
        }
        // Deterministic polarity: lexicographically greater SID initiates
        if (mySid > remoteSid) {
            const pc = this._getOrCreatePC(remoteSid, remoteUsername);
            try {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                this.socket.emit('webrtc_offer', {
                    targetSid: remoteSid,
                    sdp: offer,
                    username: this.username,
                });
                console.log(`[PeerManager] offer → ${remoteUsername} (${remoteSid.slice(0,6)})`);
            } catch (e) {
                console.error('[PeerManager] connectToPeer failed:', e);
            }
        } else {
            // We wait for their offer
            this._getOrCreatePC(remoteSid, remoteUsername);
            console.log(`[PeerManager] wait offer from ${remoteUsername} (${remoteSid.slice(0,6)})`);
        }
    }

    handlePeerDisconnect(sid) {
        const pc = this.peerConnections.get(sid);
        if (pc) {
            try { pc.close(); } catch (e) { /* ignore */ }
            this.peerConnections.delete(sid);
        }
        this.remoteStreams.delete(sid);
        this.peers.delete(sid);
        this._reconnecting.delete(sid);

        if (this.onRemoveStream) {
            this.onRemoveStream(sid);
        }
    }

    /** Called when our own socket reconnects with a new sid — tear down stale PCs. */
    handleReconnectCleanup() {
        console.log('[PeerManager] Reconnect cleanup: closing all peer connections');
        for (const [, pc] of this.peerConnections) {
            try { pc.close(); } catch (e) { /* ignore */ }
        }
        // Notify UI to remove cards
        for (const sid of this.peerConnections.keys()) {
            if (this.onRemoveStream) this.onRemoveStream(sid);
        }
        this.peerConnections.clear();
        this.remoteStreams.clear();
        this.peers.clear();
        this._reconnecting.clear();
    }

    destroy() {
        console.log('[PeerManager] Destroying mesh connections');
        for (const [, pc] of this.peerConnections) {
            try { pc.close(); } catch (e) { /* ignore */ }
        }
        this.peerConnections.clear();
        this.remoteStreams.clear();
        this.peers.clear();
        this._reconnecting.clear();
    }
}
