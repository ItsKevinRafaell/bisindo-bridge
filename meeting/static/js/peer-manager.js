/**
 * PeerManager - WebRTC peer connection manager for full mesh topology
 * Handles signaling, track management, and bitrate optimization
 */

class PeerManager {
    constructor(socket, localStream, roomId, username) {
        this.socket = socket;
        this.localStream = localStream;
        this.roomId = roomId;
        this.username = username;
        this.peers = new Map(); // sid → { pc, username, stream }

        // Callbacks (set by meeting.js)
        this.onRemoteStream = null;
        this.onRemoveStream = null;

        this.setupSignaling();
    }

    setupSignaling() {
        this.socket.on('webrtc_offer', (data) => this.handleOffer(data));
        this.socket.on('webrtc_answer', (data) => this.handleAnswer(data));
        this.socket.on('webrtc_ice', (data) => this.handleIce(data));
    }

    async connectToPeer(remoteSid, remoteUsername) {
        console.log(`[PeerManager] Connecting to ${remoteUsername} (${remoteSid})`);

        const pc = this.createPeerConnection(remoteSid, remoteUsername);

        // Add local tracks (audio + video)
        this.localStream.getTracks().forEach(track => {
            pc.addTrack(track, this.localStream);
        });

        // Create and send offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        this.socket.emit('webrtc_offer', {
            targetSid: remoteSid,
            sdp: pc.localDescription,
            username: this.username,
            room: this.roomId
        });
    }

    createPeerConnection(remoteSid, remoteUsername) {
        const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });

        // Receive remote tracks
        pc.ontrack = (event) => {
            console.log(`[PeerManager] Received track from ${remoteUsername}`);
            if (this.onRemoteStream) {
                this.onRemoteStream(remoteSid, remoteUsername, event.streams[0]);
            }
        };

        // ICE candidate → signal to peer via server
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.socket.emit('webrtc_ice', {
                    targetSid: remoteSid,
                    candidate: event.candidate,
                    room: this.roomId
                });
            }
        };

        // Connection state monitoring
        pc.onconnectionstatechange = () => {
            console.log(`[PeerManager] ${remoteUsername}: ${pc.connectionState}`);
            if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                this.handlePeerDisconnect(remoteSid);
            }
        };

        // Bitrate cap: 200kbps video, 10fps
        this.applyBitrateCap(pc);

        this.peers.set(remoteSid, { pc, username: remoteUsername, stream: null });
        return pc;
    }

    applyBitrateCap(pc) {
        // Apply bitrate cap after negotiation
        pc.addEventListener('negotiationneeded', async () => {
            const sender = pc.getSenders().find(s => s.track?.kind === 'video');
            if (sender) {
                const params = sender.getParameters();
                params.encodings = params.encodings || [{}];
                params.encodings[0].maxBitrate = 200_000; // 200kbps
                params.encodings[0].maxFramerate = 10; // 10fps
                params.encodings[0].scaleResolutionDownBy = 2; // 320x240
                await sender.setParameters(params);
                console.log('[PeerManager] Applied bitrate cap: 200kbps, 10fps');
            }
        });
    }

    async handleOffer(data) {
        console.log(`[PeerManager] Received offer from ${data.fromUsername}`);

        const pc = this.createPeerConnection(data.fromSid, data.fromUsername);

        // Add local tracks
        this.localStream.getTracks().forEach(track => {
            pc.addTrack(track, this.localStream);
        });

        // Set remote description and create answer
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        this.socket.emit('webrtc_answer', {
            targetSid: data.fromSid,
            sdp: pc.localDescription,
            room: this.roomId
        });
    }

    async handleAnswer(data) {
        console.log(`[PeerManager] Received answer from ${data.fromSid}`);
        const peer = this.peers.get(data.fromSid);
        if (peer) {
            await peer.pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        }
    }

    async handleIce(data) {
        const peer = this.peers.get(data.fromSid);
        if (peer) {
            try {
                await peer.pc.addIceCandidate(new RTCIceCandidate(data.candidate));
            } catch (err) {
                console.error('[PeerManager] ICE candidate error:', err);
            }
        }
    }

    handlePeerDisconnect(sid) {
        const peer = this.peers.get(sid);
        if (peer) {
            console.log(`[PeerManager] Peer disconnected: ${peer.username}`);
            peer.pc.close();
            this.peers.delete(sid);
            if (this.onRemoveStream) {
                this.onRemoveStream(sid);
            }
        }
    }

    destroy() {
        console.log('[PeerManager] Destroying all peer connections');
        this.peers.forEach((peer, sid) => {
            peer.pc.close();
        });
        this.peers.clear();
    }
}
