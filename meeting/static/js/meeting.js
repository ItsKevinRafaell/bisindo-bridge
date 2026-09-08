/**
 * BISINDO Meeting - Client
 * WebRTC P2P + ONNX gesture recognition + Google Meet-style UI
 */

const SERVER_URL = window.location.origin;
let socket = null;
let localStream = null;
let roomId = null;
let username = null;
let peerManager = null;

// State
let micEnabled = true;
let camEnabled = true;
let handRaised = false;
let bisindoEnabled = false; // BISINDO detection off by default
let currentLayout = 'gallery'; // 'gallery' | 'speaker'
let pinnedSpeakerSid = null;
let activeSpeakerSid = null;
let meetingStartTime = null;
let timerInterval = null;
let sidePanelOpen = false;
let activeTab = 'chat';
let unreadChatCount = 0;

// Peers already present when we joined (used to fix the join race). Set by
// room_info, consumed right after PeerManager is created in joinMeeting().
let pendingRoomPeers = null;
let selfSid = null; // authoritative sid from server room_info.selfSid

// Audio analysis for active speaker
const audioContexts = new Map(); // sid → { analyser, dataArray }

// Peer state tracking (mic/cam status)
const peerStates = new Map(); // sid → { mic: bool, cam: bool, handRaised: bool }

// MediaPipe HandLandmarker
let handLandmarker = null;
let mpCanvas = null;
let mpCtx = null;
let detectionActive = false;
let videoTs = 0;

// ONNX Model
let session = null;
let scaler = null;
let labels = null;
let modelLoaded = false;

// Sentence Builder
let sentenceBuilder = null;
let lastGestureState = null;
let gestureStartTime = 0;
const GESTURE_CONFIRM_MS = 1500;

const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
];

// ============================================================
// PRE-JOIN SCREEN
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    const saved = localStorage.getItem('bisindo_username');
    if (saved) document.getElementById('username').value = saved;

    // Auto-fill room ID from URL param or path
    const urlParams = new URLSearchParams(window.location.search);
    const roomParam = urlParams.get('room') || urlParams.get('room_id');
    const pathMatch = window.location.pathname.match(/^\/room\/(.+)$/);
    const roomIdInput = document.getElementById('roomId');
    if (roomParam) {
        roomIdInput.value = roomParam;
    } else if (pathMatch && pathMatch[1]) {
        roomIdInput.value = decodeURIComponent(pathMatch[1]);
    }

    initPreview();
});

async function initPreview() {
    const previewVideo = document.getElementById('previewVideo');
    const previewNoCam = document.getElementById('previewNoCamera');
    const previewAvatar = document.getElementById('previewAvatar');
    const errorEl = document.getElementById('previewError');

    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true
        });
        previewVideo.srcObject = localStream;
        previewVideo.style.display = '';
        previewNoCam.style.display = 'none';
        errorEl.style.display = 'none';
    } catch (err) {
        console.error('Camera error:', err);

        // Specific error messages by getUserMedia error type
        let msg;
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg = '⛔ Izin ditolak. Klik ikon 🔒 di address bar → Camera + Microphone → Allow, lalu refresh halaman.';
        } else if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
            msg = '📷 Kamera/mic tidak ditemukan. Pastikan device terpasang dan tidak dipakai aplikasi lain (Zoom, Meet, OBS).';
        } else if (err.name === 'NotReadableError') {
            msg = '🔒 Kamera/mic dipakai aplikasi lain. Tutup Zoom/Meet/OBS lalu refresh.';
        } else if (err.name === 'SecurityError') {
            msg = '🔐 Halaman harus diakses via HTTPS atau localhost. Coba akses lewat tunnel Cloudflare (HTTPS).';
        } else {
            msg = `⚠️ Tidak bisa akses kamera/mic: ${err.name}. Coba refresh atau ganti browser.`;
        }

        errorEl.textContent = msg;
        errorEl.style.display = 'block';
        previewVideo.style.display = 'none';
        previewNoCam.style.display = 'flex';

        // Try audio-only
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            previewAvatar.textContent = '🎤';
            errorEl.textContent += ' (mic-only mode)';
        } catch (e2) {
            // No media at all
            localStream = new MediaStream();
        }
    }
}

function togglePreviewMic() {
    const btn = document.getElementById('previewMicBtn');
    const track = localStream?.getAudioTracks()[0];
    if (track) {
        track.enabled = !track.enabled;
        btn.classList.toggle('off', !track.enabled);
        btn.innerHTML = track.enabled
            ? '<i class="fas fa-microphone"></i>'
            : '<i class="fas fa-microphone-slash"></i>';
    }
}

function togglePreviewCam() {
    const btn = document.getElementById('previewCamBtn');
    const track = localStream?.getVideoTracks()[0];
    const previewVideo = document.getElementById('previewVideo');
    const previewNoCam = document.getElementById('previewNoCamera');

    if (track) {
        track.enabled = !track.enabled;
        btn.classList.toggle('off', !track.enabled);
        btn.innerHTML = track.enabled
            ? '<i class="fas fa-video"></i>'
            : '<i class="fas fa-video-slash"></i>';
        previewVideo.style.display = track.enabled ? '' : 'none';
        previewNoCam.style.display = track.enabled ? 'none' : 'flex';
    }
}

// ============================================================
// JOIN MEETING
// ============================================================

async function joinMeeting() {
    const nameInput = document.getElementById('username');
    username = nameInput.value.trim();
    if (!username) {
        nameInput.style.borderColor = 'var(--danger-solid)';
        nameInput.focus();
        return;
    }
    nameInput.style.borderColor = '';

    roomId = document.getElementById('roomId').value.trim() || 'room_' + Math.random().toString(36).substr(2, 6);
    localStorage.setItem('bisindo_username', username);

    // Switch screens
    document.getElementById('preJoinScreen').style.display = 'none';
    document.getElementById('meetingScreen').classList.add('active');

    // Update UI
    document.getElementById('topbarRoom').textContent = roomId;

    // Start timer
    meetingStartTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);

    // Create local video card
    createLocalVideoCard();

    // Init SocketIO
    initSocket();

    // Init WebRTC — fetch optional TURN credentials first so peers behind
    // restrictive NAT (CGNAT, campus WiFi) can still connect via relay.
    let iceServers = ICE_SERVERS;
    try {
        const turnRes = await fetch('/api/turn');
        if (!turnRes.ok) throw new Error('turn http ' + turnRes.status);
        const turnJson = await turnRes.json();
        if (turnJson && Array.isArray(turnJson.iceServers) && turnJson.iceServers.length) {
            iceServers = ICE_SERVERS.concat(turnJson.iceServers);
            console.log(`[meeting] Loaded ${turnJson.iceServers.length} TURN server(s) from /api/turn — total ICE=${iceServers.length}`);
        } else {
            console.log(`[meeting] /api/turn empty — using built-in ICE (STUN+public TURN), count=${ICE_SERVERS.length}`);
        }
    } catch (e) {
        console.warn(`[meeting] /api/turn fetch failed, using built-in ICE (STUN+public TURN) count=${ICE_SERVERS.length}:`, e);
    }

    peerManager = new PeerManager(socket, localStream, roomId, username, iceServers);
    peerManager.onRemoteStream = (sid, peerUsername, stream) => {
        showRemoteStream(sid, peerUsername, stream);
        setupAudioDetection(sid, stream);
    };
    peerManager.onRemoveStream = (sid) => {
        removeVideoCard(sid);
        cleanupAudioDetection(sid);
    };

    // Init hand tracking - await to ensure proper initialization
    await initHandLandmarker();
    initSentenceBuilder();

    // Update mic/cam state
    updateMicCamUI();

    // FIX (join race): if we joined an already-populated room, the server's
    // room_info already lists the peers present. Connect to each of them now so
    // we don't wait forever for an offer that was sent before we finished
    // joining. (connectToPeer is idempotent and only offers if our SID is lower.)
    if (Array.isArray(pendingRoomPeers)) {
        for (const u of pendingRoomPeers) {
            if (u.sid && u.sid !== selfSid && u.sid !== socket.id) {
                peerManager.connectToPeer(u.sid, u.username);
            }
        }
        pendingRoomPeers = null;
    }
}

function initSocket() {
    socket = io(SERVER_URL);

    // Guard: socket.io auto-reconnect re-fires 'connect' with new id.
    // Treat as full rejoin only if selfSid differs.
    socket.on('connect', () => {
        console.log('Connected to server, sid=', socket.id);
        // If this is a reconnect (socket.id changed vs known selfSid), purge
        // stale peer state before re-joining to avoid duplicate cards.
        if (socket.id && selfSid && socket.id !== selfSid) {
            console.log(`[meeting] Reconnect detected ${selfSid} -> ${socket.id}, cleaning stale peers`);
            if (peerManager) peerManager.handleReconnectCleanup();
            for (const key of peerStates.keys()) peerStates.delete(key);
            for (const key of audioContexts.keys()) {
                try { audioContexts.get(key).audioCtx.close(); } catch {}
                audioContexts.delete(key);
            }
            document.querySelectorAll('.video-card:not(.local)').forEach(c => c.remove());
        }
        socket.emit('join', { room: roomId, username });
        broadcastPeerState();
    });

    socket.on('disconnect', () => console.log('Disconnected'));

    socket.on('room_info', (data) => {
        roomId = data.room;
        selfSid = data.selfSid || socket.id;
        document.getElementById('topbarRoom').textContent = data.room;
        updatePeopleList(data.users);
        // Peers already in room (exclude self). Connect ASAP if PeerManager ready;
        // otherwise stash for joinMeeting() (join race fix).
        const peers = (data.users || []).filter(u => u.sid && u.sid !== selfSid && u.sid !== socket.id);
        if (peerManager) {
            for (const u of peers) {
                peerManager.connectToPeer(u.sid, u.username);
            }
            pendingRoomPeers = null;
        } else {
            pendingRoomPeers = peers;
        }
        setTimeout(retryPendingPeers, 1000);
        setTimeout(retryPendingPeers, 3000);
        setTimeout(retryPendingPeers, 6000);
    });

    socket.on('user_joined', (data) => {
        // Ignore own echo (include_self=False should prevent, but guard anyway)
        if (data.sid === selfSid || data.sid === socket.id) return;
        addChatMessage('System', `${data.username} joined`);
        if (data.users) updatePeopleList(data.users);

        if (peerManager && data.sid) {
            peerManager.connectToPeer(data.sid, data.username);
        }
    });

    socket.on('user_left', (data) => {
        // Own leave echoed? ignore
        if (data.sid === selfSid && data.username === username) {
            // but still allow UI cleanup — handled elsewhere
        }
        addChatMessage('System', `${data.username} left`);
        if (data.users) {
            updatePeopleList(data.users);
        } else {
            // Fallback: drop by sid from UI if server omitted full list
            const kept = (window.__lastPeopleList || []).filter(u => u.sid !== data.sid);
            // Also purge by username to kill ghost with new sid
            const kept2 = kept.filter(u => u.username !== data.username || u.sid === data.sid);
            // Actually simplest: re-render dropping that sid
            updatePeopleList((window.__lastPeopleList || []).filter(u => u.sid !== data.sid));
        }

        if (peerManager && data.sid) {
            peerManager.handlePeerDisconnect(data.sid);
        }
        peerStates.delete(data.sid);
        const ac = audioContexts.get(data.sid);
        if (ac) { try { ac.audioCtx.close(); } catch {} }
        audioContexts.delete(data.sid);
    });

    // BISINDO events
    socket.on('letter_committed', (data) => {
        if (data.username !== username) handleRemoteLetter(data);
    });

    // Remote peer BISINDO state (visual indicator on their card)
    socket.on('bisindo_state', (data) => {
        if (!data || !data.username || data.username === username) return;
        // Find their card by username (not SID, since state is username-keyed)
        document.querySelectorAll('.video-card').forEach(card => {
            const label = card.querySelector('.video-label span:last-child');
            if (label && label.textContent.trim() === data.username) {
                card.classList.toggle('peer-bisindo-on', !!data.enabled);
            }
        });
    });

    socket.on('space_inserted', (data) => {
        if (data.username !== username) handleRemoteSpace(data);
    });

    socket.on('sentence_broadcast', (data) => {
        if (data.username !== username) handleRemoteSentence(data);
    });

    socket.on('chat_message', (data) => {
        addChatMessage(data.username, data.message);
        // Show badge if chat panel not open
        if (activeTab !== 'chat' || !sidePanelOpen) {
            unreadChatCount++;
            updateChatBadge();
        }
    });

    // Hand raise events — render from cached list, do not rebuild with undefined
    socket.on('hand_raise', (data) => {
        const state = peerStates.get(data.sid) || {};
        state.handRaised = true;
        peerStates.set(data.sid, state);
        updateHandBadge(data.sid, true);
        if (window.__lastPeopleList) updatePeopleList(window.__lastPeopleList);
        else updatePeopleListIconsOnly();
        addChatMessage('System', `${data.username} raised hand ✋`);
    });

    socket.on('hand_lower', (data) => {
        const state = peerStates.get(data.sid) || {};
        state.handRaised = false;
        peerStates.set(data.sid, state);
        updateHandBadge(data.sid, false);
        if (window.__lastPeopleList) updatePeopleList(window.__lastPeopleList);
        else updatePeopleListIconsOnly();
    });

    // Reaction events
    socket.on('reaction', (data) => {
        showFloatingReaction(data.emoji, data.username);
    });

    // Peer state (mic/cam)
    socket.on('peer_state', (data) => {
        const state = peerStates.get(data.sid) || {};
        state.mic = data.mic;
        state.cam = data.cam;
        peerStates.set(data.sid, state);
        updatePeerMicIcon(data.sid, data.mic);
        updatePeerCamState(data.sid, data.cam);
        if (window.__lastPeopleList) updatePeopleList(window.__lastPeopleList);
        else updatePeopleListIconsOnly();
    });
}

// ============================================================
// LOCAL VIDEO CARD
// ============================================================

function createLocalVideoCard() {
    const grid = document.getElementById('videoGrid');
    const card = document.createElement('div');
    card.className = 'video-card local';
    card.id = 'card_local';
    card.innerHTML = `
        <video id="localVideo" autoplay playsinline muted></video>
        <div class="camera-off" id="localCamOff" style="display:none;">
            <div class="avatar">${username[0].toUpperCase()}</div>
        </div>
        <div class="video-label">
            <span class="mic-icon" id="localMicIcon"><i class="fas fa-microphone"></i></span>
            <span>${username} (You)</span>
        </div>
        <div class="hand-badge" id="localHandBadge">✋</div>
        <div class="prediction-badge" id="predictionOverlay"></div>
    `;
    grid.appendChild(card);

    // Set video source
    const video = document.getElementById('localVideo');
    video.srcObject = localStream;

    // Init MediaPipe canvas overlay
    video.addEventListener('loadedmetadata', () => {
        const mpCanvasEl = document.createElement('canvas');
        mpCanvasEl.id = 'mpCanvas';
        mpCanvasEl.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;';
        card.appendChild(mpCanvasEl);
        mpCanvas = mpCanvasEl;
        mpCanvas.width = video.videoWidth || 640;
        mpCanvas.height = video.videoHeight || 480;
        mpCtx = mpCanvas.getContext('2d');
    });

    updateVideoGrid();
}

// ============================================================
// REMOTE VIDEO
// ============================================================

function showRemoteStream(sid, peerUsername, stream) {
    if (peerUsername === username) return;

    const grid = document.getElementById('videoGrid');
    let card = document.getElementById('card_' + sid);

    if (!card) {
        card = document.createElement('div');
        card.className = 'video-card';
        card.id = 'card_' + sid;
        card.innerHTML = `
            <video autoplay playsinline muted></video>
            <div class="camera-off" style="display:none;">
                <div class="avatar">${(peerUsername || '?')[0].toUpperCase()}</div>
            </div>
            <div class="video-label">
                <span class="mic-icon"><i class="fas fa-microphone"></i></span>
                <span>${peerUsername}</span>
            </div>
            <div class="hand-badge">✋</div>
            <div class="quality-indicator good">
                <div class="quality-bar"></div>
                <div class="quality-bar"></div>
                <div class="quality-bar"></div>
            </div>
        `;
        card.addEventListener('dblclick', () => pinSpeaker(sid));
        // Click to unmute audio — required by Chrome autoplay policy
        card.addEventListener('click', () => {
            const v = card.querySelector('video');
            if (v && v.muted) {
                v.muted = false;
                v.play().catch(()=>{});
                console.log('[meeting] unmuted video for', peerUsername);
            }
        });
        grid.appendChild(card);
    }

    const video = card.querySelector('video');
    if (video) {
        const tryPlay = () => {
            if (video.paused || video.readyState < 2) {
                const pr = video.play();
                if (pr && typeof pr.catch === 'function') {
                    pr.catch(() => {
                        video.muted = true;
                        video.play().catch(()=>{});
                    });
                }
            }
        };
        if (video._tryPlay) {
            try { video.removeEventListener('loadedmetadata', video._tryPlay); } catch(_){}
        }
        video._tryPlay = tryPlay;
        video.addEventListener('loadedmetadata', tryPlay, { once: true });
        video.addEventListener('loadeddata', tryPlay, { once: true });

        // Always reassign srcObject to trigger reload
        try { video.srcObject = null; } catch(_){}
        video.srcObject = stream;
        tryPlay();

        // Unmute after user gesture window (join click counts as gesture)
        // Keep trying for 5 seconds — TURN relay may arrive late
        let unmuteAttempts = 0;
        const tryUnmute = () => {
            if (video.muted && stream.getAudioTracks().length > 0 && video.readyState >= 2) {
                video.muted = false;
                video.play().then(()=>{
                    console.log('[meeting] unmuted audio for', peerUsername, 'attempt', unmuteAttempts);
                }).catch(()=>{
                    video.muted = true;
                    if (unmuteAttempts < 5) {
                        unmuteAttempts++;
                        setTimeout(tryUnmute, 500);
                    }
                });
            }
        };
        setTimeout(tryUnmute, 300);
        setTimeout(tryUnmute, 1000);
        setTimeout(tryUnmute, 2000);

        // Handle later added tracks
        if (stream) {
            const prev = stream._origOnAddTrack || null;
            stream._origOnAddTrack = prev;
            stream.onaddtrack = (ev) => {
                if (typeof prev === 'function') prev(ev);
                if (!ev || !ev.track) return;
                console.log('[meeting] onaddtrack', ev.track.kind, 'for', peerUsername);
                if (ev.track.kind === 'video' && video.paused) tryPlay();
                if (ev.track.kind === 'audio') {
                    setTimeout(tryUnmute, 300);
                }
            };
        }
    }

    // Track peer state
    peerStates.set(sid, { mic: true, cam: true, handRaised: false, username: peerUsername });
    updateVideoGrid();
}

function removeVideoCard(sid) {
    const card = document.getElementById('card_' + sid);
    if (card) {
        card.remove();
        updateVideoGrid();
    }
    if (pinnedSpeakerSid === sid) {
        pinnedSpeakerSid = null;
    }
}

// ============================================================
// ACTIVE SPEAKER DETECTION
// ============================================================

function setupAudioDetection(sid, stream) {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        audioContexts.set(sid, { audioCtx, analyser, dataArray });
        checkAudioLevel(sid);
    } catch (err) {
        console.warn('Audio detection setup failed for', sid, err);
    }
}

function checkAudioLevel(sid) {
    const ctx = audioContexts.get(sid);
    if (!ctx) return;

    const { analyser, dataArray } = ctx;
    analyser.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    const isActive = avg > 25;

    if (isActive && sid !== activeSpeakerSid) {
        activeSpeakerSid = sid;
        updateActiveSpeakerUI();
    }

    // Also check local mic
    if (isActive && sid === 'local') {
        activeSpeakerSid = 'local';
        updateActiveSpeakerUI();
    }

    setTimeout(() => checkAudioLevel(sid), 200);
}

function updateActiveSpeakerUI() {
    // Remove active-speaker from all cards
    document.querySelectorAll('.video-card.active-speaker').forEach(card => {
        card.classList.remove('active-speaker');
    });

    // Add to active speaker
    if (activeSpeakerSid) {
        const card = document.getElementById('card_' + activeSpeakerSid);
        if (card) {
            card.classList.add('active-speaker');
        }
    }

    // In speaker view, auto-switch to active speaker if not pinned
    if (currentLayout === 'speaker' && !pinnedSpeakerSid && activeSpeakerSid) {
        updateSpeakerView();
    }
}

function cleanupAudioDetection(sid) {
    const ctx = audioContexts.get(sid);
    if (ctx) {
        ctx.audioCtx.close();
        audioContexts.delete(sid);
    }
    if (activeSpeakerSid === sid) {
        activeSpeakerSid = null;
        updateActiveSpeakerUI();
    }
}

// ============================================================
// VIDEO GRID
// ============================================================

function updateVideoGrid() {
    const grid = document.getElementById('videoGrid');
    const cards = grid.querySelectorAll('.video-card');
    const count = cards.length;

    // Remove old column classes
    for (let i = 1; i <= 9; i++) {
        grid.classList.remove('cols-' + i);
    }

    if (count <= 1) grid.classList.add('cols-1');
    else if (count === 2) grid.classList.add('cols-2');
    else if (count === 3) grid.classList.add('cols-3');
    else if (count <= 4) grid.classList.add('cols-4');
    else if (count <= 6) grid.classList.add('cols-3');
    else grid.classList.add('cols-3');

    document.getElementById('participantCount').textContent = count;
}

// ============================================================
// SPEAKER VIEW / GALLERY TOGGLE
// ============================================================

function toggleLayout() {
    if (currentLayout === 'gallery') {
        currentLayout = 'speaker';
    } else {
        currentLayout = 'gallery';
    }
    applyLayout();
}

function applyLayout() {
    const videoArea = document.getElementById('videoArea');
    const grid = document.getElementById('videoGrid');

    if (currentLayout === 'speaker') {
        videoArea.classList.remove('gallery-view');
        videoArea.classList.add('speaker-view');
        updateSpeakerView();
    } else {
        videoArea.classList.remove('speaker-view');
        videoArea.classList.add('gallery-view');
        // Remove speaker classes from all cards
        grid.querySelectorAll('.video-card').forEach(card => {
            card.classList.remove('speaker-main', 'speaker-thumb');
        });
    }
}

function updateSpeakerView() {
    const grid = document.getElementById('videoGrid');
    const cards = Array.from(grid.querySelectorAll('.video-card'));
    if (cards.length === 0) return;

    // Determine main speaker
    let mainCard;
    if (pinnedSpeakerSid) {
        mainCard = document.getElementById('card_' + pinnedSpeakerSid);
    } else if (activeSpeakerSid) {
        mainCard = document.getElementById('card_' + activeSpeakerSid);
    }
    if (!mainCard) mainCard = cards[0];

    // Apply CSS classes - pure CSS grid handles the layout
    cards.forEach(card => {
        if (card === mainCard) {
            card.classList.add('speaker-main');
            card.classList.remove('speaker-thumb');
        } else {
            card.classList.add('speaker-thumb');
            card.classList.remove('speaker-main');
        }
    });
}

function pinSpeaker(sid) {
    pinnedSpeakerSid = (pinnedSpeakerSid === sid) ? null : sid;
    if (currentLayout === 'speaker') {
        updateSpeakerView();
    }
}

// ============================================================
// CONTROLS
// ============================================================

function toggleMic() {
    const track = localStream?.getAudioTracks()[0];
    if (track) {
        track.enabled = !track.enabled;
        micEnabled = track.enabled;
        updateMicCamUI();
        broadcastPeerState();
    }
}

function toggleCamera() {
    const track = localStream?.getVideoTracks()[0];
    if (track) {
        track.enabled = !track.enabled;
        camEnabled = track.enabled;
        updateMicCamUI();
        broadcastPeerState();

        // Toggle detection — only if BISINDO is explicitly enabled
        if (!camEnabled) {
            detectionActive = false;
        } else if (bisindoEnabled) {
            detectionActive = true;
            runDetection();
        } else {
            // BISINDO is off, don't start detection
            detectionActive = false;
        }
    }
}

function updateMicCamUI() {
    const micBtn = document.getElementById('micBtn');
    const camBtn = document.getElementById('camBtn');

    // Mic button
    micBtn.classList.toggle('off', !micEnabled);
    micBtn.querySelector('i').className = micEnabled ? 'fas fa-microphone' : 'fas fa-microphone-slash';

    // Cam button
    camBtn.classList.toggle('off', !camEnabled);
    camBtn.querySelector('i').className = camEnabled ? 'fas fa-video' : 'fas fa-video-slash';

    // Local video label mic icon
    const localMicIcon = document.getElementById('localMicIcon');
    if (localMicIcon) {
        localMicIcon.classList.toggle('muted', !micEnabled);
        localMicIcon.innerHTML = micEnabled
            ? '<i class="fas fa-microphone"></i>'
            : '<i class="fas fa-microphone-slash"></i>';
    }

    // Camera off state
    const localVideo = document.getElementById('localVideo');
    const localCamOff = document.getElementById('localCamOff');
    if (localVideo && localCamOff) {
        localVideo.style.display = camEnabled ? '' : 'none';
        localCamOff.style.display = camEnabled ? 'none' : 'flex';
    }
}

function broadcastPeerState() {
    if (socket && socket.connected) {
        socket.emit('peer_state', {
            room: roomId,
            username: username,
            mic: micEnabled,
            cam: camEnabled
        });
    }
}

function updatePeerMicIcon(sid, micOn) {
    const card = document.getElementById('card_' + sid);
    if (!card) return;
    const micIcon = card.querySelector('.mic-icon');
    if (micIcon) {
        micIcon.classList.toggle('muted', !micOn);
        micIcon.innerHTML = micOn
            ? '<i class="fas fa-microphone"></i>'
            : '<i class="fas fa-microphone-slash"></i>';
    }
}

function updatePeerCamState(sid, camOn) {
    const card = document.getElementById('card_' + sid);
    if (!card) return;
    const video = card.querySelector('video');
    const camOff = card.querySelector('.camera-off');
    if (video) video.style.display = camOn ? '' : 'none';
    if (camOff) camOff.style.display = camOn ? 'none' : 'flex';
}

// ============================================================
// HAND RAISE
// ============================================================

function toggleHandRaise() {
    handRaised = !handRaised;
    const btn = document.getElementById('handBtn');
    btn.classList.toggle('off', handRaised);

    // Update local badge
    const localBadge = document.getElementById('localHandBadge');
    if (localBadge) {
        localBadge.classList.toggle('visible', handRaised);
    }

    if (socket && socket.connected) {
        socket.emit(handRaised ? 'hand_raise' : 'hand_lower', {
            room: roomId,
            username: username
        });
    }

    // Auto-lower after 30s
    if (handRaised) {
        setTimeout(() => {
            if (handRaised) {
                handRaised = false;
                btn.classList.remove('off');
                if (localBadge) localBadge.classList.remove('visible');
                socket.emit('hand_lower', { room: roomId, username });
            }
        }, 30000);
    }
}

function updateHandBadge(sid, raised) {
    const card = document.getElementById('card_' + sid);
    if (!card) return;
    const badge = card.querySelector('.hand-badge');
    if (badge) {
        badge.classList.toggle('visible', raised);
    }
}

// ============================================================
// REACTIONS
// ============================================================

function toggleReactions(event) {
    const popup = document.getElementById('reactionsPopup');
    popup.classList.toggle('open');

    // Position near button
    if (popup.classList.contains('open')) {
        const btn = document.getElementById('reactBtn');
        const rect = btn.getBoundingClientRect();
        popup.style.left = rect.left + 'px';
    }
}

function sendReaction(emoji) {
    // Keep popup open for spam

    // Show locally
    showFloatingReaction(emoji, username);

    // Broadcast
    if (socket && socket.connected) {
        socket.emit('reaction', {
            room: roomId,
            username: username,
            emoji: emoji
        });
    }
}

function showFloatingReaction(emoji, fromUser) {
    const container = document.getElementById('floatingReactions');
    const el = document.createElement('div');
    el.className = 'floating-emoji';
    el.textContent = emoji;
    el.style.left = (Math.random() * 100 - 50) + 'px';
    container.appendChild(el);

    // Remove after animation
    setTimeout(() => el.remove(), 3000);
}

// Close reactions popup when clicking outside
document.addEventListener('click', (e) => {
    const popup = document.getElementById('reactionsPopup');
    const btn = document.getElementById('reactBtn');
    if (popup && !popup.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
        popup.classList.remove('open');
    }
});

// ============================================================
// SIDE PANEL
// ============================================================

function togglePanel(tab) {
    if (sidePanelOpen && activeTab === tab) {
        closePanel();
    } else {
        openPanel(tab);
    }
}

function openPanel(tab) {
    sidePanelOpen = true;
    activeTab = tab;
    document.getElementById('sidePanel').classList.add('open');
    switchTab(tab);

    // Clear chat badge
    if (tab === 'chat') {
        unreadChatCount = 0;
        updateChatBadge();
    }
}

function closePanel() {
    sidePanelOpen = false;
    document.getElementById('sidePanel').classList.remove('open');
}

function switchTab(tab) {
    activeTab = tab;

    // Update tab buttons
    document.querySelectorAll('.panel-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Update tab content
    document.querySelectorAll('.panel-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const tabContent = document.getElementById(tab + 'Tab');
    if (tabContent) tabContent.classList.add('active');

    // Update panel title
    const titles = { chat: 'Chat', people: 'Participants', bisindo: 'BISINDO' };
    document.getElementById('panelTitle').textContent = titles[tab] || tab;

    // Clear chat badge
    if (tab === 'chat') {
        unreadChatCount = 0;
        updateChatBadge();
    }
}

function updateChatBadge() {
    const badge = document.getElementById('chatBadge');
    if (badge) {
        badge.textContent = unreadChatCount;
        badge.classList.toggle('hidden', unreadChatCount === 0);
    }
}

// ============================================================
// CHAT
// ============================================================

function addChatMessage(sender, msg) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');

    if (sender === 'System') {
        div.className = 'chat-msg system';
        div.textContent = msg;
    } else {
        const isOwn = sender === username;
        div.className = `chat-msg ${isOwn ? 'own' : 'other'}`;
        div.innerHTML = `<div class="chat-sender">${sender}</div><div>${msg}</div>`;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg || !socket || !socket.connected) return;
    socket.emit('text_message', { room: roomId, username, message: msg, timestamp: new Date().toISOString() });
    // Don't add locally - server broadcasts to everyone including sender
    input.value = '';
}

// ============================================================
// PEOPLE LIST
// ============================================================


    // Retry connect for any peer not yet in peerConnections (fix rata-rata video ga masuk)
    function retryPendingPeers() {
        if (!peerManager) return;
        const allCards = document.querySelectorAll('.video-card:not(.local)');
        // Check last people list
        const list = window.__lastPeopleList || pendingRoomPeers || [];
        for (const u of list) {
            if (!u || !u.sid) continue;
            if (u.sid === selfSid || u.sid === socket.id) continue;
            if (!peerManager.peerConnections.has(u.sid)) {
                console.log('[meeting] retry connect to', u.username, u.sid.slice(0,6));
                peerManager.connectToPeer(u.sid, u.username);
            }
        }
        // Also if offer was missed, force reconnect for existing PCs that are still new/connecting after 5s
        for (const [sid, pc] of peerManager.peerConnections.entries()) {
            const st = pc.connectionState;
            if (st === 'new' || st === 'connecting') {
                // If we are lower sid we should have waited, but try force offer after timeout
                if (pc.iceConnectionState === 'new' || pc.iceConnectionState === 'checking') {
                    // let it continue, but log
                    console.log(`[meeting] still ${st}/${pc.iceConnectionState} for ${sid.slice(0,6)}`);
                }
            }
        }
    }
    // Schedule retries 1s, 3s, 6s after room_info

function updatePeopleList(users) {
    const container = document.getElementById('peopleList');
    if (!container) return;

    const seen = new Set();
    const allUsers = [];
    for (const u of (users || [])) {
        if (!u || !u.sid) continue;
        if (seen.has(u.sid)) continue;
        seen.add(u.sid);
        allUsers.push(u);
    }
    window.__lastPeopleList = allUsers;
    container.innerHTML = '';

    allUsers.forEach(u => {
        const sid = u.sid;
        const state = peerStates.get(sid) || { mic: true, cam: true, handRaised: false };
        const isLocal = sid === selfSid || sid === (socket && socket.id);

        const div = document.createElement('div');
        div.className = 'person-item';
        const safeName = String(u.username || '?').replace(/[<>&"]/g, '');
        div.innerHTML = `
            <div class="person-avatar">${(safeName || '?')[0].toUpperCase()}</div>
            <div class="person-info">
                <div class="person-name">${safeName}${isLocal ? ' (You)' : ''}</div>
                <div class="person-status">
                    <i class="fas fa-microphone ${state.mic === false ? 'muted' : ''}"></i>
                    <i class="fas fa-video ${state.cam === false ? 'muted' : ''}"></i>
                    ${state.handRaised ? '<i class="fas fa-hand-paper raised"></i>' : ''}
                </div>
            </div>
        `;
        container.appendChild(div);
    });

    const countEl = document.getElementById('participantCount');
    if (countEl) countEl.textContent = allUsers.length;
}

function updatePeopleListIconsOnly() {
    // Refresh icons without touching roster (used when only mic/cam/hand changed)
    const allUsers = window.__lastPeopleList || [];
    const container = document.getElementById('peopleList');
    if (!container) return;
    allUsers.forEach(u => {
        const state = peerStates.get(u.sid);
        if (!state) return;
        // Update via DOM walk would be more efficient, but re-render small list
    });
    // Delegate to full render using cached list
    updatePeopleList(allUsers);
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', (e) => {
    // Skip if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    // Skip if not in meeting
    if (!document.getElementById('meetingScreen').classList.contains('active')) return;

    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 'd': e.preventDefault(); toggleMic(); break;
            case 'e': e.preventDefault(); toggleCamera(); break;
            case 'w': e.preventDefault(); exitMeeting(); break;
        }
    }

    switch (e.key.toLowerCase()) {
        case 'f': toggleFullscreen(); break;
        case 'h': toggleHandRaise(); break;
        case 'c': togglePanel('chat'); break;
        case 'l': toggleLayout(); break;
    }
});

// ============================================================
// FULLSCREEN
// ============================================================

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => console.warn(err));
        document.getElementById('fullscreenBtn').querySelector('i').className = 'fas fa-compress';
    } else {
        document.exitFullscreen();
        document.getElementById('fullscreenBtn').querySelector('i').className = 'fas fa-expand';
    }
}

document.addEventListener('fullscreenchange', () => {
    const btn = document.getElementById('fullscreenBtn');
    if (btn) {
        btn.querySelector('i').className = document.fullscreenElement ? 'fas fa-compress' : 'fas fa-expand';
    }
});

// ============================================================
// TIMER
// ============================================================

function updateTimer() {
    if (!meetingStartTime) return;
    const elapsed = Math.floor((Date.now() - meetingStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    const timerEl = document.getElementById('meetingTimer');
    if (timerEl) timerEl.textContent = `${mins}:${secs}`;
}

// ============================================================
// EXIT
// ============================================================

function exitMeeting() {
    // Show custom exit modal instead of browser confirm()
    const modal = document.getElementById('exitModal');
    if (modal) modal.style.display = 'flex';
}

function closeExitModal() {
    const modal = document.getElementById('exitModal');
    if (modal) modal.style.display = 'none';
}

function confirmExit() {
    // Hide modal
    const modal = document.getElementById('exitModal');
    if (modal) modal.style.display = 'none';

    // Stop all detection and processing
    detectionActive = false;
    bisindoEnabled = false;

    // Stop MediaPipe hand landmarker
    if (handLandmarker) {
        handLandmarker.close();
        handLandmarker = null;
    }

    // Stop timer
    if (timerInterval) clearInterval(timerInterval);

    // Disconnect main socket
    if (socket) {
        socket.emit('leave', { room: roomId, username });
        socket.disconnect();
        socket = null;
    }

    // Destroy SFU peer manager (this closes all WebRTC connections)
    if (peerManager) {
        peerManager.destroy();
        peerManager = null;
    }

    // Stop all media tracks (camera + mic)
    if (localStream) {
        localStream.getTracks().forEach(track => {
            track.stop();
            track.enabled = false;
        });
        localStream = null;
    }

    // Clear local video element
    const localVideo = document.getElementById('localVideo');
    if (localVideo) {
        localVideo.srcObject = null;
    }

    // Cleanup audio contexts
    audioContexts.forEach((ctx) => ctx.audioCtx.close());
    audioContexts.clear();

    // Reset state
    peerStates.clear();
    activeSpeakerSid = null;
    pinnedSpeakerSid = null;
    handRaised = false;

    // Switch screens
    document.getElementById('meetingScreen').classList.remove('active');
    document.getElementById('preJoinScreen').style.display = 'flex';
}

// ============================================================
// COPY ROOM LINK
// ============================================================

function copyRoomLink() {
    const link = `${window.location.origin}/room/${roomId}`;
    navigator.clipboard.writeText(link).then(() => {
        const el = document.getElementById('topbarRoom');
        const orig = el.textContent;
        el.textContent = 'Copied!';
        setTimeout(() => el.textContent = orig, 1500);
    });
}

// ============================================================
// ONNX MODEL
// ============================================================

async function loadModel() {
    const statusEl = document.getElementById('modelStatus');
    if (statusEl) statusEl.textContent = 'Loading model...';

    try {
        session = await ort.InferenceSession.create('/static/models/model.onnx');
        const [scalerRes, labelsRes] = await Promise.all([
            fetch('/static/models/scaler.json').then(r => r.json()),
            fetch('/static/models/labels.json').then(r => r.json()),
        ]);
        scaler = scalerRes;
        labels = labelsRes;
        modelLoaded = true;
        console.log(`ONNX model loaded: ${scaler.mean.length} features, ${labels.length} classes`);
        if (statusEl) statusEl.textContent = 'Model ready';
    } catch (err) {
        console.error('Model load failed:', err);
        if (statusEl) statusEl.textContent = 'Model failed: ' + (err.message || err);
    }
}

function landmarksToFeatures(results) {
    if (!results.landmarks || results.landmarks.length === 0) return null;
    const hand1 = results.landmarks[0];
    const hand1Flat = [];
    hand1.forEach(p => hand1Flat.push(p.x, p.y));
    while (hand1Flat.length < 42) hand1Flat.push(0);

    let hand2Flat = new Array(42).fill(0);
    if (results.landmarks.length > 1) {
        const hand2 = results.landmarks[1];
        hand2Flat = [];
        hand2.forEach(p => hand2Flat.push(p.x, p.y));
        while (hand2Flat.length < 42) hand2Flat.push(0);
    }
    return hand1Flat.concat(hand2Flat);
}

function normalizeFeaturesHandCentric(raw) {
    function normalizeHand(arr) {
        const wx = arr[0], wy = arr[1];
        const centered = [];
        for (let i = 0; i < arr.length; i += 2) {
            centered.push(arr[i] - wx, arr[i+1] - wy);
        }
        let maxD = 0;
        for (let i = 0; i < centered.length; i += 2) {
            const d = Math.sqrt(centered[i]**2 + centered[i+1]**2);
            if (d > maxD) maxD = d;
        }
        if (maxD > 0) for (let i = 0; i < centered.length; i++) centered[i] /= maxD;
        return centered;
    }
    return normalizeHand(raw.slice(0, 42)).concat(normalizeHand(raw.slice(42, 84)));
}

function scaleFeatures(features) {
    const out = new Array(features.length);
    for (let i = 0; i < features.length; i++) {
        out[i] = (features[i] - scaler.mean[i]) / scaler.scale[i];
    }
    return out;
}

async function predictModel(features) {
    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 1, features.length]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const outputKey = Object.keys(results)[0];
    const probs = results[outputKey].data;
    return Array.from(probs);
}

// ============================================================
// GESTURE DETECTION
// ============================================================

function detectGesture(landmarks, numHands = 1) {
    if (!landmarks || landmarks.length < 21) return null;
    if (numHands >= 2) return 'signing';
    const wrist = landmarks[0];
    const fingertips = [4, 8, 12, 16, 20];

    let maxDist = 0;
    for (let i = 0; i < 21; i++) {
        for (let j = i + 1; j < 21; j++) {
            const dx = landmarks[i].x - landmarks[j].x;
            const dy = landmarks[i].y - landmarks[j].y;
            const dz = landmarks[i].z - landmarks[j].z;
            const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (d > maxDist) maxDist = d;
        }
    }
    if (maxDist < 1e-6) return 'signing';

    const tipDists = fingertips.map(i => {
        const dx = landmarks[i].x - wrist.x;
        const dy = landmarks[i].y - wrist.y;
        const dz = landmarks[i].z - wrist.z;
        return Math.sqrt(dx*dx + dy*dy + dz*dz) / maxDist;
    });

    const meanTipDist = tipDists.reduce((a, b) => a + b, 0) / tipDists.length;
    if (meanTipDist < 0.3) return 'fist';

    if (tipDists.every(d => d > 0.5)) {
        const tipLandmarks = fingertips.map(i => landmarks[i]);
        let spreadSum = 0, spreadCount = 0;
        for (let i = 0; i < tipLandmarks.length; i++) {
            for (let j = i + 1; j < tipLandmarks.length; j++) {
                const dx = tipLandmarks[i].x - tipLandmarks[j].x;
                const dy = tipLandmarks[i].y - tipLandmarks[j].y;
                spreadSum += Math.sqrt(dx*dx + dy*dy);
                spreadCount++;
            }
        }
        if (spreadSum / spreadCount > 0.1) return 'palm';
    }

    return 'signing';
}

// ============================================================
// MEDIAPIPE HAND LANDMARKER
// ============================================================

async function initHandLandmarker() {
    try {
        const { HandLandmarker, FilesetResolver } = await import(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs"
        );

        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: 2
        });

        console.log('✅ MediaPipe HandLandmarker initialized');
    } catch (err) {
        console.error('HandLandmarker init failed:', err);
    }
}

function toggleBisindo() {
    bisindoEnabled = !bisindoEnabled;
    const btn = document.getElementById('bisindoBtn');
    btn.classList.toggle('off', !bisindoEnabled);
    document.getElementById('card_local')?.classList.toggle('bisindo-on', bisindoEnabled);

    // Broadcast our BISINDO state to room
    socket.emit('bisindo_state', { enabled: bisindoEnabled, username, room: roomId });

    if (bisindoEnabled) {
        detectionActive = true;
        runDetection();
        console.log('BISINDO detection enabled');
    } else {
        detectionActive = false;
        console.log('BISINDO detection disabled');
    }
}

// === BISINDO confidence display (set after each prediction) ===
let lastPrediction = { letter: null, confidence: 0.0, ts: 0 };

function updateConfidenceDisplay(letter, confidence) {
    const confEl = document.getElementById('bisindoConfidence');
    if (!confEl) return;
    if (!letter) {
        confEl.textContent = '—';
        confEl.className = '';
        return;
    }
    const pct = Math.round(confidence * 100);
    confEl.textContent = `${letter} · ${pct}%`;
    confEl.className = confidence >= 0.7 ? 'high' : confidence >= 0.5 ? 'mid' : 'low';
}

// === TTS per-peer opt-in (default OFF to avoid audio spam) ===
let bisindoTTS = false;
function toggleBisindoTTS() {
    bisindoTTS = !bisindoTTS;
    const btn = document.getElementById('bisindoTtsBtn');
    btn?.classList.toggle('on', bisindoTTS);
}

let detectionLoopRunning = false;

// ── EVAL INSTRUMENTATION (browser E2E FPS / latency) ──────────────
// Trigger from DevTools console:  startFpsBench()   (collects 100 frames)
// Results auto-print mean/median/P95 for per-frame time and derived FPS.
window.__fpsBench = { active: false, frames: [], target: 100 };
window.startFpsBench = function (n = 100) {
    window.__fpsBench = { active: true, frames: [], target: n };
    console.log(`[fps-bench] collecting ${n} frames... keep a hand in view`);
};
function __fpsReport(f) {
    const s = [...f].sort((a, b) => a - b);
    const mean = f.reduce((a, b) => a + b, 0) / f.length;
    const med = s[Math.floor(s.length / 2)];
    const p95 = s[Math.floor(s.length * 0.95)];
    const fps = (t) => (1000 / t).toFixed(1);
    console.log('[fps-bench] ==== END-TO-END PIPELINE (detect + ONNX + UI) ====');
    console.log(`[fps-bench] frames=${f.length}`);
    console.log(`[fps-bench] per-frame ms  mean=${mean.toFixed(2)}  median=${med.toFixed(2)}  P95=${p95.toFixed(2)}`);
    console.log(`[fps-bench] FPS           mean=${fps(mean)}  median=${fps(med)}  P95(worst)=${fps(p95)}`);
    console.log('[fps-bench] COPY THE THREE LINES ABOVE ^');
}

function runDetection() {
    if (!detectionActive || !handLandmarker) {
        detectionLoopRunning = false;
        return;
    }

    detectionLoopRunning = true;
    const localVideo = document.getElementById('localVideo');
    if (localVideo && localVideo.readyState >= 2) {
        videoTs += 33;
        const __b = window.__fpsBench;
        const __t0 = __b.active ? performance.now() : 0;
        const results = handLandmarker.detectForVideo(localVideo, videoTs);
        processResults(results);
        if (__b.active) {
            __b.frames.push(performance.now() - __t0);
            if (__b.frames.length >= __b.target) {
                __b.active = false;
                __fpsReport(__b.frames);
            }
        }
    }

    // Only schedule next frame if still active
    if (detectionActive) {
        requestAnimationFrame(runDetection);
    } else {
        detectionLoopRunning = false;
    }
}

function processResults(results) {
    if (!mpCtx) return;
    mpCtx.clearRect(0, 0, mpCanvas.width, mpCanvas.height);

    const handDetected = results.landmarks && results.landmarks.length > 0;
    let gestureState = null;
    let prediction = null;

    if (handDetected) {
        results.landmarks.forEach((lms, handIdx) => drawHand(lms, handIdx));

        const rawGesture = detectGesture(results.landmarks[0], results.landmarks.length);
        const now = Date.now();
        if (rawGesture !== lastGestureState) {
            lastGestureState = rawGesture;
            gestureStartTime = now;
        }
        if ((rawGesture === 'fist' || rawGesture === 'palm') &&
            (now - gestureStartTime >= GESTURE_CONFIRM_MS)) {
            gestureState = rawGesture;
        } else {
            gestureState = 'signing';
        }

        const isGesture = rawGesture === 'fist' || rawGesture === 'palm';
        const shouldSkipPrediction = isGesture && results.landmarks.length === 1;
        const gestureForBuilder = shouldSkipPrediction ? rawGesture : gestureState;

        if (modelLoaded && !shouldSkipPrediction) {
            const rawFeats = landmarksToFeatures(results);
            if (rawFeats) {
                const normalized = normalizeFeaturesHandCentric(rawFeats);
                const scaled = scaleFeatures(normalized);
                predictModel(scaled).then(probs => {
                    const indexed = probs.map((p, i) => ({ letter: labels[i], p }));
                    indexed.sort((a, b) => b.p - a.p);
                    const top = indexed[0];

                    // Update prediction overlay
                    const overlay = document.getElementById('predictionOverlay');
                    if (overlay) {
                        overlay.textContent = top.p > 0.5 ? top.letter : '?';
                        overlay.style.display = top.p > 0.5 ? 'block' : 'none';
                    }

                    // Update BISINDO panel prediction
                    const predLetter = document.getElementById('predLetter');
                    const confFill = document.getElementById('confidenceFill');
                    if (predLetter) predLetter.textContent = top.p > 0.5 ? top.letter : '-';
                    if (confFill) confFill.style.width = (top.p * 100) + '%';

                    // Update sidebar confidence display (compact)
                    updateConfidenceDisplay(top.p > 0.5 ? top.letter : null, top.p);

                    if (top.p > 0.5) {
                        prediction = { letter: top.letter, confidence: top.p };
                    }

                    if (sentenceBuilder) {
                        sentenceBuilder.feed(prediction, handDetected, gestureForBuilder);
                    }
                });
            }
        } else if (sentenceBuilder && shouldSkipPrediction) {
            sentenceBuilder.feed(null, handDetected, gestureForBuilder);
        }

        // Draw gesture indicator on canvas
        if (gestureState === 'fist') {
            mpCtx.fillStyle = 'rgba(239, 68, 68, 0.7)';
            mpCtx.fillRect(mpCanvas.width - 120, mpCanvas.height - 40, 110, 30);
            mpCtx.fillStyle = '#fff';
            mpCtx.font = '16px Poppins';
            mpCtx.fillText('✊ BACKSPACE', mpCanvas.width - 110, mpCanvas.height - 18);
        } else if (gestureState === 'palm') {
            mpCtx.fillStyle = 'rgba(59, 130, 246, 0.7)';
            mpCtx.fillRect(mpCanvas.width - 120, mpCanvas.height - 40, 110, 30);
            mpCtx.fillStyle = '#fff';
            mpCtx.font = '16px Poppins';
            mpCtx.fillText('🖐 DELETE', mpCanvas.width - 110, mpCanvas.height - 18);
        }
    } else {
        lastGestureState = null;
        if (sentenceBuilder) {
            sentenceBuilder.feed(null, false, null);
        }
    }
}

function drawHand(landmarks, handIdx) {
    const w = mpCanvas.width, h = mpCanvas.height;
    const color = handIdx === 0 ? '#00FF88' : '#FFB800';
    mpCtx.strokeStyle = color;
    mpCtx.lineWidth = 3;
    // Mirror x coordinates to match CSS scaleX(-1) on video
    const mx = (x) => (1 - x) * w;
    for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = landmarks[s], p2 = landmarks[e];
        if (!p1 || !p2) continue;
        mpCtx.beginPath();
        mpCtx.moveTo(mx(p1.x), p1.y * h);
        mpCtx.lineTo(mx(p2.x), p2.y * h);
        mpCtx.stroke();
    }
    mpCtx.fillStyle = handIdx === 0 ? '#FF4444' : '#FF8800';
    for (const p of landmarks) {
        mpCtx.beginPath();
        mpCtx.arc(mx(p.x), p.y * h, 5, 0, Math.PI * 2);
        mpCtx.fill();
    }
}

// ============================================================
// SENTENCE BUILDER
// ============================================================

function initSentenceBuilder() {
    sentenceBuilder = new SentenceBuilder({
        gestureHoldMs: 2000,
        stabilityMs: 500,
        spaceNoHandMs: 2000,
        enterNoHandMs: 4000,
        onLetterCommit: (letter, word) => {
            const el = document.getElementById('buildingWord');
            if (el) el.textContent = word;
            if (socket && socket.connected) {
                socket.emit('letter_committed', { room: roomId, username, letter, t_sent: Date.now() });
            }
        },
        onWordComplete: (word, allWords) => {
            const chip = document.createElement('span');
            chip.className = 'word-chip';
            chip.textContent = word;
            const chipsEl = document.getElementById('wordChips');
            if (chipsEl) chipsEl.appendChild(chip);
            const el = document.getElementById('buildingWord');
            if (el) el.textContent = '';
            if (socket && socket.connected) {
                socket.emit('space_inserted', { room: roomId, username });
            }
        },
        onSentenceComplete: (sentence, allSentences) => {
            const div = document.createElement('div');
            div.className = 'sentence-item own';
            div.textContent = sentence;
            const historyEl = document.getElementById('sentenceHistory');
            if (historyEl) historyEl.prepend(div);
            const chipsEl = document.getElementById('wordChips');
            if (chipsEl) chipsEl.innerHTML = '';
            if (socket && socket.connected) {
                socket.emit('sentence_completed', { room: roomId, username, sentence });
            }
        },
        onGesture: (gesture) => {
            console.log('Gesture:', gesture);
        }
    });

    loadModel();
}

function manualSpace() {
    if (sentenceBuilder) sentenceBuilder.manualSpace();
}
function manualEnter() {
    if (sentenceBuilder) sentenceBuilder.manualEnter();
}
function clearSentence() {
    if (sentenceBuilder) sentenceBuilder.clearAll();
    const el = document.getElementById('buildingWord');
    if (el) el.textContent = '';
    const chips = document.getElementById('wordChips');
    if (chips) chips.innerHTML = '';
}
function speakSentence() {
    if (!sentenceBuilder) return;
    const state = sentenceBuilder.getState();
    const text = state.sentences[state.sentences.length - 1] || state.buildingSentence;
    if (!text) return;
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'id-ID';
        utterance.rate = 0.9;
        speechSynthesis.speak(utterance);
    }
}

// Remote sentence handlers
// EVAL: broadcast latency. Requires sender & receiver clocks roughly synced
// (same LAN / NTP). window.__bcastLat collects (recv - t_sent) ms samples.
window.__bcastLat = [];
window.reportBcastLat = function () {
    const f = window.__bcastLat;
    if (!f.length) { console.log('[bcast-lat] no samples yet'); return; }
    const s = [...f].sort((a, b) => a - b);
    const mean = f.reduce((a, b) => a + b, 0) / f.length;
    console.log('[bcast-lat] ==== emit -> remote render latency (ms) ====');
    console.log(`[bcast-lat] n=${f.length} mean=${mean.toFixed(1)} median=${s[Math.floor(s.length/2)].toFixed(1)} P95=${s[Math.floor(s.length*0.95)].toFixed(1)} min=${s[0].toFixed(1)} max=${s[s.length-1].toFixed(1)}`);
    console.log('[bcast-lat] COPY THE LINE ABOVE ^  (note: clock-sync dependent)');
};
function handleRemoteLetter(data) {
    if (typeof data.t_sent === 'number') {
        window.__bcastLat.push(Date.now() - data.t_sent);
    }
    addChatMessage(data.username, `[letter] ${data.letter}`);
}

function handleRemoteSpace(data) {
    addChatMessage(data.username, '[space]');
}

function handleRemoteSentence(data) {
    const div = document.createElement('div');
    div.className = 'sentence-item';
    div.innerHTML = `<strong>${data.username}:</strong> ${data.sentence}`;
    const historyEl = document.getElementById('sentenceHistory');
    if (historyEl) historyEl.prepend(div);

    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(data.sentence);
        utterance.lang = 'id-ID';
        utterance.rate = 0.9;
        speechSynthesis.speak(utterance);
    }
}


// Prevent ghost user after refresh: tell server we leave on unload
window.addEventListener('beforeunload', () => {
    try {
        if (socket && socket.connected && roomId) {
            socket.emit('leave', { room: roomId, username });
            // Also disconnect quickly to trigger server cleanup
            socket.disconnect();
        }
    } catch(_){}
});
