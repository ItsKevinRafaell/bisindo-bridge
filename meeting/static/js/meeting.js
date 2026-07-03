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
let currentLayout = 'gallery'; // 'gallery' | 'speaker'
let pinnedSpeakerSid = null;
let activeSpeakerSid = null;
let meetingStartTime = null;
let timerInterval = null;
let sidePanelOpen = false;
let activeTab = 'chat';
let unreadChatCount = 0;

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
    } catch (err) {
        console.error('Camera error:', err);
        errorEl.textContent = 'Tidak bisa akses kamera/mic. Pastikan izin browser sudah diberikan.';
        errorEl.style.display = 'block';
        previewVideo.style.display = 'none';
        previewNoCam.style.display = 'flex';

        // Try audio-only
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            previewAvatar.textContent = '?';
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

    // Init WebRTC
    peerManager = new PeerManager(socket, localStream, roomId, username);
    peerManager.onRemoteStream = (sid, peerUsername, stream) => {
        showRemoteStream(sid, peerUsername, stream);
        setupAudioDetection(sid, stream);
    };
    peerManager.onRemoveStream = (sid) => {
        removeVideoCard(sid);
        cleanupAudioDetection(sid);
    };

    // Init hand tracking
    initHandLandmarker();
    initSentenceBuilder();

    // Update mic/cam state
    updateMicCamUI();
}

function initSocket() {
    socket = io(SERVER_URL);

    socket.on('connect', () => {
        console.log('Connected to server');
        socket.emit('join', { room: roomId, username });
        // Broadcast initial state
        broadcastPeerState();
    });

    socket.on('disconnect', () => console.log('Disconnected'));

    socket.on('room_info', (data) => {
        roomId = data.room;
        document.getElementById('topbarRoom').textContent = data.room;
        updatePeopleList(data.users);
    });

    socket.on('user_joined', (data) => {
        addChatMessage('System', `${data.username} joined`);
        updatePeopleList(data.users);

        if (peerManager && data.sid) {
            peerManager.connectToPeer(data.sid, data.username);
        }

        // Auto-downgrade: disable video if >6 peers
        if (peerManager && peerManager.peers.size > 6) {
            const videoTrack = localStream.getVideoTracks()[0];
            if (videoTrack) {
                videoTrack.enabled = false;
                camEnabled = false;
                updateMicCamUI();
                addChatMessage('System', 'Video disabled (too many peers for stability)');
            }
        }
    });

    socket.on('user_left', (data) => {
        addChatMessage('System', `${data.username} left`);
        updatePeopleList(data.users);

        if (peerManager) {
            for (const [sid, peer] of peerManager.peers) {
                if (peer.username === data.username) {
                    peerManager.handlePeerDisconnect(sid);
                    break;
                }
            }
        }
        peerStates.delete(data.sid);
        audioContexts.delete(data.sid);
    });

    // BISINDO events
    socket.on('letter_committed', (data) => {
        if (data.username !== username) handleRemoteLetter(data);
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

    // Hand raise events
    socket.on('hand_raise', (data) => {
        const state = peerStates.get(data.sid) || {};
        state.handRaised = true;
        peerStates.set(data.sid, state);
        updateHandBadge(data.sid, true);
        updatePeopleList();
        addChatMessage('System', `${data.username} raised hand ✋`);
    });

    socket.on('hand_lower', (data) => {
        const state = peerStates.get(data.sid) || {};
        state.handRaised = false;
        peerStates.set(data.sid, state);
        updateHandBadge(data.sid, false);
        updatePeopleList();
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
        updatePeopleList();
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
            <video autoplay playsinline></video>
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
        grid.appendChild(card);
    }

    const video = card.querySelector('video');
    if (video) {
        video.srcObject = stream;
        video.play().catch(err => console.warn('Video play failed:', err));
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
        // Remove any speaker view elements
        const mainSpeaker = document.querySelector('.main-speaker');
        const filmstrip = document.querySelector('.filmstrip');
        if (mainSpeaker) mainSpeaker.remove();
        if (filmstrip) filmstrip.remove();
        grid.style.display = '';
    }
}

function updateSpeakerView() {
    const grid = document.getElementById('videoGrid');
    const videoArea = document.getElementById('videoArea');

    // Hide the grid
    grid.style.display = 'none';

    // Remove old speaker view elements
    const oldMain = videoArea.querySelector('.main-speaker');
    const oldFilm = videoArea.querySelector('.filmstrip');
    if (oldMain) oldMain.remove();
    if (oldFilm) oldFilm.remove();

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

    // Create main speaker area
    const mainSpeaker = document.createElement('div');
    mainSpeaker.className = 'main-speaker';
    mainSpeaker.appendChild(mainCard.cloneNode(true));
    videoArea.insertBefore(mainSpeaker, grid);

    // Create filmstrip
    const filmstrip = document.createElement('div');
    filmstrip.className = 'filmstrip';
    cards.forEach(card => {
        if (card !== mainCard) {
            filmstrip.appendChild(card.cloneNode(true));
        }
    });
    videoArea.insertBefore(filmstrip, grid);
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

        // Toggle detection
        if (!camEnabled) {
            detectionActive = false;
        } else {
            detectionActive = true;
            runDetection();
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
    // Close popup
    document.getElementById('reactionsPopup').classList.remove('open');

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
    addChatMessage(username, msg);
    input.value = '';
}

// ============================================================
// PEOPLE LIST
// ============================================================

function updatePeopleList(users) {
    const container = document.getElementById('peopleList');
    if (!container) return;

    // Build list from users array + peer states
    const allUsers = users || [];
    container.innerHTML = '';

    allUsers.forEach(u => {
        const sid = u.sid;
        const state = peerStates.get(sid) || { mic: true, cam: true, handRaised: false };
        const isLocal = u.username === username;

        const div = document.createElement('div');
        div.className = 'person-item';
        div.innerHTML = `
            <div class="person-avatar">${(u.username || '?')[0].toUpperCase()}</div>
            <div class="person-info">
                <div class="person-name">${u.username}${isLocal ? ' (You)' : ''}</div>
                <div class="person-status">
                    <i class="fas fa-microphone ${state.mic === false ? 'muted' : ''}"></i>
                    <i class="fas fa-video ${state.cam === false ? 'muted' : ''}"></i>
                    ${state.handRaised ? '<i class="fas fa-hand-paper raised"></i>' : ''}
                </div>
            </div>
        `;
        container.appendChild(div);
    });

    document.getElementById('participantCount').textContent = allUsers.length;
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
    if (confirm('Keluar dari meeting?')) {
        detectionActive = false;
        if (timerInterval) clearInterval(timerInterval);

        if (socket) {
            socket.emit('leave', { room: roomId, username });
            socket.disconnect();
        }
        if (peerManager) {
            peerManager.destroy();
            peerManager = null;
        }
        if (localStream) localStream.getTracks().forEach(track => track.stop());

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

        detectionActive = true;
        runDetection();
        console.log('✅ MediaPipe HandLandmarker initialized');
    } catch (err) {
        console.error('HandLandmarker init failed:', err);
    }
}

function runDetection() {
    if (!detectionActive || !handLandmarker) return;

    const localVideo = document.getElementById('localVideo');
    if (localVideo && localVideo.readyState >= 2) {
        videoTs += 33;
        const results = handLandmarker.detectForVideo(localVideo, videoTs);
        processResults(results);
    }
    requestAnimationFrame(runDetection);
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
                socket.emit('letter_committed', { room: roomId, username, letter });
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
function handleRemoteLetter(data) {
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
