/**
 * Teman Meeting Kamu - Client
 * With client-side MediaPipe hand detection
 */

const SERVER_URL = window.location.origin;
let socket = null;
let localStream = null;
let roomId = null;
let username = null;
let predictionBuffer = [];
let videoShareInterval = null;
let lastFrameTime = 0;

// MediaPipe
let handsDetector = null;
let mpCanvas = null;
let mpCtx = null;
let detectionActive = false;
let lastPredictionTime = 0;
const PREDICTION_THROTTLE = 300; // ms between predictions

document.addEventListener('DOMContentLoaded', async () => {
    const saved = localStorage.getItem('bisindo_username');
    if (saved) document.getElementById('username').value = saved;
    init();
});

async function init() {
    socket = io(SERVER_URL);

    socket.on('connect', () => console.log('Connected'));
    socket.on('disconnect', () => console.log('Disconnected'));

    socket.on('room_info', (data) => {
        roomId = data.room;
        document.getElementById('currentRoom').textContent = data.room;
        updateUsersList(data.users);
    });

    socket.on('user_joined', (data) => {
        addChatMessage('System', `${data.username} joined`);
        updateUsersList(data.users);
    });

    socket.on('user_left', (data) => {
        addChatMessage('System', `${data.username} left`);
        updateUsersList(data.users);
        removeVideoCard(data.username);
    });

    socket.on('prediction', (data) => {
        if (data.error) return;
        handlePrediction(data);
    });

    socket.on('video_frame', (data) => {
        showRemoteVideo(data.username, data.frame);
    });

    socket.on('chat_message', (data) => {
        addChatMessage(data.username, data.message);
    });
}

async function joinMeeting() {
    // Validate name
    const nameInput = document.getElementById('username');
    username = nameInput.value.trim();
    if (!username) {
        nameInput.style.borderColor = 'var(--danger)';
        nameInput.focus();
        return;
    }
    nameInput.style.borderColor = '';

    roomId = document.getElementById('roomId').value.trim() || 'room_' + Math.random().toString(36).substr(2, 6);

    document.getElementById('localUsername').textContent = username;
    document.getElementById('currentRoom').textContent = roomId;
    document.getElementById('joinForm').style.display = 'none';
    document.getElementById('meetingContainer').classList.add('active');

    localStorage.setItem('bisindo_username', username);

    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true
        });

        const localVideo = document.getElementById('localVideo');
        localVideo.srcObject = localStream;

        localVideo.onloadedmetadata = () => {
            updateVideoGrid();
            if (socket.connected) {
                socket.emit('join', { room: roomId, username });
            } else {
                socket.on('connect', () => {
                    socket.emit('join', { room: roomId, username });
                });
            }
            initMediaPipe();
            startVideoSharing();
        };
    } catch (err) {
        alert('Camera error: ' + err.message);
    }
}

async function initMediaPipe() {
    const localVideo = document.getElementById('localVideo');
    const card = document.getElementById('localCard');

    // Create overlay canvas for drawing landmarks
    mpCanvas = document.createElement('canvas');
    mpCanvas.id = 'mpCanvas';
    mpCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;';
    card.appendChild(mpCanvas);

    localVideo.addEventListener('loadedmetadata', () => {
        mpCanvas.width = localVideo.videoWidth;
        mpCanvas.height = localVideo.videoHeight;
    });
    mpCanvas.width = 640;
    mpCanvas.height = 480;
    mpCtx = mpCanvas.getContext('2d');

    try {
        // Load MediaPipe Hands from LOCAL files
        await loadScript('/static/mediapipe/hands.js');
        await loadScript('/static/mediapipe/drawing_utils.js');

        handsDetector = new Hands({
            locateFile: (file) => `/static/mediapipe/${file}`
        });
        handsDetector.setOptions({
            maxNumHands: 1,
            modelComplexity: 0,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
            useCpuInference: true
        });
        handsDetector.onResults(onHandResults);

        detectionActive = true;
        runDetection();
        console.log('✅ MediaPipe Hands initialized');
    } catch (err) {
        console.error('MediaPipe init failed:', err);
    }
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.crossOrigin = 'anonymous';
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

function runDetection() {
    if (!detectionActive) return;

    const localVideo = document.getElementById('localVideo');
    if (localVideo && localVideo.readyState >= 2) {
        handsDetector.send({ image: localVideo }).catch(() => {});
    }
    requestAnimationFrame(runDetection);
}

const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],       // thumb
    [0,5],[5,6],[6,7],[7,8],       // index
    [0,9],[9,10],[10,11],[11,12],  // middle
    [0,13],[13,14],[14,15],[15,16],// ring
    [0,17],[17,18],[18,19],[19,20],// pinky
    [5,9],[9,13],[13,17]           // palm
];

function onHandResults(results) {
    if (!mpCtx) return;

    mpCtx.clearRect(0, 0, mpCanvas.width, mpCanvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        const w = mpCanvas.width;
        const h = mpCanvas.height;

        // Draw connections
        mpCtx.strokeStyle = '#00FF88';
        mpCtx.lineWidth = 3;
        HAND_CONNECTIONS.forEach(([s, e]) => {
            const p1 = landmarks[s], p2 = landmarks[e];
            if (p1 && p2) {
                mpCtx.beginPath();
                mpCtx.moveTo(p1.x * w, p1.y * h);
                mpCtx.lineTo(p2.x * w, p2.y * h);
                mpCtx.stroke();
            }
        });

        // Draw points
        mpCtx.fillStyle = '#FF4444';
        landmarks.forEach(p => {
            mpCtx.beginPath();
            mpCtx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
            mpCtx.fill();
        });

        // Send landmarks to server for prediction (throttled)
        const now = Date.now();
        if (now - lastPredictionTime > PREDICTION_THROTTLE) {
            lastPredictionTime = now;
            const flat = [];
            landmarks.forEach(p => flat.push(p.x, p.y, p.z));
            socket.emit('predict_landmarks', {
                room: roomId,
                username: username,
                landmarks: flat
            });
        }
    }
}

function exitMeeting() {
    if (confirm('Keluar dari meeting?')) {
        detectionActive = false;
        socket.emit('leave', { room: roomId, username });
        if (localStream) localStream.getTracks().forEach(track => track.stop());
        if (videoShareInterval) clearInterval(videoShareInterval);

        document.getElementById('meetingContainer').classList.remove('active');
        document.getElementById('joinForm').style.display = 'flex';
        predictionBuffer = [];

        // Remove canvas
        if (mpCanvas && mpCanvas.parentNode) {
            mpCanvas.parentNode.removeChild(mpCanvas);
        }
    }
}

function updateVideoGrid() {
    const grid = document.getElementById('videoGrid');
    const cards = grid.querySelectorAll('.video-card');

    grid.classList.remove('single', 'two', 'three', 'four', 'five', 'six', 'many');
    const count = cards.length;

    if (count === 1) grid.classList.add('single');
    else if (count === 2) grid.classList.add('two');
    else if (count === 3) grid.classList.add('three');
    else if (count === 4) grid.classList.add('four');
    else if (count === 5) grid.classList.add('five');
    else if (count === 6) grid.classList.add('six');
    else grid.classList.add('many');
}

function startVideoSharing() {
    if (videoShareInterval) return;
    const localVideo = document.getElementById('localVideo');

    videoShareInterval = setInterval(() => {
        if (!localStream || !localVideo.srcObject || !socket || !roomId || !socket.connected) return;

        const now = Date.now();
        if (now - lastFrameTime < 200) return;
        lastFrameTime = now;

        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);

        socket.emit('video_frame', {
            room: roomId,
            username: username,
            frame: canvas.toDataURL('image/jpeg', 0.5)
        });
    }, 200);
}

function showRemoteVideo(peerName, frameData) {
    if (peerName === username) return;

    let card = document.getElementById('card_' + peerName);

    if (!card) {
        card = document.createElement('div');
        card.className = 'video-card';
        card.id = 'card_' + peerName;
        card.innerHTML = `
            <img src="${frameData}" alt="${peerName}">
            <div class="username">${peerName}</div>
        `;
        document.getElementById('videoGrid').appendChild(card);
        updateVideoGrid();
    } else {
        const img = card.querySelector('img');
        if (img) img.src = frameData;
    }
}

function removeVideoCard(peerName) {
    const card = document.getElementById('card_' + peerName);
    if (card) {
        card.remove();
        updateVideoGrid();
    }
}

function handlePrediction(data) {
    const overlay = document.getElementById('predictionOverlay');
    if (overlay) {
        overlay.textContent = data.letter;
        overlay.style.display = 'block';
    }

    const confidenceFill = document.getElementById('confidenceFill');
    if (confidenceFill) confidenceFill.style.width = `${data.confidence * 100}%`;

    if (data.confidence > 0.5) {
        if (predictionBuffer.length === 0 || predictionBuffer[predictionBuffer.length - 1].letter !== data.letter) {
            predictionBuffer.push({ letter: data.letter, user: data.username });
            if (predictionBuffer.length > 20) predictionBuffer.shift();
            updatePredictionHistory();
        }
    }
}

function updatePredictionHistory() {
    const container = document.getElementById('predictionHistory');
    container.innerHTML = '';
    if (predictionBuffer.length === 0) {
        container.innerHTML = '<span style="color:var(--text-muted)">Akan muncul...</span>';
        return;
    }
    predictionBuffer.forEach(p => {
        const span = document.createElement('span');
        span.className = 'predicted-letter';
        span.textContent = `${p.letter}`;
        span.title = `${p.user}: ${p.letter}`;
        container.appendChild(span);
    });
}

function toggleCamera() {
    const videoTrack = localStream?.getVideoTracks()[0];
    if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        const btn = document.getElementById('cameraBtn');
        btn.classList.toggle('active', videoTrack.enabled);
        btn.classList.toggle('muted', !videoTrack.enabled);
        if (!videoTrack.enabled) detectionActive = false;
        else { detectionActive = true; runDetection(); }
    }
}

function toggleMic() {
    const audioTrack = localStream?.getAudioTracks()[0];
    if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        const btn = document.getElementById('micBtn');
        btn.classList.toggle('active', audioTrack.enabled);
        btn.classList.toggle('muted', !audioTrack.enabled);
    }
}

function updateUsersList(users) {
    const container = document.getElementById('usersList');
    document.getElementById('userCount').textContent = users.length;
    container.innerHTML = '';
    users.forEach(u => {
        const div = document.createElement('div');
        div.className = 'user-item';
        div.innerHTML = `
            <div class="user-avatar">${(u.username||'?')[0].toUpperCase()}</div>
            <span class="user-name">${u.username}</span>
            <span class="status-dot"></span>
        `;
        container.appendChild(div);
    });
}

function addChatMessage(sender, msg) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    const isOwn = sender === username;
    div.className = `chat-message ${isOwn ? 'own' : 'other'}`;
    div.innerHTML = `<strong>${sender}:</strong> ${msg}`;
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

function copyRoomLink() {
    const link = `${window.location.origin}/room/${roomId}`;
    navigator.clipboard.writeText(link).then(() => {
        alert('Room link copied!');
    });
}
