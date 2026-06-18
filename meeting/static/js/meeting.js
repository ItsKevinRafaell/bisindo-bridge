/**
 * BISINDO Meeting Client v5 - SIMPLE
 * No MediaPipe on client. Server does all processing.
 */

const SERVER_URL = window.location.origin;
let socket = null;
let localStream = null;
let roomId = null;
let username = null;
let isBisindoMode = true;
let predictionBuffer = [];
let videoShareInterval = null;
let lastFrameTime = 0;
let mediapipeReady = true; // Server handles it

document.addEventListener('DOMContentLoaded', async () => {
    const saved = localStorage.getItem('bisindo_username');
    if (saved) document.getElementById('username').value = saved;
    init();
});

async function init() {
    console.log('🚀 Initializing...');

    socket = io(SERVER_URL);

    socket.on('connect', () => {
        console.log('✅ Connected');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('❌ Disconnected');
        updateConnectionStatus(false);
    });

    socket.on('room_info', (data) => {
        console.log('📋 Room:', data);
        updateUsersList(data.users);
        roomId = data.room;
        document.getElementById('currentRoom').textContent = data.room;
    });

    socket.on('user_joined', (data) => {
        console.log('👤 Joined:', data.username);
        addChatMessage('System', `${data.username} joined`);
        updateUsersList(data.users);
    });

    socket.on('user_left', (data) => {
        console.log('👤 Left:', data.username);
        addChatMessage('System', `${data.username} left`);
        updateUsersList(data.users);
    });

    socket.on('prediction', (data) => {
        if (data.error) {
            console.error('❌ Prediction error:', data.error);
            return;
        }
        handlePrediction(data);
    });

    socket.on('video_frame', (data) => {
        console.log('📹 Video from:', data.username);
        if (data.frame) {
            showRemoteVideo(data.username, data.frame);
        }
    });

    socket.on('chat_message', (data) => {
        addChatMessage(data.username, data.message);
    });

    console.log('✅ Client ready! Waiting for camera...');
}

async function joinMeeting() {
    username = document.getElementById('username').value || 'User_' + Math.random().toString(36).substr(2, 4);
    roomId = document.getElementById('roomId').value || 'room_' + Math.random().toString(36).substr(2, 6);

    document.getElementById('localUsername').textContent = username;
    document.getElementById('currentRoom').textContent = roomId;
    document.getElementById('joinForm').style.display = 'none';
    document.getElementById('meetingContainer').classList.add('active');

    if (username) localStorage.setItem('bisindo_username', username);

    try {
        console.log('📷 Requesting camera...');
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true
        });

        const localVideo = document.getElementById('localVideo');
        localVideo.srcObject = localStream;

        localVideo.onloadedmetadata = () => {
            console.log('📹 Camera:', localVideo.videoWidth, 'x', localVideo.videoHeight);

            if (socket.connected) {
                socket.emit('join', { room: roomId, username });
            } else {
                socket.on('connect', () => {
                    socket.emit('join', { room: roomId, username });
                });
            }

            startVideoSharing();
            console.log('✅ Meeting started!');
        };
    } catch (err) {
        console.error('❌ Camera failed:', err);
        alert('Camera error: ' + err.message);
    }
}

function startVideoSharing() {
    if (videoShareInterval) return;
    const localVideo = document.getElementById('localVideo');

    videoShareInterval = setInterval(() => {
        if (!localStream || !localVideo.srcObject || !socket || !roomId || !socket.connected) return;

        const now = Date.now();
        if (now - lastFrameTime < 200) return;
        lastFrameTime = now;

        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = localVideo.videoWidth || 640;
        canvas.height = localVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(localVideo, 0, 0);

        // Send frame to server for processing
        const frameData = canvas.toDataURL('image/jpeg', 0.5);

        socket.emit('video_frame', {
            room: roomId,
            username: username,
            frame: frameData
        });
    }, 200);
    console.log('✅ Video sharing started');
}

function showRemoteVideo(peerName, frameData) {
    const remoteVideo = document.getElementById('remoteVideo');
    if (remoteVideo) {
        // Create img element or use img tag
        const container = remoteVideo.parentElement;
        let img = container.querySelector('img.remote-img');
        if (!img) {
            img = document.createElement('img');
            img.className = 'remote-img';
            img.style.cssText = 'width:100%;height:240px;object-fit:cover;background:#000;';
            remoteVideo.style.display = 'none';
            container.insertBefore(img, remoteVideo);
        }
        img.src = frameData;
    }
}

function updateConnectionStatus(online) {
    console.log('Connection:', online ? 'ONLINE' : 'OFFLINE');
}

function handlePrediction(data) {
    const overlay = document.getElementById('predictionOverlay');
    if (overlay) {
        overlay.textContent = data.letter;
        overlay.style.display = 'block';
    }

    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    if (confidenceFill) confidenceFill.style.width = `${data.confidence * 100}%`;
    if (confidenceText) confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    if (data.confidence > 0.5) {
        if (predictionBuffer.length === 0 || predictionBuffer[predictionBuffer.length - 1] !== data.letter) {
            predictionBuffer.push(data.letter);
            if (predictionBuffer.length > 20) predictionBuffer.shift();
            updatePredictionHistory();
        }
    }
}

function updatePredictionHistory() {
    const container = document.getElementById('predictionHistory');
    if (!container) return;

    container.innerHTML = '';
    if (predictionBuffer.length === 0) {
        container.innerHTML = '<span style="color:rgba(255,255,255,0.5)">Akan muncul di sini...</span>';
        return;
    }
    predictionBuffer.forEach(letter => {
        const span = document.createElement('span');
        span.className = 'predicted-letter';
        span.textContent = letter;
        container.appendChild(span);
    });
}

function toggleCamera() {
    const videoTrack = localStream?.getVideoTracks()[0];
    if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        const btn = document.getElementById('cameraBtn');
        btn.classList.toggle('active', videoTrack.enabled);
        btn.textContent = videoTrack.enabled ? '📹 Camera' : '📹 Camera Off';
    }
}

function toggleMic() {
    const audioTrack = localStream?.getAudioTracks()[0];
    if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        const btn = document.getElementById('micBtn');
        btn.classList.toggle('active', audioTrack.enabled);
        btn.textContent = audioTrack.enabled ? '🎤 Mic' : '🎤 Mic Off';
    }
}

function toggleBisindo() {
    isBisindoMode = !isBisindoMode;
    const btn = document.getElementById('bisindoBtn');
    btn.classList.toggle('active', isBisindoMode);
    console.log('BISINDO Mode:', isBisindoMode ? 'ON' : 'OFF');
}

function updateUsersList(users) {
    const container = document.getElementById('usersList');
    if (!container) return;
    container.innerHTML = '';
    users.forEach(u => {
        const div = document.createElement('div');
        div.className = 'user-item';
        div.innerHTML = `<div class="user-avatar">${(u.username||'?')[0].toUpperCase()}</div><span>${u.username}</span><span class="status-indicator online"></span>`;
        container.appendChild(div);
    });
}

function addChatMessage(sender, msg) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const div = document.createElement('div');
    div.className = `chat-message ${sender === username ? 'own' : 'other'}`;
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
