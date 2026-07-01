/**
 * Teman Meeting Kamu - Client
 * With client-side ONNX inference + Sentence Builder
 */

const SERVER_URL = window.location.origin;
let socket = null;
let localStream = null;
let roomId = null;
let username = null;
let videoShareInterval = null;
let lastFrameTime = 0;

// MediaPipe HandLandmarker (2 hands)
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
const GESTURE_CONFIRM_MS = 800;

const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
];

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

    socket.on('letter_committed', (data) => {
        if (data.username !== username) handleRemoteLetter(data);
    });

    socket.on('space_inserted', (data) => {
        if (data.username !== username) handleRemoteSpace(data);
    });

    socket.on('sentence_broadcast', (data) => {
        if (data.username !== username) handleRemoteSentence(data);
    });

    socket.on('video_frame', (data) => {
        showRemoteVideo(data.username, data.frame);
    });

    socket.on('chat_message', (data) => {
        addChatMessage(data.username, data.message);
    });

    // Load TF.js model in background
    loadModel();
}

// --- ONNX Model ---

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
        console.error('Error details:', JSON.stringify(err, null, 2));
        if (statusEl) statusEl.textContent = 'Model failed: ' + (err.message || err);
    }
}

// --- Feature extraction (84 features: xy only) ---

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
    // CNN expects [1, 1, 84] = (batch, channels, features)
    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 1, features.length]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const outputKey = Object.keys(results)[0];
    const probs = results[outputKey].data;
    return Array.from(probs);
}

// --- Gesture Detection ---

function detectGesture(landmarks) {
    if (!landmarks || landmarks.length < 21) return null;
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

// --- MediaPipe HandLandmarker ---

async function joinMeeting() {
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
            initHandLandmarker();
            initSentenceBuilder();
            startVideoSharing();
        };
    } catch (err) {
        alert('Camera error: ' + err.message);
    }
}

async function initHandLandmarker() {
    const localVideo = document.getElementById('localVideo');
    const card = document.getElementById('localCard');

    // Create overlay canvas
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
        // Draw landmarks
        results.landmarks.forEach((lms, handIdx) => drawHand(lms, handIdx));

        // Gesture detection with 800ms confirmation
        const rawGesture = detectGesture(results.landmarks[0]);
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

        // CNN prediction (if model loaded)
        if (modelLoaded) {
            const rawFeats = landmarksToFeatures(results);
            if (rawFeats) {
                const normalized = normalizeFeaturesHandCentric(rawFeats);
                const scaled = scaleFeatures(normalized);
                predictModel(scaled).then(probs => {
                    const indexed = probs.map((p, i) => ({ letter: labels[i], p }));
                    indexed.sort((a, b) => b.p - a.p);
                    const top = indexed[0];

                    // Update overlay
                    const overlay = document.getElementById('predictionOverlay');
                    if (overlay) {
                        overlay.textContent = top.p > 0.5 ? top.letter : '?';
                        overlay.style.display = 'block';
                    }

                    if (top.p > 0.5) {
                        prediction = { letter: top.letter, confidence: top.p };
                    }

                    // Feed to sentence builder
                    if (sentenceBuilder) {
                        sentenceBuilder.feed(prediction, handDetected, gestureState);
                    }
                });
            }
        }

        // Show gesture indicator on canvas
        if (gestureState === 'fist') {
            mpCtx.fillStyle = 'rgba(239, 68, 68, 0.7)';
            mpCtx.fillRect(mpCanvas.width - 120, mpCanvas.height - 40, 110, 30);
            mpCtx.fillStyle = '#fff';
            mpCtx.font = '16px Poppins';
            mpCtx.fillText('✊ SPACE', mpCanvas.width - 110, mpCanvas.height - 18);
        } else if (gestureState === 'palm') {
            mpCtx.fillStyle = 'rgba(59, 130, 246, 0.7)';
            mpCtx.fillRect(mpCanvas.width - 120, mpCanvas.height - 40, 110, 30);
            mpCtx.fillStyle = '#fff';
            mpCtx.font = '16px Poppins';
            mpCtx.fillText('🖐 ENTER', mpCanvas.width - 110, mpCanvas.height - 18);
        }
    } else {
        lastGestureState = null;
        // Feed no-hand to sentence builder
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
    for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = landmarks[s], p2 = landmarks[e];
        if (!p1 || !p2) continue;
        mpCtx.beginPath();
        mpCtx.moveTo(p1.x * w, p1.y * h);
        mpCtx.lineTo(p2.x * w, p2.y * h);
        mpCtx.stroke();
    }
    mpCtx.fillStyle = handIdx === 0 ? '#FF4444' : '#FF8800';
    for (const p of landmarks) {
        mpCtx.beginPath();
        mpCtx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
        mpCtx.fill();
    }
}

// --- Sentence Builder ---

function initSentenceBuilder() {
    sentenceBuilder = new SentenceBuilder({
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
}

// Manual controls
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

// --- Remote sentence handlers ---

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

    // Auto-speak remote sentences
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(data.sentence);
        utterance.lang = 'id-ID';
        utterance.rate = 0.9;
        speechSynthesis.speak(utterance);
    }
}

// --- Exit ---

function exitMeeting() {
    if (confirm('Keluar dari meeting?')) {
        detectionActive = false;
        socket.emit('leave', { room: roomId, username });
        if (localStream) localStream.getTracks().forEach(track => track.stop());
        if (videoShareInterval) clearInterval(videoShareInterval);

        document.getElementById('meetingContainer').classList.remove('active');
        document.getElementById('joinForm').style.display = 'flex';

        if (mpCanvas && mpCanvas.parentNode) {
            mpCanvas.parentNode.removeChild(mpCanvas);
        }
    }
}

// --- Video Grid ---

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

// --- Video Sharing ---

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

// --- Controls ---

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

// --- Sidebar ---

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
