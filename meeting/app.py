"""
BISINDO Meeting Server - Flask-SocketIO
Real-time video meeting with BISINDO gesture recognition
"""

import os
import base64
import io
import json
import numpy as np
import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import BISINDO components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.landmark_classifier import LandmarkClassifier
from src.tts_engine import TTSEngine

# Config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'bisindo-meeting-secret-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Add COOP/COEP headers only for MediaPipe WASM files
@app.after_request
def add_coop_coep_headers(response):
    if '/mediapipe/' in request.path or request.path.endswith('.wasm'):
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'models')

# Initialize BISINDO classifier
classifier = None
tts_engine = TTSEngine(language='id')

def init_classifier():
    """Load BISINDO landmark classifier"""
    global classifier

    model_path = os.path.join(DATA_DIR, 'landmark_classifier.pkl')
    scaler_path = os.path.join(DATA_DIR, 'landmark_classifier_scaler.pkl')
    label_path = os.path.join(DATA_DIR, 'landmark_classifier_labels.pkl')

    # Fallback to backup-v1 models if not found
    if not os.path.exists(model_path):
        model_path = os.path.join(DATA_DIR, '..', 'backup-v1-image-based', 'models', 'enhanced_classifier.pkl')
        scaler_path = os.path.join(DATA_DIR, '..', 'backup-v1-image-based', 'models', 'enhanced_scaler.pkl')
        label_path = os.path.join(DATA_DIR, '..', 'backup-v1-image-based', 'models', 'enhanced_labels.pkl')

    try:
        classifier = LandmarkClassifier(model_path, scaler_path, label_path)
        print(f"✅ BISINDO classifier loaded from {model_path}")
    except Exception as e:
        print(f"⚠️ Classifier load failed: {e}")
        classifier = None

# Room management
rooms = {}  # room_id -> {users: [], last_prediction: {}}

@app.route('/')
def index():
    """Meeting home page"""
    return render_template('index.html')

@app.route('/room/<room_id>')
def room(room_id):
    """Specific meeting room"""
    return render_template('index.html', room_id=room_id)

# SocketIO Events
@socketio.on('connect')
def on_connect():
    print(f"👤 User connected: {request.sid if 'request' in dir() else 'unknown'}")
    emit('connected', {'sid': 'demo-sid'})

@socketio.on('disconnect')
def on_disconnect():
    print("👤 User disconnected")

@socketio.on('join')
def on_join(data):
    """User joins a meeting room"""
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')

    join_room(room_id)

    if room_id not in rooms:
        rooms[room_id] = {'users': [], 'history': []}

    rooms[room_id]['users'].append({'sid': 'demo', 'username': username})

    # Notify others
    emit('user_joined', {
        'username': username,
        'users': rooms[room_id]['users']
    }, room=room_id)

    emit('room_info', {
        'room': room_id,
        'users': rooms[room_id]['users']
    })

@socketio.on('leave')
def on_leave(data):
    """User leaves a meeting room"""
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')

    leave_room(room_id)

    if room_id in rooms:
        rooms[room_id]['users'] = [u for u in rooms[room_id]['users'] if u['username'] != username]

    emit('user_left', {'username': username}, room=room_id)

@socketio.on('text_message')
def on_text_message(data):
    """Broadcast text message to room"""
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')
    message = data.get('message', '')

    emit('chat_message', {
        'username': username,
        'message': message,
        'timestamp': data.get('timestamp', '')
    }, room=room_id)

@socketio.on('video_frame')
def on_video_frame(data):
    """Process video frame: extract landmarks and predict, then relay to peers"""
    room_id = data.get('room', 'default')
    username = data.get('username', 'unknown')
    frame_data = data.get('frame', '')

    if not room_id or not frame_data:
        return

    print(f"📹 Video from {username}, frame size: {len(frame_data)} bytes")

    # Broadcast to other users
    emit('video_frame', {
        'username': username,
        'frame': frame_data
    }, room=room_id, include_self=False)

    # Process for BISINDO if classifier is available
    if not classifier:
        return

    try:
        # Decode base64 to image
        img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Extract landmarks
        landmarks = classifier.extract_landmarks(frame)

        if landmarks and len(landmarks) == 63:
            # Predict
            prediction = classifier.predict(landmarks)

            print(f"🎯 {username}: {prediction['letter']} ({prediction['confidence']:.2f})")

            if prediction['confidence'] > 0.5:
                # Broadcast prediction
                emit('prediction', {
                    'letter': prediction['letter'],
                    'confidence': prediction['confidence'],
                    'username': username
                }, room=room_id, include_self=True)

                # TTS
                letter_audio = tts_engine.text_to_audio(prediction['letter'])
                if letter_audio:
                    audio_b64 = base64.b64encode(letter_audio).decode('utf-8')
                    emit('audio', {
                        'audio': f'data:audio/mp3;base64,{audio_b64}',
                        'letter': prediction['letter']
                    }, room=room_id, include_self=True)

    except Exception as e:
        print(f"❌ Frame processing error: {e}")

@socketio.on('clear_buffer')
def on_clear_buffer(data):
    """Clear letter buffer for this user"""
    room_id = data.get('room', 'default')
    if room_id in rooms:
        rooms[room_id]['history'] = []

# Initialize and run
if __name__ == '__main__':
    import socket

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    init_classifier()

    local_ip = get_local_ip()
    print("🚀 BISINDO Meeting Server starting...")
    print(f"   Local:   http://localhost:5000")
    print(f"   Network: http://{local_ip}:5000")
    print(f"\n📋 Share with friends on same network: http://{local_ip}:5000")
    print(f"   Or share specific room: http://{local_ip}:5000/room/nameroom")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
