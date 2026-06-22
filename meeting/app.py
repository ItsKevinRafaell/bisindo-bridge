"""
BISINDO Meeting Server - Flask-SocketIO + REST capture ingest
Real-time video meeting with BISINDO gesture recognition.

Schema (CSV, 67 cols):
  letter, image_path, split, num_hands, contributor, lm0_x..lm20_z (63)

Inference: client-side (browser). Server keeps a copy of the .pkl for
optional server-side prediction in /predict_landmarks socket event.
"""

import os
import csv
import socket
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.landmark_classifier import LandmarkClassifier

# ---- Config ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'bisindo-meeting-secret-2026'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HEADER = ["letter", "image_path", "split", "num_hands", "contributor"]
HEADER += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
CSV_PATH = os.path.join(DATA_DIR, 'landmarks_captured_v2.csv')

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
TRAIN_TARGET = 50000  # target samples per letter

# In-memory state
training_data = {}   # letter -> [{landmarks, hand_count, contributor, from_csv}]
rooms = {}
train_users = {}
contributor_stats = {}
classifier = None


# ---- Classifier ----
def init_classifier():
    global classifier
    paths = {
        'model': os.path.join(MODEL_DIR, 'landmark_classifier.pkl'),
        'scaler': os.path.join(MODEL_DIR, 'landmark_classifier_scaler.pkl'),
        'labels': os.path.join(MODEL_DIR, 'landmark_classifier_labels.pkl'),
    }
    try:
        classifier = LandmarkClassifier(paths['model'], paths['scaler'], paths['labels'])
        print(f"✅ Classifier loaded ({len(classifier.label_encoder.classes_)} classes)")
    except Exception as e:
        print(f"⚠️  Classifier load failed: {e}")
        classifier = None


# ---- CSV load/save ----
def load_existing_training_data():
    training_data.clear()
    if not os.path.exists(CSV_PATH):
        print(f"ℹ️  No CSV at {CSV_PATH}")
        return
    valid, skipped = 0, 0
    with open(CSV_PATH, 'r') as f:
        for row in csv.DictReader(f):
            try:
                hand1 = [float(row[f'lm{i}_{c}']) for i in range(21) for c in ('x', 'y', 'z')]
                letter = row['letter']
                contributor = row.get('contributor') or 'Unknown'
                training_data.setdefault(letter, []).append({
                    'landmarks': [hand1],
                    'hand_count': 2,
                    'contributor': contributor,
                    'from_csv': True,
                })
                contributor_stats[contributor] = contributor_stats.get(contributor, 0) + 1
                valid += 1
            except (ValueError, KeyError):
                skipped += 1
    print(f"✅ Loaded {valid} samples (skipped {skipped})")


def append_row(letter, hand1, contributor, source):
    """Append a single 67-col row to CSV_PATH. Returns row count for letter."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    next_idx = 0
    if file_exists:
        with open(CSV_PATH, 'r') as f:
            for r in csv.DictReader(f):
                if r['letter'] == letter:
                    next_idx += 1

    row = {
        'letter': letter,
        'image_path': f'{source}_{letter}_{next_idx}',
        'split': 'train',
        'num_hands': 2,
        'contributor': contributor,
    }
    for i in range(21):
        row[f'lm{i}_x'] = hand1[i * 3]
        row[f'lm{i}_y'] = hand1[i * 3 + 1]
        row[f'lm{i}_z'] = hand1[i * 3 + 2]

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---- HTTP routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/room/<room_id>')
def room(room_id):
    return render_template('index.html', room_id=room_id)

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/api/health')
def api_health():
    counts = {l: len(training_data.get(l, [])) for l in LETTERS}
    return jsonify({
        'status': 'ok',
        'classifier': classifier is not None,
        'samples': sum(len(v) for v in training_data.values()),
        'target': TRAIN_TARGET,
        'counts': counts,
    })


@app.route('/api/sample', methods=['POST'])
def api_sample():
    """REST ingest: capture page → server → CSV.

    Body JSON: {letter, hand1:[63], hand2:[63], contributor}
    Hand 2 is accepted for schema completeness; we currently store hand 1 only
    (2-hand storage doubles file size for the same info — kaggle augmented
    rows had hand 2 zero anyway).
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid JSON'}), 400

    letter = (data.get('letter') or '').upper()
    hand1 = data.get('hand1') or []
    contributor = (data.get('contributor') or 'Anonymous').strip()[:40] or 'Anonymous'

    if letter not in LETTERS:
        return jsonify({'ok': False, 'error': f'letter must be A-Z'}), 400
    if len(hand1) != 63:
        return jsonify({'ok': False, 'error': 'hand1 must be 63 floats'}), 400
    try:
        hand1 = [float(v) for v in hand1]
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'hand1 must be numeric'}), 400

    append_row(letter, hand1, contributor, source='web')
    training_data.setdefault(letter, []).append({
        'landmarks': [hand1],
        'hand_count': 2,
        'contributor': contributor,
        'from_csv': True,
    })
    contributor_stats[contributor] = contributor_stats.get(contributor, 0) + 1
    socketio.emit('train_info', get_train_info())

    return jsonify({'ok': True, 'letter': letter, 'count': len(training_data[letter])})


@app.route('/train/download')
def download_train_data():
    if not os.path.exists(CSV_PATH):
        return "No data yet", 404
    return send_file(CSV_PATH, as_attachment=True)


# ---- SocketIO: meeting ----
@socketio.on('connect')
def on_connect():
    emit('connected', {'sid': request.sid})

@socketio.on('disconnect')
def on_disconnect():
    pass

@socketio.on('join')
def on_join(data):
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')
    join_room(room_id)
    rooms.setdefault(room_id, {'users': [], 'history': []})['users'].append({'username': username})
    emit('room_info', {'room': room_id, 'users': rooms[room_id]['users']}, room=room_id)
    emit('user_joined', {'username': username, 'users': rooms[room_id]['users']}, room=room_id, include_self=False)

@socketio.on('leave')
def on_leave(data):
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')
    leave_room(room_id)
    if room_id in rooms:
        rooms[room_id]['users'] = [u for u in rooms[room_id]['users'] if u['username'] != username]
    emit('user_left', {'username': username}, room=room_id)

@socketio.on('text_message')
def on_text_message(data):
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')
    emit('chat_message', {'username': username, 'message': data.get('message', '')}, room=room_id)

@socketio.on('video_frame')
def on_video_frame(data):
    room_id = data.get('room', 'default')
    username = data.get('username', 'unknown')
    frame_data = data.get('frame', '')
    if room_id and frame_data:
        emit('video_frame', {'username': username, 'frame': frame_data}, room=room_id, include_self=False)

@socketio.on('predict_landmarks')
def on_predict_landmarks(data):
    """Client sends 63 floats (1 hand, normalized). Pad to 126 for the model."""
    if not classifier:
        return
    landmarks = data.get('landmarks', [])
    if len(landmarks) != 63:
        return
    try:
        feats = list(landmarks) + [0.0] * 63
        prediction = classifier.predict(feats)
        if prediction['confidence'] > 0.5:
            emit('prediction', {
                'letter': prediction['letter'],
                'confidence': prediction['confidence'],
                'username': data.get('username', 'unknown'),
            }, room=data.get('room', 'default'))
    except Exception as e:
        print(f"Predict error: {e}")


# ---- SocketIO: training (legacy /train page) ----
@socketio.on('train_join')
def on_train_join(data=None):
    username = (data or {}).get('username') or f"User_{request.sid[:6]}"
    train_users[request.sid] = username
    contributor_stats.setdefault(username, 0)
    emit('train_info', get_train_info())

@socketio.on('capture_landmark')
def on_capture_landmark(data):
    letter = (data.get('letter') or 'A').upper()
    landmarks = data.get('landmarks') or []
    contributor = data.get('contributor') or train_users.get(request.sid, 'Anonymous')

    if letter not in LETTERS or not landmarks or len(landmarks[0]) != 63:
        return

    hand1 = list(landmarks[0])
    if len(hand1) != 63:
        return
    if len(hand1) != 63:
        return
    try:
        hand1 = [float(v) for v in hand1]
    except (ValueError, TypeError):
        return

    append_row(letter, hand1, contributor, source='sock')
    training_data.setdefault(letter, []).append({
        'landmarks': [hand1],
        'hand_count': 2,
        'contributor': contributor,
        'from_csv': True,
    })
    contributor_stats[contributor] = contributor_stats.get(contributor, 0) + 1

    count = len(training_data[letter])
    total = sum(len(v) for v in training_data.values())
    emit('captured', {
        'letter': letter,
        'count': count,
        'total': total,
        'hand_count': 2,
        'contributor_stats': contributor_stats,
    })
    emit('train_info', get_train_info(), broadcast=True)


def get_train_info():
    counts = {l: len(training_data.get(l, [])) for l in LETTERS}
    filtered = {u: n for u, n in contributor_stats.items()
                if u not in ('Unknown', 'Anonymous') and n > 0}
    return {
        'total': sum(counts.values()),
        'done': sum(1 for c in counts.values() if c >= TRAIN_TARGET),
        'total_letters': 26,
        'target': TRAIN_TARGET,
        'counts': counts,
        'users': list(set(train_users.values())),
        'contributor_stats': filtered,
    }


# ---- Main ----
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == '__main__':
    host = os.environ.get('BISINDO_HOST', '127.0.0.1')
    port = int(os.environ.get('BISINDO_PORT', '5000'))
    init_classifier()
    load_existing_training_data()
    print("🚀 BISINDO Server starting…")
    print(f"   Local:   http://localhost:{port}")
    print(f"   Network: http://{get_local_ip()}:{port}")
    print(f"   Capture: POST {host}:{port}/api/sample")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
