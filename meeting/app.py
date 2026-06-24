"""
BISINDO Meeting Server - Flask-SocketIO + REST capture ingest
Real-time video meeting with BISINDO gesture recognition.

Schema (CSV, 67 cols):
  letter, image_path, split, num_hands, contributor, lm0_x..lm20_z (63)

Inference: client-side (browser). Server keeps a copy of the .pkl for
optional server-side prediction in /predict_landmarks socket event.

VPS-ready:
  - Atomic CSV writes (write → fsync → rename)
  - Periodic auto-backup (every 2000 samples)
  - Debounced auto-push (single thread, not per-request)
  - /api/stats — realtime JSON stats for dashboard
  - /api/download — discoverable CSV download
  - /api/train — trigger RF training on VPS
"""

import os
import csv
import shutil
import socket
import subprocess
import json
import threading
import tempfile
import logging
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

log = logging.getLogger("bisindo")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Optional classifier (non-fatal if MediaPipe model missing on VPS)
try:
    from src.landmark_classifier import LandmarkClassifier
except ImportError:
    LandmarkClassifier = None

# ---- Auto-push config (VPS-ready) ----
AUTO_PUSH_THRESHOLD = 500
PUSH_LOCK = threading.Lock()
PUSH_STATE_FILE = os.path.join(os.path.dirname(__file__), '..', '.push_state.json')
BACKUP_INTERVAL = 2000  # backup CSV every 2000 samples

def _load_push_state():
    if os.path.exists(PUSH_STATE_FILE):
        with open(PUSH_STATE_FILE) as f:
            return json.load(f)
    return {'last_push_count': 0, 'last_push_time': None}

def _save_push_state(state):
    tmp = PUSH_STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, PUSH_STATE_FILE)

# Debounced auto-push: single background thread, wakes every 30s
_push_event = threading.Event()
_push_thread = None

def _push_worker():
    """Single background thread that checks and pushes CSV periodically."""
    while True:
        _push_event.wait(timeout=30)  # wake every 30s or on signal
        _push_event.clear()
        csv_path = CSV_PATH
        if not os.path.exists(csv_path):
            continue
        state = _load_push_state()
        try:
            with open(csv_path) as f:
                total_rows = sum(1 for _ in f) - 1
        except:
            continue
        new_samples = total_rows - state['last_push_count']
        if new_samples < AUTO_PUSH_THRESHOLD:
            continue
        with PUSH_LOCK:
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                subprocess.run(['git', '-C', repo_root, 'add', 'dataset/landmarks_captured_v2.csv'],
                              capture_output=True, timeout=30)
                result = subprocess.run(
                    ['git', '-C', repo_root, 'diff', '--cached', '--quiet'],
                    capture_output=True, timeout=10)
                if result.returncode == 0:
                    continue  # nothing staged
                subprocess.run(['git', '-C', repo_root, 'commit', '-m',
                              f'data: auto-commit {new_samples} new samples (total {total_rows})'],
                              capture_output=True, timeout=30)
                result = subprocess.run(['git', '-C', repo_root, 'push', 'origin', 'main'],
                                      capture_output=True, timeout=120)
                if result.returncode == 0:
                    log.info(f'✅ Auto-pushed {new_samples} samples (total {total_rows})')
                    _save_push_state({
                        'last_push_count': total_rows,
                        'last_push_time': datetime.now().isoformat()
                    })
                else:
                    log.warning(f'Auto-push failed: {result.stderr.decode()[:200]}')
            except Exception as e:
                log.warning(f'Auto-push error: {e}')

def _signal_push():
    """Wake the push thread immediately."""
    _push_event.set()

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
TRAIN_TARGET = 5000  # target samples per letter

# In-memory state
training_data = {}   # letter -> [{landmarks, hand_count, contributor, from_csv}]
rooms = {}
train_users = {}
contributor_stats = {}
letter_counters = {}  # letter -> int (next index, avoids reading CSV every capture)
classifier = None


# ---- Classifier ----
def init_classifier():
    global classifier
    if LandmarkClassifier is None:
        log.warning("Classifier module not available (skip inference)")
        classifier = None
        return
    paths = {
        'model': os.path.join(MODEL_DIR, 'landmark_classifier.pkl'),
        'scaler': os.path.join(MODEL_DIR, 'landmark_classifier_scaler.pkl'),
        'labels': os.path.join(MODEL_DIR, 'landmark_classifier_labels.pkl'),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        log.info("Model files missing — running without classifier (train first)")
        classifier = None
        return
    try:
        classifier = LandmarkClassifier(paths['model'], paths['scaler'], paths['labels'])
        log.info(f"✅ Classifier loaded ({len(classifier.label_encoder.classes_)} classes)")
    except Exception as e:
        log.warning(f"Classifier load failed (non-fatal): {e}")
        classifier = None


# ---- CSV load/save (atomic, with backup) ----
def load_existing_training_data():
    training_data.clear()
    letter_counters.clear()
    if not os.path.exists(CSV_PATH):
        log.info(f"No CSV at {CSV_PATH}")
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
                letter_counters[letter] = letter_counters.get(letter, 0) + 1
                valid += 1
            except (ValueError, KeyError):
                skipped += 1
    log.info(f"✅ Loaded {valid} samples (skipped {skipped}) across {len(letter_counters)} letters")


def backup_csv():
    """Create a timestamped backup of the CSV file."""
    if not os.path.exists(CSV_PATH):
        return
    backup_dir = os.path.join(DATA_DIR, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(backup_dir, f'landmarks_captured_v2_{stamp}.csv')
    shutil.copy2(CSV_PATH, dst)
    # Keep only last 5 backups
    backups = sorted([
        os.path.join(backup_dir, f) for f in os.listdir(backup_dir)
        if f.startswith('landmarks_captured_v2_') and f.endswith('.csv')
    ])
    for old in backups[:-5]:
        os.remove(old)
    log.info(f"💾 Backup: {dst}")


# Thread-safe counter for letter indices
COUNTER_LOCK = threading.Lock()
_last_backup_count = 0

def append_row(letter, hand1, contributor, source, num_hands=2):
    """Append a single 67-col row to CSV_PATH. Atomic writes + periodic backup."""
    global _last_backup_count
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    # Fast counter from memory (no CSV re-read)
    with COUNTER_LOCK:
        next_idx = letter_counters.get(letter, 0)
        letter_counters[letter] = next_idx + 1
        total_count = sum(letter_counters.values())

    row = {
        'letter': letter,
        'image_path': f'{source}_{letter}_{next_idx}',
        'split': 'train',
        'num_hands': num_hands,
        'contributor': contributor,
    }
    for i in range(21):
        row[f'lm{i}_x'] = hand1[i * 3]
        row[f'lm{i}_y'] = hand1[i * 3 + 1]
        row[f'lm{i}_z'] = hand1[i * 3 + 2]

    # Atomic append: write to temp then append to main file
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    # Periodic backup every BACKUP_INTERVAL samples
    if total_count - _last_backup_count >= BACKUP_INTERVAL:
        _last_backup_count = total_count
        threading.Thread(target=backup_csv, daemon=True).start()

    # Signal push thread (debounced)
    _signal_push()


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

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

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
    hand2 = data.get('hand2') or []
    contributor = (data.get('contributor') or 'Anonymous').strip()[:40] or 'Anonymous'

    if letter not in LETTERS:
        return jsonify({'ok': False, 'error': f'letter must be A-Z'}), 400
    if len(hand1) != 63:
        return jsonify({'ok': False, 'error': 'hand1 must be 63 floats'}), 400
    try:
        hand1 = [float(v) for v in hand1]
        hand2 = [float(v) for v in hand2] if hand2 else []
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'hand1 must be numeric'}), 400

    # Determine actual hand count
    hand_count = 1
    if len(hand2) == 63:
        hand_count = 2

    append_row(letter, hand1, contributor, source='web', num_hands=hand_count)
    total = sum(len(v) for v in training_data.values())
    print(f"📥 [{contributor}] Letter {letter} → CSV written (total: {total} samples)")
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


@app.route('/api/stats')
def api_stats():
    """Realtime stats for dashboard (JSON)."""
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH) as f:
                total_rows = sum(1 for _ in f) - 1
        except:
            total_rows = 0
    else:
        total_rows = 0

    counts = {l: len(training_data.get(l, [])) for l in LETTERS}
    return jsonify({
        'ok': True,
        'csv_rows': total_rows,
        'csv_size_mb': round(os.path.getsize(CSV_PATH) / 1024 / 1024, 2) if os.path.exists(CSV_PATH) else 0,
        'memory_samples': sum(counts.values()),
        'counts': counts,
        'target_per_letter': TRAIN_TARGET,
        'classifier_loaded': classifier is not None,
        'model_exists': os.path.exists(os.path.join(MODEL_DIR, 'landmark_classifier.pkl')),
    })


@app.route('/api/download')
def api_download():
    """Download CSV file."""
    if not os.path.exists(CSV_PATH):
        return jsonify({'ok': False, 'error': 'No data yet'}), 404
    return send_file(CSV_PATH, as_attachment=True, download_name='landmarks_captured_v2.csv')


@app.route('/api/train', methods=['POST'])
def api_train():
    """Trigger RF training on VPS (skip MLP/TF.js to save RAM)."""
    if not os.path.exists(CSV_PATH):
        return jsonify({'ok': False, 'error': 'No CSV data'}), 400

    def train_async():
        try:
            log.info("🧠 Training started (RF only, VPS-optimized)...")
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import pickle

            df = pd.read_csv(CSV_PATH, low_memory=False)
            cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
            X1 = df[cols].astype(np.float32).values
            X1 = np.nan_to_num(X1, nan=0.0)
            X2 = np.zeros_like(X1)
            X = np.concatenate([X1, X2], axis=1)
            y = df["letter"].astype(str).values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=200,  # smaller for VPS
                max_depth=None,
                min_samples_split=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            rf.fit(X_train_s, y_train)
            pred = rf.predict(X_test_s)
            acc = accuracy_score(y_test, pred)

            labels = sorted(np.unique(y_train).tolist())
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(os.path.join(MODEL_DIR, 'landmark_classifier.pkl'), 'wb') as f:
                pickle.dump(rf, f)
            with open(os.path.join(MODEL_DIR, 'landmark_classifier_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            with open(os.path.join(MODEL_DIR, 'landmark_classifier_labels.pkl'), 'wb') as f:
                pickle.dump({"classes": labels, "n_classes": len(labels)}, f)

            log.info(f"✅ Training done! Accuracy: {acc:.4f}")
            init_classifier()  # reload
        except Exception as e:
            log.error(f"Training failed: {e}")

    threading.Thread(target=train_async, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Training started in background'})


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

    # Start debounced push thread
    _push_thread = threading.Thread(target=_push_worker, daemon=True, name='auto-push')
    _push_thread.start()

    init_classifier()
    load_existing_training_data()
    log.info("🚀 BISINDO Server starting…")
    log.info(f"   Local:   http://localhost:{port}")
    log.info(f"   Network: http://{get_local_ip()}:{port}")
    log.info(f"   Capture: POST {host}:{port}/api/sample")
    log.info(f"   Stats:   GET  {host}:{port}/api/stats")
    log.info(f"   Train:   POST {host}:{port}/api/train")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
