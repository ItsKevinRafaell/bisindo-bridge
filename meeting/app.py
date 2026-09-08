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
  - CSV is NOT git-tracked (too large; use /api/download instead)
  - /api/stats — realtime JSON stats for dashboard
  - /api/download — discoverable CSV download
  - /api/train — trigger RF training on VPS
"""

import os
import csv
import shutil
import socket
import subprocess
import secrets
import json
import threading
import tempfile
import logging
import atexit
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

# ---- Backup config (VPS-ready) ----
BACKUP_INTERVAL = 2000  # backup CSV every 2000 samples

# ---- Config ----
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('BISINDO_SECRET_KEY') or secrets.token_urlsafe(32)
_cors_origins = [o.strip() for o in os.environ.get('BISINDO_CORS_ORIGINS', '*').split(',') if o.strip()]
CORS(app, origins=_cors_origins)
socketio = SocketIO(app, cors_allowed_origins=_cors_origins, async_mode='threading')

HEADER = ["letter", "image_path", "split", "num_hands", "contributor"]
HEADER += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
CSV_PATH = os.path.join(DATA_DIR, 'landmarks_captured_v2.csv')
COUNTERS_PATH = os.path.join(DATA_DIR, 'counters.json')

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
TRAIN_TARGET = 5000  # target samples per letter

# In-memory state
training_data = {}   # letter -> [{landmarks, hand_count, contributor, from_csv}]
rooms = {}
ROOMS_LOCK = threading.RLock()
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


# ---- Counter persistence ----
def save_counters():
    """Save current counters to JSON file (for debugging/monitoring)."""
    try:
        with open(COUNTERS_PATH, 'w') as f:
            json.dump(letter_counters, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save counters: {e}")

def rebuild_counters():
    """Rebuild counters from CSV. Call this if counters get out of sync."""
    letter_counters.clear()
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, 'r') as f:
                for row in csv.DictReader(f):
                    letter = row.get('letter', '').upper()
                    if letter in LETTERS:
                        letter_counters[letter] = letter_counters.get(letter, 0) + 1
        except Exception as e:
            log.error(f"Counter rebuild failed: {e}")
    save_counters()
    return letter_counters

# ---- CSV load/save (atomic, with backup) ----
def load_existing_training_data():
    training_data.clear()
    letter_counters.clear()  # Always rebuild from CSV
    if not os.path.exists(CSV_PATH):
        log.info(f"No CSV at {CSV_PATH}")
        return
    valid, skipped = 0, 0
    with open(CSV_PATH, 'r') as f:
        for row in csv.DictReader(f):
            try:
                # Skip rows with missing/None landmark values
                if any(row.get(f'lm{i}_{c}') is None or row.get(f'lm{i}_{c}') == ''
                       for i in range(21) for c in ('x', 'y', 'z')):
                    skipped += 1
                    continue

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
    save_counters()  # Save accurate counters


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

    # Save counters to JSON (every 10 samples to reduce I/O)
    if next_idx % 10 == 0:
        threading.Thread(target=save_counters, daemon=True).start()

    # Periodic backup every BACKUP_INTERVAL samples
    if total_count - _last_backup_count >= BACKUP_INTERVAL:
        _last_backup_count = total_count
        threading.Thread(target=backup_csv, daemon=True).start()


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

@app.route('/api/turn')
def api_turn():
    """Return TURN credentials for WebRTC relay.

    TURN credentials are read from the environment (never hard-coded in the
    client). Supported providers:
      - Cloudflare Calls: BISINDO_TURN_URL (e.g. turn:turn.cloudflare.com:3478),
        BISINDO_TURN_USERNAME, BISINDO_TURN_CREDENTIAL
      - Any RFC-5766 TURN: same vars, or BISINDO_TURN_URL as a comma list.

    If unset, returns an empty list and the client falls back to built-in ICE_SERVERS (STUN + public free TURN)
    (works for peers that can traverse NAT without relay).
    """
    urls = (os.environ.get('BISINDO_TURN_URL') or '').strip()
    if not urls:
        return jsonify({'iceServers': []})
    servers = []
    for u in urls.split(','):
        u = u.strip()
        if not u:
            continue
        entry = {'urls': u}
        user = os.environ.get('BISINDO_TURN_USERNAME')
        cred = os.environ.get('BISINDO_TURN_CREDENTIAL')
        if user:
            entry['username'] = user
        if cred:
            entry['credential'] = cred
        servers.append(entry)
    return jsonify({'iceServers': servers})


@app.route('/api/health')
def api_health():
    counts = {l: letter_counters.get(l, 0) for l in LETTERS}
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

    return jsonify({'ok': True, 'letter': letter, 'count': letter_counters.get(letter, 0)})


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

    counts = {l: letter_counters.get(l, 0) for l in LETTERS}
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


def _remove_sid_from_rooms(sid, broadcast=True):
    """Drop sid from every room user list. Prevents ghost/duplicate.
    Also leaves SocketIO room and prunes empty rooms (audit fix)."""
    to_clean = []
    with ROOMS_LOCK:
        for room_id, room in list(rooms.items()):
            users = room.get('users') or []
            kept = []
            removed = []
            for u in users:
                if u.get('sid') == sid:
                    removed.append(u)
                else:
                    kept.append(u)
            if not removed:
                continue
            room['users'] = kept
            to_clean.append((room_id, removed, list(kept)))
            if not room['users']:
                rooms.pop(room_id, None)
    for room_id, removed, kept in to_clean:
        try:
            leave_room(room_id)
        except Exception:
            pass
        if broadcast:
            for u in removed:
                try:
                    emit('user_left', {
                        'username': u.get('username', 'Anonymous'),
                        'sid': sid,
                        'users': kept,
                    }, room=room_id)
                except Exception:
                    pass


def _add_user_to_room(room_id, sid, username):
    """Atomic add with sid dedup only (NOT username — same username = different person)."""
    with ROOMS_LOCK:
        room = rooms.setdefault(room_id, {'users': [], 'history': []})
        users_list = room['users']
        # Dedup by sid ONLY — allow same username from different devices
        users_list[:] = [u for u in users_list if u.get('sid') != sid]
        entry = {'username': username, 'sid': sid}
        users_list.append(entry)
        return list(users_list)


def _remove_user_from_room(room_id, sid):
    with ROOMS_LOCK:
        if room_id not in rooms:
            return []
        rooms[room_id]['users'] = [
            u for u in rooms[room_id]['users'] if u.get('sid') != sid
        ]
        kept = list(rooms[room_id]['users'])
        if not kept:
            rooms.pop(room_id, None)
        return kept


def _get_room_users(room_id):
    with ROOMS_LOCK:
        return list(rooms.get(room_id, {}).get('users', []))


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    _remove_sid_from_rooms(sid, broadcast=True)
    # Audit fix: train_users leak
    try:
        train_users.pop(sid, None)
    except Exception:
        pass


@socketio.on('join')
def on_join(data):
    room_id = (data.get('room') or 'default').strip()
    username = (data.get('username') or 'Anonymous').strip() or 'Anonymous'
    sid = request.sid
    join_room(room_id)
    # Atomic dedup by both sid and username — handles reconnect with new sid
    all_users = _add_user_to_room(room_id, sid, username)
    # Send full roster only to the joiner (avoids stomping others' UI mid-call)
    emit('room_info', {'room': room_id, 'users': all_users, 'selfSid': sid})
    emit('user_joined', {
        'username': username,
        'sid': sid,
        'users': all_users,
    }, room=room_id, include_self=False)


@socketio.on('leave')
def on_leave(data):
    room_id = (data.get('room') or 'default').strip()
    username = data.get('username', 'Anonymous')
    sid = request.sid
    leave_room(room_id)
    kept = _remove_user_from_room(room_id, sid)
    emit('user_left', {'username': username, 'sid': sid, 'users': kept}, room=room_id)

@socketio.on('text_message')
def on_text_message(data):
    room_id = data.get('room', 'default')
    username = data.get('username', 'Anonymous')
    emit('chat_message', {'username': username, 'message': data.get('message', '')}, room=room_id)

# ---- SocketIO: WebRTC signaling ----
@socketio.on('webrtc_offer')
def on_webrtc_offer(data):
    """Relay SDP offer from one peer to another."""
    emit('webrtc_offer', {
        'fromSid': request.sid,
        'fromUsername': data.get('username'),
        'sdp': data['sdp']
    }, to=data['targetSid'])


@socketio.on('webrtc_answer')
def on_webrtc_answer(data):
    """Relay SDP answer from one peer to another."""
    emit('webrtc_answer', {
        'fromSid': request.sid,
        'sdp': data['sdp']
    }, to=data['targetSid'])


@socketio.on('webrtc_ice')
def on_webrtc_ice(data):
    """Relay ICE candidate between peers."""
    emit('webrtc_ice', {
        'fromSid': request.sid,
        'candidate': data['candidate']
    }, to=data['targetSid'])


# ---- SocketIO: hand raise + reactions ----
@socketio.on('hand_raise')
def on_hand_raise(data):
    emit('hand_raise', {
        'username': data.get('username'),
        'sid': request.sid
    }, room=data.get('room'), include_self=False)


@socketio.on('hand_lower')
def on_hand_lower(data):
    emit('hand_lower', {
        'username': data.get('username'),
        'sid': request.sid
    }, room=data.get('room'), include_self=False)


@socketio.on('reaction')
def on_reaction(data):
    emit('reaction', {
        'username': data.get('username'),
        'emoji': data.get('emoji'),
        'sid': request.sid
    }, room=data.get('room'), include_self=False)


@socketio.on('peer_state')
def on_peer_state(data):
    """Broadcast mic/cam state changes to peers."""
    emit('peer_state', {
        'sid': request.sid,
        'username': data.get('username'),
        'mic': data.get('mic'),
        'cam': data.get('cam')
    }, room=data.get('room'), include_self=False)


@socketio.on('bisindo_state')
def on_bisindo_state(data):
    """Broadcast BISINDO on/off indicator for peer cards (green dot)."""
    room_id = data.get('room') or 'default'
    emit('bisindo_state', {
        'username': data.get('username', 'Anonymous'),
        'sid': request.sid,
        'enabled': bool(data.get('enabled')),
    }, room=room_id, include_self=False)


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


# ---- SocketIO: sentence building ----
@socketio.on('letter_committed')
def on_letter_committed(data):
    """Broadcast a committed letter to the room."""
    room_id = data.get('room', 'default')
    emit('letter_committed', {
        'username': data.get('username', 'unknown'),
        'letter': data.get('letter', ''),
        't_sent': data.get('t_sent'),
    }, room=room_id, include_self=False)


@socketio.on('space_inserted')
def on_space_inserted(data):
    """Broadcast a space insertion to the room."""
    room_id = data.get('room', 'default')
    emit('space_inserted', {
        'username': data.get('username', 'unknown'),
    }, room=room_id, include_self=False)


@socketio.on('sentence_completed')
def on_sentence_completed(data):
    """Broadcast a completed sentence to the room."""
    room_id = data.get('room', 'default')
    sentence = data.get('sentence', '')
    if sentence:
        emit('sentence_broadcast', {
            'username': data.get('username', 'unknown'),
            'sentence': sentence,
            'timestamp': datetime.now().isoformat(),
        }, room=room_id, include_self=False)


@socketio.on('tts_request')
def on_tts_request(data):
    """Generate TTS audio and broadcast to room."""
    room_id = data.get('room', 'default')
    text = data.get('text', '')
    if not text:
        return
    try:
        from src.tts_engine import TTSEngine
        tts = TTSEngine()
        audio_data = tts.text_to_audio(text)
        if audio_data:
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            emit('tts_audio', {
                'username': data.get('username', 'unknown'),
                'text': text,
                'audio': f'data:audio/mp3;base64,{audio_b64}',
            }, room=room_id)
    except Exception as e:
        log.error(f"TTS error: {e}")


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
    counts = {l: letter_counters.get(l, 0) for l in LETTERS}
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


def start_cloudflared_tunnel(port, enabled=True):
    """Start Cloudflare Quick Tunnel for public HTTPS access."""
    if not enabled:
        return None

    try:
        proc = subprocess.Popen(
            ['cloudflared', 'tunnel', '--url', f'http://localhost:{port}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Read stderr to find tunnel URL (cloudflared outputs to stderr)
        import threading
        tunnel_url = None

        def read_output():
            nonlocal tunnel_url
            for line in proc.stderr:
                if '.trycloudflare.com' in line:
                    # Extract URL from line like: "https://xxx-xxx.trycloudflare.com | ..."
                    import re
                    match = re.search(r'https://[^\s|]+', line)
                    if match:
                        tunnel_url = match.group(0)
                        log.info(f"   🌐 Tunnel: {tunnel_url}")
                        break

        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()

        # Give it a moment to start
        import time
        time.sleep(2)

        return proc, tunnel_url
    except FileNotFoundError:
        log.warning("cloudflared not found, skipping tunnel")
        return None
    except Exception as e:
        log.warning(f"Tunnel failed: {e}")
        return None


def start_sfu_server(sfu_port=4501):
    """Start the SFU (Selective Forwarding Unit) Node.js server."""
    sfu_dir = os.path.join(BASE_DIR, 'sfu')
    sfu_script = os.path.join(sfu_dir, 'server.js')
    if not os.path.exists(sfu_script):
        log.warning(f"SFU script not found at {sfu_script}, skipping SFU")
        return None
    try:
        proc = subprocess.Popen(
            ['node', sfu_script],
            cwd=sfu_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        log.info(f"   SFU:     ws://localhost:{sfu_port}")

        def read_sfu_output():
            for line in proc.stdout:
                log.info(f"[SFU] {line.rstrip()}")

        threading.Thread(target=read_sfu_output, daemon=True).start()
        return proc
    except FileNotFoundError:
        log.warning("node not found, cannot start SFU server")
        return None
    except Exception as e:
        log.warning(f"SFU start failed: {e}")
        return None


if __name__ == '__main__':
    host = os.environ.get('BISINDO_HOST', '127.0.0.1')
    port = int(os.environ.get('BISINDO_PORT', '4500'))
    sfu_port = int(os.environ.get('BISINDO_SFU_PORT', '4501'))
    enable_tunnel = os.environ.get('BISINDO_TUNNEL', '1') == '1'

    init_classifier()
    load_existing_training_data()
    log.info("🚀 BISINDO Server starting…")
    log.info(f"   Local:   http://localhost:{port}")
    log.info(f"   Network: http://{get_local_ip()}:{port}")
    log.info(f"   Capture: POST {host}:{port}/api/sample")
    log.info(f"   Stats:   GET  {host}:{port}/api/stats")
    log.info(f"   Train:   POST {host}:{port}/api/train")

    # Start SFU server (only if explicitly enabled; needs a public UDP-reachable
    # host, which a NAT LXC VPS is not — leave off here to save RAM/disk).
    sfu_proc = None
    if os.environ.get('BISINDO_ENABLE_SFU') == '1':
        sfu_proc = start_sfu_server(sfu_port)

    # Start Cloudflare tunnel
    tunnel_result = start_cloudflared_tunnel(port, enabled=enable_tunnel)
    tunnel_proc = None
    if tunnel_result:
        tunnel_proc, tunnel_url = tunnel_result

    # Cleanup on exit
    def cleanup():
        if sfu_proc and sfu_proc.poll() is None:
            log.info("Shutting down SFU server...")
            sfu_proc.terminate()
        if tunnel_proc and tunnel_proc.poll() is None:
            log.info("Shutting down Cloudflare tunnel...")
            tunnel_proc.terminate()
    atexit.register(cleanup)

    _socketio_kwargs = {'host': host, 'port': port, 'debug': False}
    # This deployment uses the Werkzeug server as a lightweight signaling
    # relay (SocketIO only). Set BISINDO_ALLOW_UNSAFE_WERKZEUG=0 to refuse.
    if os.environ.get('BISINDO_ALLOW_UNSAFE_WERKZEUG', '1') == '1':
        _socketio_kwargs['allow_unsafe_werkzeug'] = True
    socketio.run(app, **_socketio_kwargs)
