#!/usr/bin/env python3
"""BISINDO Webcam Inference - Smart Mode with Sentence Building."""
import os, json, argparse, time, numpy as np, cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import sys
sys.path.append('src')
from gesture_detector import get_hand_state
from sentence_builder import SentenceBuilder

# Letters that need 2 hands
TWO_HAND_LETTERS = set("ABDFGHJKMNPQSTWXY")

class CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class Classifier:
    def __init__(self, model_path, scaler_path, labels_path):
        # Auto-detect input_dim from scaler
        with open(scaler_path) as f:
            d = json.load(f)
        input_dim = len(d["mean"])

        self.mean = np.array(d["mean"])
        self.scale = np.array(d["scale"])
        self.input_dim = input_dim

        with open(labels_path) as f:
            self.labels = json.load(f)

        self.model = CNN(input_dim=input_dim, num_classes=len(self.labels))
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        base = python.BaseOptions(model_asset_path=self._hand_model())
        opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(opts)

    def _hand_model(self):
        p = os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task")
        return p if os.path.exists(p) else None

    def predict(self, landmarks, num_hands=1, confidence_threshold=0.5):
        x = np.array(landmarks, dtype=np.float32)
        x = np.nan_to_num(x)

        # Hand-centric normalization (same as training)
        x = normalize_features(x)

        # StandardScaler normalization
        x = ((x - self.mean) / self.scale).reshape(1, 1, -1)
        out = self.model(torch.FloatTensor(x))
        probs = F.softmax(out, dim=1).squeeze()
        conf, pred = torch.max(probs, 0)
        pred_letter = self.labels[pred.item()]

        # Confidence threshold: if too low, return None
        if conf.item() < confidence_threshold:
            return None

        # Post-processing: disambiguate letters that look similar
        # but differ in hand count
        # 1-hand letters: C, E, I, L, O, R, U, V, Y, Z
        # 2-hand letters: A, B, D, F, G, H, J, K, M, N, P, Q, S, T, W, X
        ONE_HAND = set("CEILORUVYZ")
        TWO_HAND = set("ABDFGHJKMNPQSTWX")

        if pred_letter in TWO_HAND and num_hands == 1:
            # Model predicts 2-hand letter but only 1 hand detected
            # Find best 1-hand alternative
            for i in range(len(self.labels)):
                if self.labels[i] in ONE_HAND and probs[i] > 0.1:
                    pred_letter = self.labels[i]
                    break

        elif pred_letter in ONE_HAND and num_hands == 2:
            # Model predicts 1-hand letter but 2 hands detected
            # Find best 2-hand alternative
            for i in range(len(self.labels)):
                if self.labels[i] in TWO_HAND and probs[i] > 0.1:
                    pred_letter = self.labels[i]
                    break

        return {
            "letter": pred_letter,
            "confidence": conf.item(),
            "probabilities": {self.labels[i]: probs[i].item() for i in range(len(self.labels))}
        }

def normalize_hand(landmarks):
    """Normalize a single hand: translate to wrist, scale by max distance.
    landmarks: (21, 2) array. Returns same shape."""
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    dists = np.linalg.norm(centered, axis=1)
    max_dist = dists.max()
    if max_dist > 0:
        centered = centered / max_dist
    return centered


def normalize_features(flat):
    """Apply hand-centric normalization to 84-feature flat array."""
    x = np.array(flat, dtype=np.float32)
    h1 = normalize_hand(x[:42].reshape(21, 2)).flatten()
    h2 = normalize_hand(x[42:].reshape(21, 2)).flatten()
    return np.concatenate([h1, h2])


def extract(frame, detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_img)
    if not res.hand_landmarks:
        return None, 0, None

    def hand_to_xy(hand):
        return [p for pt in hand for p in [pt.x, pt.y]]

    # Hand 1
    hand1 = hand_to_xy(res.hand_landmarks[0])

    # Hand 2 (if exists, else zeros)
    if len(res.hand_landmarks) >= 2:
        hand2 = hand_to_xy(res.hand_landmarks[1])
    else:
        hand2 = [0.0] * 42

    raw = hand1 + hand2
    # Return raw landmarks for gesture detection (first hand only, 21x3 format)
    raw_lms = [[pt.x, pt.y, pt.z] for pt in res.hand_landmarks[0]]
    return raw, len(res.hand_landmarks), raw_lms

def smart_predict(clf, lm, num_hands):
    """Smart prediction considering 1 vs 2 hands."""
    if num_hands == 1:
        return clf.predict(lm)

    # 2 hands - get predictions for both
    result1 = clf.predict(lm)  # This is average, not accurate

    # Alternative: check if prediction matches 2-hand letter
    # For now, just return the prediction
    return clf.predict(lm)

def draw(frame, landmarks, num_hands, prediction=None):
    h, w = frame.shape[:2]
    # Hand 1: first 42 values (21 landmarks x 2)
    pts1 = [(int(landmarks[i]*w), int(landmarks[i+1]*h)) for i in range(0, 42, 2)]
    # Hand 2: next 42 values
    pts2 = [(int(landmarks[i]*w), int(landmarks[i+1]*h)) for i in range(42, 84, 2)]

    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

    # Draw hand 1 (green)
    for s, e in conns:
        cv2.line(frame, pts1[s], pts1[e], (0, 255, 0), 2)
    for pt in pts1:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # Draw hand 2 (blue) if present
    if num_hands == 2:
        for s, e in conns:
            cv2.line(frame, pts2[s], pts2[e], (255, 0, 0), 2)
        for pt in pts2:
            cv2.circle(frame, pt, 4, (255, 0, 0), -1)

    info = f"{num_hands} hand(s)"
    if prediction:
        info += f" -> {prediction['letter']} ({prediction['confidence']:.2f})"
    else:
        info += " -> ? (low confidence)"
    cv2.putText(frame, info, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/dl/cnn_2hand_model.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--connect", help="Connect to meeting server (e.g. http://localhost:4500)")
    p.add_argument("--room", default="default", help="Room name for meeting server")
    p.add_argument("--username", default="User", help="Username for meeting server")
    args = p.parse_args()

    base = args.model.replace("_model.pt", "")
    clf = Classifier(args.model, f"{base}_scaler.json", f"{base}_labels.json")
    print(f"Loaded: {len(clf.labels)} classes")
    print(f"2-hand letters: {sorted(TWO_HAND_LETTERS)}")
    print("Press 'q' to quit")

    # Initialize sentence builder
    builder = SentenceBuilder(stability_ms=500, space_no_hand_ms=2000, enter_no_hand_ms=4000)

    def on_letter(letter, word):
        print(f"\n[Letter] {letter} -> word: {word}")
        if sio:
            sio.emit('letter_committed', {
                'room': args.room,
                'username': args.username,
                'letter': letter
            })

    def on_word(word, words):
        print(f"\n[Space] word: {word}")
        if sio:
            sio.emit('space_inserted', {
                'room': args.room,
                'username': args.username
            })

    def on_sentence(sentence, sentences):
        print(f"\n[Sentence] {sentence}")
        if sio:
            sio.emit('sentence_completed', {
                'room': args.room,
                'username': args.username,
                'sentence': sentence
            })

    def on_gesture(gesture):
        print(f"\n[Gesture] {gesture}")

    builder.on_letter_commit = on_letter
    builder.on_word_complete = on_word
    builder.on_sentence_complete = on_sentence
    builder.on_gesture = on_gesture

    # Connect to meeting server if requested
    sio = None
    if args.connect:
        try:
            import socketio
            sio = socketio.Client()

            @sio.event
            def connect():
                sio.emit('join', {'room': args.room, 'username': args.username})
                print(f"Connected to meeting: {args.connect}, room: {args.room}")

            @sio.on('sentence_broadcast')
            def on_broadcast(data):
                if data.get('username') != args.username:
                    print(f"\n[Remote] {data['username']}: {data['sentence']}")

            sio.connect(args.connect)
        except ImportError:
            print("python-socketio not installed, skipping meeting connection")
        except Exception as e:
            print(f"Failed to connect: {e}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t, fps = time.time(), 0
    gesture_start = None
    last_gesture = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        fps += 1
        if time.time() - t >= 1:
            t, fps = time.time(), 0

        lm, num_hands, raw_lms = extract(frame, clf.detector)

        # Detect gesture
        gesture = None
        if raw_lms:
            raw_gesture = get_hand_state(raw_lms)
            now = time.time()
            if raw_gesture != last_gesture:
                last_gesture = raw_gesture
                gesture_start = now
            # Confirm gesture after 0.8s
            if raw_gesture in ['fist', 'palm'] and (now - gesture_start >= 0.8):
                gesture = raw_gesture

        hand_detected = lm is not None

        if lm:
            r = clf.predict(lm, num_hands)
            frame = draw(frame, lm, num_hands, r)

            if r is None:
                cv2.putText(frame, "?", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                print(f"\r?: low confidence", end="", flush=True)
                # Feed to sentence builder
                builder.feed(None, hand_detected, gesture)
            else:
                txt = f"{r['letter']} ({r['confidence']:.2f})"
                color = (0, 255, 0) if r['confidence'] > 0.8 else (0, 255, 255)
                cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                print(f"\r{r['letter']}: {r['confidence']:.3f}", end="", flush=True)
                # Feed to sentence builder
                builder.feed(r, hand_detected, gesture)

                # Emit to meeting server
                if sio and builder.current_word:
                    pass  # Will emit on space/enter
        else:
            cv2.putText(frame, "No hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Feed to sentence builder (no hand detected)
            builder.feed(None, False, None)

        # Display sentence state
        state = builder.get_state()
        building = state['building_sentence']
        if building:
            cv2.putText(frame, f"> {building}_", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display gesture indicator
        if gesture == 'fist':
            cv2.putText(frame, "SPACE", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif gesture == 'palm':
            cv2.putText(frame, "ENTER", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("BISINDO", frame)

        # Check for manual space/enter
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            builder.manual_space()
        elif key == 13:  # Enter key
            builder.manual_enter()
            # Emit sentence to meeting server
            if sio and builder.sentences:
                sio.emit('sentence_completed', {
                    'room': args.room,
                    'username': args.username,
                    'sentence': builder.sentences[-1]
                })

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()