#!/usr/bin/env python3
"""BISINDO Webcam Inference - Smart Mode."""
import os, json, argparse, time, numpy as np, cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Letters that need 2 hands
TWO_HAND_LETTERS = set("ABDFGHJKMNPQSTWXY")

class CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64*8, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
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

    def predict(self, landmarks, num_hands=1):
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

        # Post-processing: C vs S disambiguation
        # C = 1 hand, S = 2 hands
        if pred_letter == 'S' and num_hands == 1:
            pred_letter = 'C'
        elif pred_letter == 'C' and num_hands == 2:
            # Check if S has high probability
            s_idx = self.labels.index('S') if 'S' in self.labels else -1
            if s_idx >= 0 and probs[s_idx] > 0.3:
                pred_letter = 'S'

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
        return None, 0

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
    return raw, len(res.hand_landmarks)

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
    cv2.putText(frame, info, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/dl/cnn_2hand_model.pt")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    base = args.model.replace("_model.pt", "")
    clf = Classifier(args.model, f"{base}_scaler.json", f"{base}_labels.json")
    print(f"Loaded: {len(clf.labels)} classes")
    print(f"2-hand letters: {sorted(TWO_HAND_LETTERS)}")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t, fps = time.time(), 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fps += 1
        if time.time() - t >= 1:
            t, fps = time.time(), 0

        lm, num_hands = extract(frame, clf.detector)
        if lm:
            r = clf.predict(lm, num_hands)
            frame = draw(frame, lm, num_hands, r)

            # Show prediction
            txt = f"{r['letter']} ({r['confidence']:.2f})"
            color = (0, 255, 0) if r['confidence'] > 0.8 else (0, 255, 255)
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            print(f"\r{r['letter']}: {r['confidence']:.3f}", end="", flush=True)
        else:
            cv2.putText(frame, "No hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("BISINDO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()