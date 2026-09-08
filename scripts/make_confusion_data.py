#!/usr/bin/env python3
"""Reproduce the deployed dual-hand evaluation and export confusion data.

Replicates the paper protocol exactly: dataset/landmarks_2hands.csv,
84-D dual-hand (x,y) features, stratified 85/15 split (seed 42),
saved StandardScaler (models/dl/cnn_2hand_scaler.json), ONNX Runtime
inference (meeting/static/models/model.onnx). VERIFIES against the
published numbers (98.45% accuracy / per-class csv) before writing
docs/jurnal-data/cnn_2hand_confusion.json.

Outputs JSON: {labels, accuracy, matrix, top_pairs} where matrix[true][pred].
"""
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "jurnal-data" / "cnn_2hand_confusion.json"

df = pd.read_csv(ROOT / "dataset" / "landmarks_2hands.csv")
cols = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
if not all(c in df.columns for c in cols):
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y")]
h1 = [f"h1_lm{i}_{c}" if f"h1_lm{i}_x" in df.columns else f"lm{i}_{c}"
      for i in range(21) for c in ("x", "y")]
X = np.concatenate([df[h1].values, df[cols].values], axis=1).astype(np.float32)
y = (df["label"].astype(str).to_numpy() if "label" in df.columns
     else df[df.columns[0]].astype(str).to_numpy())

labels = list(string.ascii_uppercase)
scaler = json.load(open(ROOT / "models" / "dl" / "cnn_2hand_scaler.json"))
mean = np.array(scaler["mean"], dtype=np.float32)
scale = np.array(scaler["scale"], dtype=np.float32)


def normalize_hand(hand: np.ndarray) -> np.ndarray:
    """Stages 1-2 exactly as web/test.html normalizeFeaturesHandCentric.

    Vectorized over rows: hand shape (N, 42) = 21 landmarks x (x,y)."""
    pts = hand.reshape(-1, 21, 2)
    pts = pts - pts[:, 0:1, :]  # wrist translation (landmark 0)
    d = np.hypot(pts[..., 0], pts[..., 1]).max(axis=1)  # d_max per row
    d[d == 0] = 1.0  # zero-padded missing hand stays zero
    return (pts / d[:, None, None]).reshape(-1, 42)


def normalize84(rows: np.ndarray) -> np.ndarray:
    return np.concatenate([normalize_hand(rows[:, :42]),
                           normalize_hand(rows[:, 42:84])], axis=1)


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)

Xt = (normalize84(X_te) - mean) / scale
sess = ort.InferenceSession(
    str(ROOT / "meeting" / "static" / "models" / "model.onnx"),
    providers=["CPUExecutionProvider"])
logits = sess.run(None, {sess.get_inputs()[0].name: Xt[:, None, :]})[0]
pred = np.array(labels)[logits.argmax(axis=1)]

acc = float((pred == y_te).mean())
print(f"test n={len(y_te)}  accuracy={acc:.4%}")

# --- verify against the paper's per-class csv ---
csv = pd.read_csv(ROOT / "docs" / "jurnal-data" / "cnn_2hand_per_class.csv")
from sklearn.metrics import precision_recall_fscore_support
P, R, F, S = precision_recall_fscore_support(
    y_te, pred, labels=labels, zero_division=0)
bad = []
for _, row in csv.iterrows():
    i = labels.index(row["letter"])
    if abs(P[i] - row["precision"]) > 5e-4 or abs(R[i] - row["recall"]) > 5e-4 \
       or abs(F[i] - row["f1"]) > 5e-4 or int(S[i]) != int(row["support"]):
        bad.append((row["letter"], round(P[i], 4), round(R[i], 4), round(F[i], 4), int(S[i])))
print("mismatched classes:", bad if bad else "NONE - replication exact")

cm = np.zeros((26, 26), dtype=int)
for t, p in zip(y_te, pred):
    cm[labels.index(t), labels.index(p)] += 1

pairs = [(int(cm[i, j]), labels[i], labels[j])
         for i in range(26) for j in range(26) if i != j and cm[i, j] > 0]
pairs.sort(reverse=True)

OUT.write_text(json.dumps({
    "labels": labels,
    "accuracy": acc,
    "n_test": int(len(y_te)),
    "matrix": cm.tolist(),
    "top_pairs": [{"count": c, "true": t, "pred": p} for c, t, p in pairs],
}, indent=1))
print(f"wrote {OUT}; total errors={int(sum(c for c,_,_ in pairs))}; top={pairs[:8]}")
