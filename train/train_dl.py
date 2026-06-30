#!/usr/bin/env python3
"""Train CNN with landmark dataset."""
import os, json, argparse, logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "dl")


def normalize_hand(hand_coords):
    """Normalize single hand: translate to wrist, scale by hand size."""
    wrist = hand_coords[0]  # First landmark is wrist
    centered = hand_coords - wrist

    # Scale by max distance from wrist (hand size)
    distances = np.linalg.norm(centered, axis=1)
    max_dist = distances.max()
    if max_dist > 0:
        centered = centered / max_dist

    return centered


def normalize_features(X, input_dim):
    """Apply hand-centric normalization to make model distance-invariant."""
    X_norm = X.copy()

    if input_dim == 84:
        # 2-hand xy format
        for i in range(X.shape[0]):
            # Hand 1 (first 42 values)
            h1 = X[i, :42].reshape(21, 2)
            h1_norm = normalize_hand(h1)
            X_norm[i, :42] = h1_norm.flatten()

            # Hand 2 (next 42 values)
            h2 = X[i, 42:].reshape(21, 2)
            if np.any(h2 != 0):  # Only normalize if hand 2 exists
                h2_norm = normalize_hand(h2)
                X_norm[i, 42:] = h2_norm.flatten()

    elif input_dim == 63:
        # 1-hand xyz format
        for i in range(X.shape[0]):
            hand = X[i].reshape(21, 3)
            hand_norm = normalize_hand(hand)
            X_norm[i] = hand_norm.flatten()

    elif input_dim == 126:
        # 2-hand xyz format
        for i in range(X.shape[0]):
            h1 = X[i, :63].reshape(21, 3)
            h1_norm = normalize_hand(h1)
            X_norm[i, :63] = h1_norm.flatten()

            h2 = X[i, 63:].reshape(21, 3)
            if np.any(h2 != 0):
                h2_norm = normalize_hand(h2)
                X_norm[i, 63:] = h2_norm.flatten()

    return X_norm


def load_data(path):
    log.info(f"Loading {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  rows={len(df)}, letters={df['letter'].nunique()}")

    # Auto-detect column format
    cols_2hand_xyz = [f"h1_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    cols_2hand_xy = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
    cols_1hand_xyz = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]

    if all(c in df.columns for c in cols_2hand_xyz):
        # 2-hand xyz format: h1_lm* + h2_lm* (126 features)
        h1_cols = cols_2hand_xyz
        h2_cols = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
        X1 = df[h1_cols].values.astype(np.float32)
        X2 = df[h2_cols].values.astype(np.float32)
        X = np.nan_to_num(np.concatenate([X1, X2], axis=1))
        input_dim = 126
        log.info("  format: 2-hand xyz (126 features)")
    elif all(c in df.columns for c in cols_2hand_xy):
        # 2-hand xy format: lm* + h2_lm* (84 features)
        h1_cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y")]
        h2_cols = cols_2hand_xy
        X1 = df[h1_cols].values.astype(np.float32)
        X2 = df[h2_cols].values.astype(np.float32)
        X = np.nan_to_num(np.concatenate([X1, X2], axis=1))
        input_dim = 84
        log.info("  format: 2-hand xy (84 features)")
    else:
        # 1-hand xyz format: lm* (63 features)
        X = np.nan_to_num(df[cols_1hand_xyz].values.astype(np.float32))
        input_dim = 63
        log.info("  format: 1-hand xyz (63 features)")

    y = np.array(df["letter"].astype(str))

    # Apply hand-centric normalization
    X = normalize_features(X, input_dim)
    log.info(f"  input_dim={input_dim}, applied hand-centric normalization")

    return X, y, input_dim


class CNN(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
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


def train(model, train_loader, val_loader, epochs):
    log.info(f"Training {epochs} epochs...")
    start = datetime.now()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            preds = model(X.to(DEVICE)).argmax(1)
            correct = (preds == y.to(DEVICE)).sum().item()
            acc = correct / len(y)
            log.info(f"  Epoch {epoch+1}/{epochs} - val_acc={acc:.4f}")

    elapsed = (datetime.now() - start).total_seconds()
    model.eval()
    all_preds = torch.cat([model(X.to(DEVICE)).argmax(1) for X, _ in val_loader]).cpu().numpy()
    all_labels = torch.cat([y for _, y in val_loader]).cpu().numpy()
    acc = (all_preds == all_labels).mean()

    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    log.info(f"Done in {elapsed:.1f}s - acc={acc:.4f}, f1={f1:.4f}")
    return float(acc), float(f1), elapsed


def save(model, scaler, labels, acc, f1, elapsed, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(MODEL_DIR, f"{name}_model.pt"))
    with open(os.path.join(MODEL_DIR, f"{name}_scaler.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)
    with open(os.path.join(MODEL_DIR, f"{name}_labels.json"), "w") as f:
        json.dump(list(labels), f)
    with open(os.path.join(MODEL_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1, "time": elapsed, "trained_at": datetime.now().isoformat()}, f, indent=2)
    log.info(f"Saved: {MODEL_DIR}/{name}_*")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/landmarks_captured_v2.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--name", default="cnn")
    args = p.parse_args()

    X, y, input_dim = load_data(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr).astype(np.int64)
    y_te = le.transform(y_te).astype(np.int64)

    tr_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr.reshape(-1, 1, input_dim)), torch.LongTensor(y_tr)), batch_size=args.batch, shuffle=True)
    te_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te.reshape(-1, 1, input_dim)), torch.LongTensor(y_te)), batch_size=args.batch)

    model = CNN(input_dim=input_dim, num_classes=len(le.classes_)).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    acc, f1, elapsed = train(model, tr_loader, te_loader, args.epochs)
    save(model, scaler, le.classes_, acc, f1, elapsed, args.name)
    log.info(f"Complete! acc={acc:.4f}, f1={f1:.4f}")