#!/usr/bin/env python3
"""
BISINDO Bridge - DL Team Training Script (PyTorch)
Trains deep learning models (MLP, CNN) on landmark features.

Team: DL (2 orang)
Usage: python train/train_dl.py --model cnn --arch 1 --epochs 50

CNN Architectures:
  --arch 1: Baseline CNN (64->128->64)
  --arch 2: Deep CNN (128->256->128->64)
  --arch 3: Wide CNN (256->512->256)

Requires: pip install torch
"""

import os
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("dl_train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models", "dl")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_captured_v2.csv")


def load_data(csv_path):
    """Load and preprocess landmark data."""
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"  rows={len(df)}, letters={df['letter'].nunique()}")
    return df


def build_features(df):
    """Build feature matrix from landmarks (63 features per sample)."""
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    X = df[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(df["letter"].astype(str))
    return X, y


class MLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=26, hidden=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, input_dim=63, num_classes=26, arch=1, kernel_size=3):
        super().__init__()

        configs = {
            1: [64, 128, 64],
            2: [128, 256, 128, 64],
            3: [256, 512, 256],
        }
        filters = configs.get(arch, [64, 128, 64])

        layers = [nn.Conv1d(1, filters[0], kernel_size, padding=kernel_size//2)]
        for i in range(len(filters) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Conv1d(filters[i], filters[i+1], kernel_size, padding=kernel_size//2))

        self.conv = nn.Sequential(*layers)
        # Use AdaptiveAvgPool to handle variable output sizes
        self.pool = nn.AdaptiveAvgPool1d(8)  # Fixed output length of 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1] * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


def train_model(model, train_loader, val_loader, epochs=50):
    """Train PyTorch model."""
    log.info(f"Training {model.__class__.__name__}... epochs={epochs}")
    start = datetime.now()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            acc = correct / total
            log.info(f"  Epoch {epoch+1}/{epochs} - loss={total_loss/len(train_loader):.4f} - val_acc={acc:.4f}")

    elapsed = (datetime.now() - start).total_seconds()

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total

    log.info(f"Training done in {elapsed:.1f}s, accuracy={acc:.4f}")
    return float(acc), elapsed


def save_model(model, scaler, label_encoder, acc, elapsed, model_name, arch=None):
    """Save model and artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # PyTorch model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pt")
    torch.save(model.state_dict(), model_path)

    # Scaler JSON
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }, f)

    # Labels JSON
    labels_path = os.path.join(MODEL_DIR, f"{model_name}_labels.json")
    with open(labels_path, "w") as f:
        json.dump(list(label_encoder.classes_), f)

    # Metrics
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "training_time_seconds": float(elapsed),
        "trained_at": datetime.now().isoformat(),
        "framework": "pytorch",
    }
    if arch:
        metrics["architecture"] = arch

    metrics_path = os.path.join(MODEL_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Saved to {MODEL_DIR}/{model_name}_*")


def main():
    parser = argparse.ArgumentParser(description="Train DL models (PyTorch)")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn",
                        help="Model type (default: cnn)")
    parser.add_argument("--arch", type=int, default=1, choices=[1, 2, 3],
                        help="CNN architecture (1=baseline, 2=deep, 3=wide)")
    parser.add_argument("--kernel", type=int, default=3, choices=[3, 5, 7],
                        help="CNN kernel size (default: 3)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    args = parser.parse_args()

    # Load data
    df = load_data(CSV_PATH)
    X, y = build_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    log.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train).astype(np.int64)
    y_test_enc = label_encoder.transform(y_test).astype(np.int64)
    num_classes = len(label_encoder.classes_)

    # Create dataloaders
    if args.model == "mlp":
        X_train_t = torch.FloatTensor(X_train_s)
        X_test_t = torch.FloatTensor(X_test_s)
        model_name = "mlp"
    else:  # cnn
        X_train_t = torch.FloatTensor(X_train_s.reshape(-1, 1, 63))
        X_test_t = torch.FloatTensor(X_test_s.reshape(-1, 1, 63))
        model_name = f"cnn_arch{args.arch}_k{args.kernel}"

    y_train_t = torch.LongTensor(y_train_enc)
    y_test_t = torch.LongTensor(y_test_enc)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Build model
    if args.model == "mlp":
        model = MLP(input_dim=63, num_classes=num_classes).to(DEVICE)
    else:
        model = CNN1D(input_dim=63, num_classes=num_classes, arch=args.arch, kernel_size=args.kernel).to(DEVICE)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    acc, elapsed = train_model(model, train_loader, val_loader, args.epochs)

    # Save
    arch_str = f"arch{args.arch}_kernel{args.kernel}" if args.model == "cnn" else "mlp"
    save_model(model, scaler, label_encoder, acc, elapsed, model_name, arch=arch_str)

    log.info(f"Training complete! Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()