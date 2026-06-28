#!/usr/bin/env python3
"""
BISINDO Bridge - Model Comparison Script
Compares ML vs DL model performance.

Usage: python eval/compare.py

Output: models/comparison_report.md
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("compare")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_captured_v2.csv")


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
        configs = {1: [64, 128, 64], 2: [128, 256, 128, 64], 3: [256, 512, 256]}
        filters = configs.get(arch, [64, 128, 64])
        layers = [nn.Conv1d(1, filters[0], kernel_size, padding=kernel_size//2)]
        for i in range(len(filters) - 1):
            layers.extend([nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(filters[i], filters[i+1], kernel_size, padding=kernel_size//2)])
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1] * (input_dim // (2**len(filters))), 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def load_data():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    X = df[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(df["letter"].astype(str))
    return X, y


def load_sklearn_model(name):
    """Load sklearn model."""
    import pickle
    model_path = os.path.join(MODEL_DIR, "ml", f"{name}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "ml", f"{name}_scaler.pkl")
    labels_path = os.path.join(MODEL_DIR, "ml", f"{name}_labels.pkl")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, labels_path]):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels_data = pickle.load(f)
        labels = labels_data.get("classes", labels_data) if isinstance(labels_data, dict) else labels_data
    return model, scaler, list(labels)


def load_pytorch_model(name):
    """Load PyTorch model."""
    model_path = os.path.join(MODEL_DIR, "dl", f"{name}_model.pt")
    scaler_path = os.path.join(MODEL_DIR, "dl", f"{name}_scaler.json")
    labels_path = os.path.join(MODEL_DIR, "dl", f"{name}_labels.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, labels_path]):
        return None
    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return model_path, scaler_data, labels


def predict_sklearn(model, scaler, X):
    X_s = scaler.transform(X)
    return model.predict(X_s)


def predict_pytorch(model_path, scaler_data, X, is_cnn=False):
    """Predict with PyTorch model."""
    if is_cnn:
        X = X.reshape(-1, 1, 63)
    mean = np.array(scaler_data["mean"])
    scale = np.array(scaler_data["scale"])
    X_s = ((X - mean) / scale).astype(np.float32)
    X_t = torch.FloatTensor(X_s)
    model = MLP() if "mlp" in model_path else CNN1D()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        outputs = model(X_t)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


def evaluate_models(X_test, y_test):
    results = []

    # Sklearn models
    for name in ["rf", "svm"]:
        log.info(f"Loading {name}...")
        data = load_sklearn_model(name)
        if data:
            model, scaler, labels = data
            y_pred = predict_sklearn(model, scaler, X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({"name": name.upper(), "accuracy": acc})
            log.info(f"{name.upper()} accuracy: {acc:.4f}")

    # PyTorch models
    for name in os.listdir(os.path.join(MODEL_DIR, "dl")):
        if name.endswith("_model.pt"):
            model_name = name.replace("_model.pt", "")
            is_cnn = "cnn" in model_name
            log.info(f"Loading {model_name}...")
            data = load_pytorch_model(model_name)
            if data:
                model_path, scaler_data, labels = data
                y_pred = predict_pytorch(model_path, scaler_data, X_test, is_cnn)
                y_pred_labels = np.array(labels)[y_pred]
                acc = accuracy_score(y_test, y_pred_labels)
                results.append({"name": model_name.upper(), "accuracy": acc})
                log.info(f"{model_name.upper()} accuracy: {acc:.4f}")

    return results


def generate_report(results):
    lines = [
        "# BISINDO Model Comparison Report",
        f"_Generated: {datetime.now().isoformat()}_",
        "",
        "## Summary",
        "",
        "| Model | Accuracy |",
        "|-------|----------|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r['accuracy']:.4f} |")

    valid = [r for r in results if r]
    if valid:
        winner = max(valid, key=lambda x: x["accuracy"])
        lines.extend(["", f"**Best Model: {winner['name']}** ({winner['accuracy']:.2%} accuracy)"])

    return "\n".join(lines)


def main():
    log.info("Loading data...")
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    results = evaluate_models(X_test, y_test)
    report = generate_report(results)

    report_path = os.path.join(MODEL_DIR, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    log.info(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()