#!/usr/bin/env python3
"""
BISINDO Bridge - Model Comparison Script
Compares ML (RF, SVM) vs DL (MLP, CNN) performance.

Usage: python eval/compare.py

Output: models/comparison_report.md
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("compare")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_captured_v2.csv")


def load_data():
    """Load landmark data."""
    df = pd.read_csv(CSV_PATH, low_memory=False)
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    X = df[cols].astype(np.float32).values
    X = np.nan_to_num(X, nan=0.0)
    y = df["letter"].astype(str).values
    return X, y


def load_sklearn_model(name):
    """Load sklearn model and artifacts."""
    model_path = os.path.join(MODEL_DIR, "ml", f"{name}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "ml", f"{name}_scaler.pkl")
    labels_path = os.path.join(MODEL_DIR, "ml", f"{name}_labels.pkl")

    if not os.path.exists(model_path):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels_data = pickle.load(f)
        labels = labels_data.get("classes", labels_data)

    return model, scaler, labels


def load_keras_model(name):
    """Load Keras model and artifacts."""
    from tensorflow import keras

    model_path = os.path.join(MODEL_DIR, "dl", f"{name}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, "dl", f"{name}_scaler.json")
    labels_path = os.path.join(MODEL_DIR, "dl", f"{name}_labels.json")

    if not os.path.exists(model_path):
        return None, None, None

    model = keras.models.load_model(model_path)
    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)
    with open(labels_path, "r") as f:
        labels = json.load(f)

    return model, scaler_data, labels


def predict_sklearn(model, scaler, X):
    """Predict with sklearn model."""
    X_s = scaler.transform(X)
    return model.predict(X_s)


def predict_keras(model, scaler_data, X):
    """Predict with Keras model."""
    mean = np.array(scaler_data["mean"])
    scale = np.array(scaler_data["scale"])
    X_s = (X - mean) / scale
    pred = model.predict(X_s, verbose=0)
    pred_idx = np.argmax(pred, axis=1)
    return np.array(labels)[pred_idx]


def evaluate_model(name, model, X_test, y_test, labels):
    """Evaluate a single model."""
    if model is None:
        return None

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "name": name,
        "accuracy": float(acc),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0)),
        "per_letter": {l: report[l] for l in labels if l in report}
    }


def generate_report(results):
    """Generate markdown comparison report."""
    lines = [
        "# BISINDO Model Comparison Report",
        f"_Generated: {datetime.now().isoformat()}_",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Weighted F1 |",
        "|-------|----------|-------------|",
    ]

    for r in results:
        if r:
            lines.append(f"| {r['name']} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} |")

    # Winner
    valid_results = [r for r in results if r]
    if valid_results:
        winner = max(valid_results, key=lambda x: x["accuracy"])
        lines.extend([
            "",
            f"**Best Model: {winner['name']}** ({winner['accuracy']:.2%} accuracy)",
            "",
            "## Per-Letter Comparison",
            "",
        ])

        # Table header
        letters = sorted(set().union(*[set(r["per_letter"].keys()) for r in valid_results]))
        lines.append("| Letter | " + " | ".join(r["name"] for r in valid_results) + " |")
        lines.append("|--------|" + "|".join(["---------" for _ in valid_results]) + "|")

        for letter in letters:
            row = [letter]
            for r in valid_results:
                if letter in r["per_letter"]:
                    f1 = r["per_letter"][letter]["f1-score"]
                    row.append(f"{f1:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    return "
".join(lines)


def main():
    # Load data
    log.info("Loading data...")
    X, y = load_data()

    # Split (same as training)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    results = []

    # Load sklearn models
    for name in ["rf", "svm"]:
        log.info(f"Loading {name}...")
        model, scaler, labels = load_sklearn_model(name)
        if model:
            y_pred = predict_sklearn(model, scaler, X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "name": name.upper(),
                "accuracy": float(acc),
                "per_letter": {}
            })
            log.info(f"{name.upper()} accuracy: {acc:.4f}")

    # Load Keras models
    for name in ["mlp", "cnn"]:
        log.info(f"Loading {name}...")
        try:
            model, scaler_data, labels = load_keras_model(name)
            if model:
                mean = np.array(scaler_data["mean"])
                scale = np.array(scaler_data["scale"])
                X_s = (X_test - mean) / scale
                pred = model.predict(X_s, verbose=0)
                pred_idx = np.argmax(pred, axis=1)
                y_pred = np.array(labels)[pred_idx]
                acc = accuracy_score(y_test, y_pred)
                results.append({
                    "name": name.upper(),
                    "accuracy": float(acc),
                    "per_letter": {}
                })
                log.info(f"{name.upper()} accuracy: {acc:.4f}")
        except Exception as e:
            log.warning(f"Failed to load {name}: {e}")

    # Generate report
    report = generate_report(results)
    report_path = os.path.join(MODEL_DIR, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    log.info(f"✅ Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()