#!/usr/bin/env python3
"""
BISINDO Bridge - ML Team Training Script
Trains traditional ML models (Random Forest, SVM) on landmark features.

Team: ML (2 orang)
Usage: python train/train_ml.py [--model rf|svm] [--epochs 100]

Output: models/ml/*.pkl, models/ml/metrics.json
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("ml_train")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ml")
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
    X = df[cols].astype(np.float32).values
    X = np.nan_to_num(X, nan=0.0)
    y = df["letter"].astype(str).values
    return X, y


def train_rf(X_train, y_train, n_estimators=300):
    log.info(f"Training Random Forest (n={n_estimators})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    log.info("Training SVM...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, label_encoder):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, labels=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

    return {
        "accuracy": float(acc),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_letter": {l: report[l] for l in label_encoder.classes_ if l in report}
    }


def save_model(model, scaler, label_encoder, metrics, model_name):
    """Save model and artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Scaler
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Labels
    labels_path = os.path.join(MODEL_DIR, f"{model_name}_labels.pkl")
    with open(labels_path, "wb") as f:
        pickle.dump({"classes": list(label_encoder.classes_)}, f)

    # Metrics
    metrics_path = os.path.join(MODEL_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)

    log.info(f"✅ Saved to {MODEL_DIR}/{model_name}_*")


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--model", choices=["rf", "svm", "both"], default="both",
                        help="Model to train (default: both)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees for RF (default: 300)")
    args = parser.parse_args()

    # Load data
    import pandas as pd
    df = load_data(CSV_PATH)
    X, y = build_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    log.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Train
    if args.model in ["rf", "both"]:
        rf = train_rf(X_train_s, y_train, args.n_estimators)
        rf_metrics = evaluate(rf, X_test_s, y_test, label_encoder)
        save_model(rf, scaler, label_encoder, rf_metrics, "rf")
        log.info(f"RF Accuracy: {rf_metrics['accuracy']:.4f}")

    if args.model in ["svm", "both"]:
        svm = train_svm(X_train_s, y_train)
        svm_metrics = evaluate(svm, X_test_s, y_test, label_encoder)
        save_model(svm, scaler, label_encoder, svm_metrics, "svm")
        log.info(f"SVM Accuracy: {svm_metrics['accuracy']:.4f}")

    log.info("✅ Training complete!")


if __name__ == "__main__":
    main()
