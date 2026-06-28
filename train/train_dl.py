#!/usr/bin/env python3
"""
BISINDO Bridge - DL Team Training Script
Trains deep learning models (MLP, CNN) on landmark features.

Team: DL (2 orang)
Usage: python train/train_dl.py [--model mlp|cnn] [--epochs 50]

Output: models/dl/*.h5, models/dl/metrics.json

Note: Requires tensorflow and tensorflowjs
  pip install tensorflow tensorflowjs
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("dl_train")

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
    X = df[cols].astype(np.float32).values
    X = np.nan_to_num(X, nan=0.0)
    y = df["letter"].astype(str).values
    return X, y


def build_mlp(num_classes, input_dim=63):
    """Build MLP model for landmark classification."""
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(num_classes, input_dim=63):
    """Build 1D CNN model for landmark classification."""
    from tensorflow import keras
    # Reshape for CNN: (samples, features, channels)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=256):
    """Train model and evaluate."""
    from tensorflow import keras
    log.info(f"Training {model.name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return float(acc), history


def save_model(model, scaler, label_encoder, acc, model_name):
    """Save model and artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Keras model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.h5")
    model.save(model_path)

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
    metrics_path = os.path.join(MODEL_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": acc,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)

    log.info(f"Saved to {MODEL_DIR}/{model_name}_*")


def export_tfjs(model_path, output_dir):
    """Export Keras model to TensorFlow.js format."""
    try:
        import tensorflowjs as tfjs
        os.makedirs(output_dir, exist_ok=True)
        tfjs.converters.save_keras_model(model_path, output_dir)
        log.info(f"TF.js model -> {output_dir}")
    except ImportError:
        log.warning("tensorflowjs not installed: pip install tensorflowjs")


def main():
    parser = argparse.ArgumentParser(description="Train DL models")
    parser.add_argument("--model", choices=["mlp", "cnn", "both"], default="both",
                        help="Model to train (default: both)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
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
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    # One-hot encode
    from tensorflow import keras
    y_train_oh = keras.utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_oh = keras.utils.to_categorical(y_test_enc, num_classes=num_classes)

    # Train
    if args.model in ["mlp", "both"]:
        mlp = build_mlp(num_classes)
        mlp_acc, _ = train_model(mlp, X_train_s, y_train_oh, X_test_s, y_test_oh, args.epochs)
        save_model(mlp, scaler, label_encoder, mlp_acc, "mlp")
        log.info(f"MLP Accuracy: {mlp_acc:.4f}")

    if args.model in ["cnn", "both"]:
        cnn = build_cnn(num_classes)
        cnn_acc, _ = train_model(cnn, X_train_s, y_train_oh, X_test_s, y_test_oh, args.epochs)
        save_model(cnn, scaler, label_encoder, cnn_acc, "cnn")
        log.info(f"CNN Accuracy: {cnn_acc:.4f}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
