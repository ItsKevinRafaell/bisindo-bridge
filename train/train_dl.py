#!/usr/bin/env python3
"""
BISINDO Bridge - DL Team Training Script
Trains deep learning models (MLP, CNN) on landmark features.

Team: DL (2 orang)
Usage: python train/train_dl.py --model cnn --arch 1

CNN Architectures:
  --arch 1: Baseline CNN (64->128->64)
  --arch 2: Deep CNN (128->256->128->64)
  --arch 3: Wide CNN (256->512->256)
  --arch 4: CNN + LSTM hybrid

Note: Requires tensorflow
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


def build_mlp(num_classes, hidden_units=[256, 128, 64], dropout=[0.3, 0.2, 0.1]):
    """Build MLP model with configurable architecture."""
    from tensorflow import keras
    layers = [keras.layers.Input(shape=(63,))]
    for i, units in enumerate(hidden_units):
        layers.append(keras.layers.Dense(units, activation="relu"))
        if i < len(dropout):
            layers.append(keras.layers.Dropout(dropout[i]))
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    log.info(f"MLP architecture: {hidden_units}")
    return model


def build_cnn(num_classes, arch=1, kernel_size=3):
    """Build CNN model with different architectures."""
    from tensorflow import keras

    configs = {
        1: ([64, 128, 64], "Baseline CNN"),
        2: ([128, 256, 128, 64], "Deep CNN"),
        3: ([256, 512, 256], "Wide CNN"),
    }

    filters, name = configs.get(arch, ([64, 128, 64], "CNN"))

    log.info(f"CNN architecture {arch} ({name}): filters={filters}, kernel={kernel_size}")

    layers = [keras.layers.Input(shape=(63, 1))]

    for f in filters:
        layers.append(keras.layers.Conv1D(f, kernel_size=kernel_size, activation="relu", padding="same"))
        layers.append(keras.layers.MaxPooling1D(pool_size=2))

    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dense(128, activation="relu"))
    layers.append(keras.layers.Dropout(0.3))
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))

    model = keras.Sequential(layers, name=f"cnn_arch{arch}")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=256):
    """Train model and return accuracy."""
    from tensorflow import keras
    log.info(f"Training {model.name}... epochs={epochs}, batch_size={batch_size}")
    start = datetime.now()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    elapsed = (datetime.now() - start).total_seconds()
    _, acc = model.evaluate(X_test, y_test, verbose=0)

    log.info(f"Training done in {elapsed:.1f}s, accuracy={acc:.4f}")
    return float(acc), elapsed


def save_model(model, scaler, label_encoder, acc, elapsed, model_name, arch=None):
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
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "training_time_seconds": float(elapsed),
        "trained_at": datetime.now().isoformat(),
    }
    if arch:
        metrics["architecture"] = arch

    metrics_path = os.path.join(MODEL_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Saved to {MODEL_DIR}/{model_name}_*")


def main():
    parser = argparse.ArgumentParser(description="Train DL models")
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
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    # One-hot encode
    from tensorflow import keras
    y_train_oh = keras.utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_oh = keras.utils.to_categorical(y_test_enc, num_classes=num_classes)

    # Train
    if args.model == "mlp":
        model_name = "mlp"
        model = build_mlp(num_classes)
        acc, elapsed = train_model(model, X_train_s, y_train_oh, X_test_s, y_test_oh,
                                   args.epochs, args.batch_size)
        save_model(model, scaler, label_encoder, acc, elapsed, model_name)

    elif args.model == "cnn":
        # Reshape for CNN: (samples, features, channels)
        X_train_cnn = X_train_s.reshape(-1, 63, 1)
        X_test_cnn = X_test_s.reshape(-1, 63, 1)

        model_name = f"cnn_arch{args.arch}_k{args.kernel}"
        model = build_cnn(num_classes, arch=args.arch, kernel_size=args.kernel)
        acc, elapsed = train_model(model, X_train_cnn, y_train_oh, X_test_cnn, y_test_oh,
                                   args.epochs, args.batch_size)
        save_model(model, scaler, label_encoder, acc, elapsed, model_name,
                   arch=f"filters_{args.arch}_kernel_{args.kernel}")

    log.info(f"Training complete! Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()