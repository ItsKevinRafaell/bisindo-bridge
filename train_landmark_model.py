#!/usr/bin/env python3
"""
BISINDO Landmark Model Training
Reads unified 67-col CSV, trains RandomForest + MLP, exports TF.js for browser.

Schema: letter, image_path, split, num_hands, contributor, lm0_x..lm20_z (63 cols)
Model input: 126 features = hand1 (63) + zero-padded hand2 (63).

Outputs:
  models/landmark_classifier.pkl              — sklearn RF (laptop server)
  models/landmark_classifier_scaler.pkl
  models/landmark_classifier_labels.pkl
  models/landmark_classifier_metadata.json
  models/report.md                           — per-letter metrics
  models/mlp_model.h5                         — Keras MLP (TF.js source)
  web/models/model.json + shards              — TF.js for /test page
  web/models/labels.json
  web/models/scaler.json
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("train")

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_DEFAULT = os.path.join(BASE, "dataset", "landmarks_captured_v2.csv")
MODEL_DIR = os.path.join(BASE, "models")
WEB_MODEL_DIR = os.path.join(BASE, "web", "models")

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_HAND = 21 * 3  # 63


def load_data(csv_path):
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"  rows={len(df)}, letters={df['letter'].nunique()}, "
             f"contributors={df['contributor'].nunique()}")
    return df


def build_features(df):
    """Hand 1 (63 cols) → padded to 126 features (hand1 + zero hand2)."""
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing[:5]}…")

    X1 = df[cols].astype(np.float32).values
    X1 = np.nan_to_num(X1, nan=0.0)
    # Pad hand 2 with zeros
    X2 = np.zeros_like(X1)
    X = np.concatenate([X1, X2], axis=1).astype(np.float32)
    y = df["letter"].astype(str).values
    return X, y


def train_rf(X_train, y_train, X_test, y_test, n_estimators=300):
    log.info(f"Training Random Forest (n={n_estimators})…")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    log.info(f"RF test accuracy: {acc:.4f}")
    return rf, pred, acc


def train_mlp(X_train, y_train, X_test, y_test, epochs=40):
    """Train a small MLP for browser export."""
    import tensorflow as tf
    from tensorflow import keras

    letters = sorted(set(y_train))
    cls_idx = {l: i for i, l in enumerate(letters)}
    y_train_i = np.array([cls_idx[l] for l in y_train])
    y_test_i = np.array([cls_idx[l] for l in y_test])
    y_train_oh = keras.utils.to_categorical(y_train_i, num_classes=len(letters))
    y_test_oh = keras.utils.to_categorical(y_test_i, num_classes=len(letters))

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(letters), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    log.info(f"MLP: {model.count_params()} params, classes={len(letters)}")
    model.fit(X_train, y_train_oh, validation_data=(X_test, y_test_oh),
              epochs=epochs, batch_size=256, verbose=2)
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    log.info(f"MLP test accuracy: {test_acc:.4f}")
    return model, letters, test_acc


def save_sklearn(model, scaler, labels, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "landmark_classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "landmark_classifier_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "landmark_classifier_labels.pkl"), "wb") as f:
        pickle.dump({"classes": list(labels), "n_classes": len(labels)}, f)
    meta = {
        "model_type": type(model).__name__,
        "n_features": int(scaler.mean_.shape[0]),
        "classes": list(labels),
        "n_classes": len(labels),
        "trained_at": datetime.now().isoformat(),
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()},
    }
    with open(os.path.join(out_dir, "landmark_classifier_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def save_tfjs(keras_model, scaler, labels, out_dir):
    """Export Keras model + scaler + labels for browser."""
    os.makedirs(out_dir, exist_ok=True)
    # Save Keras
    h5_path = os.path.join(out_dir, "mlp_model.h5")
    keras_model.save(h5_path)

    # Convert to TF.js
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(keras_model, out_dir)
        log.info(f"TF.js model → {out_dir}")
    except ImportError:
        log.warning("tensorflowjs not installed; install with: pip install tensorflowjs")
        log.warning(f"Keras model saved at {h5_path} — convert manually")

    # Labels JSON
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(list(labels), f)

    # Scaler JSON
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }, f)


def write_report(metrics, labels, y_test, pred_rf, out_dir):
    report = classification_report(y_test, pred_rf, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, pred_rf, labels=labels)
    md = [
        "# BISINDO Model Report",
        f"_Generated: {datetime.now().isoformat()}_",
        "",
        "## Overall",
        f"- Test accuracy (RF): **{metrics['rf_accuracy']:.4f}**",
        f"- Test accuracy (MLP): **{metrics.get('mlp_accuracy', 0):.4f}**",
        f"- Classes: {len(labels)}",
        f"- Features: {metrics.get('n_features', 126)}",
        "",
        "## Per-letter (Random Forest)",
        "",
        "| Letter | Precision | Recall | F1 | Support |",
        "|--------|-----------|--------|----|---------|",
    ]
    cls = classification_report(y_test, pred_rf, labels=labels,
                                output_dict=True, zero_division=0)
    for l in labels:
        if l in cls:
            r = cls[l]
            md.append(f"| {l} | {r['precision']:.3f} | {r['recall']:.3f} | "
                      f"{r['f1-score']:.3f} | {int(r['support'])} |")
    md.append("")
    md.append("## Confusion matrix")
    md.append("```")
    md.append("    " + " ".join(f"{l:>3}" for l in labels))
    for i, row in enumerate(cm):
        md.append(f"{labels[i]:>3} " + " ".join(f"{v:>3}" for v in row))
    md.append("```")
    md.append("")
    md.append("## Classification report")
    md.append("```")
    md.append(report)
    md.append("```")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    log.info(f"Report → {out_dir}/report.md")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    if not os.path.exists(csv_path):
        log.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = load_data(csv_path)
    X, y = build_features(df)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    log.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    labels = sorted(np.unique(y_train).tolist())
    log.info(f"Classes: {labels}")

    # 1. Random Forest (sklearn, for laptop server)
    rf, pred_rf, rf_acc = train_rf(X_train_s, y_train, X_test_s, y_test)
    metrics = {"rf_accuracy": rf_acc, "n_features": X.shape[1]}
    save_sklearn(rf, scaler, labels, metrics, MODEL_DIR)

    # 2. MLP (Keras, for browser)
    mlp_acc = 0.0
    try:
        mlp, mlp_labels, mlp_acc = train_mlp(X_train_s, y_train, X_test_s, y_test)
        metrics["mlp_accuracy"] = mlp_acc
        save_tfjs(mlp, scaler, mlp_labels, WEB_MODEL_DIR)
    except ImportError:
        log.warning("tensorflow not installed — skipping MLP/TF.js export")
        log.warning("pip install tensorflow tensorflowjs")

    write_report(metrics, labels, y_test, pred_rf, MODEL_DIR)
    log.info("✅ Done.")


if __name__ == "__main__":
    main()