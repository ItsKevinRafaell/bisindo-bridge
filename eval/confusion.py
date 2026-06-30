#!/usr/bin/env python3
"""Generate confusion matrix from trained model."""
import json, os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def load_features(df):
    """Auto-detect format and extract features."""
    cols_2hand_xy = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
    cols_2hand_xyz = [f"h1_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]

    if all(c in df.columns for c in cols_2hand_xyz):
        h1 = [f"h1_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
        h2 = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
        X = np.concatenate([df[h1].values, df[h2].values], axis=1)
        dim = 126
    elif all(c in df.columns for c in cols_2hand_xy):
        h1 = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y")]
        h2 = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
        X = np.concatenate([df[h1].values, df[h2].values], axis=1)
        dim = 84
    else:
        cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
        X = df[cols].values
        dim = 63

    return np.nan_to_num(X.astype(np.float32)), dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn_2hand", help="Model name (e.g. cnn_2hand)")
    args = p.parse_args()

    model_dir = os.path.join(BASE, "models", "dl")
    model_name = args.model

    # Load model artifacts
    model_path = os.path.join(model_dir, f"{model_name}_model.pt")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.json")
    labels_path = os.path.join(model_dir, f"{model_name}_labels.json")

    with open(scaler_path) as f:
        sd = json.load(f)
    with open(labels_path) as f:
        labels = json.load(f)

    # Load same data and split
    df = pd.read_csv(os.path.join(BASE, "dataset", "landmarks_2hands.csv"))
    X, dim = load_features(df)
    y = df["letter"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    le = LabelEncoder()
    le.fit(y_tr)
    y_te_enc = le.transform(y_te)

    # Load and predict
    model = CNN(input_dim=dim, num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    X_t = torch.FloatTensor(X_te_s.reshape(-1, 1, dim))
    with torch.no_grad():
        preds = model(X_t).argmax(1).numpy()

    # Confusion matrix
    cm = confusion_matrix(y_te_enc, preds)
    print(f"\n=== Confusion Matrix ({model_name}) ===")
    print(f"Test samples: {len(y_te)}")
    print(f"Classes: {labels}")
    print()

    # Print classification report
    print(classification_report(y_te_enc, preds, target_names=labels, digits=3))

    # Find most confused pairs
    print("Most confused pairs (actual → predicted: count):")
    errors = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                errors.append((cm[i][j], labels[i], labels[j]))
    errors.sort(reverse=True)
    for count, actual, predicted in errors[:10]:
        print(f"  {actual} → {predicted}: {count}")

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels)*0.8)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (acc={np.diag(cm).sum()/cm.sum():.4f})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            if val > 0:
                color = "white" if cm_norm[i][j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im)
    plt.tight_layout()
    out = os.path.join(BASE, "eval", f"confusion_{model_name}.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
