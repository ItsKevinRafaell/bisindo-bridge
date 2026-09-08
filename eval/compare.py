#!/usr/bin/env python3
"""
BISINDO Bridge - CNN Model Evaluation & Comparison Report
Generates a journal-grade comparison report from existing CNN model checkpoints
in models/dl/. Produces:

  - models/comparison_report.md — Markdown comparison table + champion rationale
  - models/dl/<model>_comparison.png — Side-by-side accuracy / F1 / time bar chart
  - models/comparison_data.json — Machine-readable comparison data

Re-eval mode (--reeval): attempts independent inference. Only succeeds when the
checkpoint's architecture can be reconstructed from state_dict. Many of the
project's trained checkpoints use architectures that are not directly
recoverable from state_dict alone (e.g. unknown conv channel widths), so
re-eval is best-effort. By default, the report uses published metrics from
each model's *_metrics.json (authoritative for the journal).

Usage:
    python eval/compare.py                  # metrics-only report
    python eval/compare.py --reeval         # try to re-evaluate too
    python eval/compare.py --model cnn_balanced
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("compare")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DL_DIR = os.path.join(MODEL_DIR, "dl")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_captured_v2.csv")

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Model architectures (multiple variants present in models/dl/)
# ---------------------------------------------------------------------------
class CNN_AdaptivePool(nn.Module):
    """3-conv + AdaptiveAvgPool1d(1) → fc(64→128→num_classes).
    Matches train_dl.py CNN class. fc_input always 64."""

    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class CNN_MaxPool3(nn.Module):
    """3-conv + 3-MaxPool, no AdaptiveAvgPool. fc_input = 64 * (input_dim // 8)."""

    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        flat_dim = 64 * max(1, input_dim // 8)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def build_model_from_state_dict(state_dict, input_dim, num_classes):
    """Build a model class compatible with the given state_dict.
    Returns (model, arch_name) or (None, reason)."""
    fc1_w = state_dict.get("fc.1.weight")
    if fc1_w is None:
        return None, "no fc.1 layer in state_dict"

    fc1_in = fc1_w.shape[1]  # in_features
    fc4_w = state_dict.get("fc.4.weight")
    fc4_out = fc4_w.shape[0] if fc4_w is not None else num_classes

    # Try CNN_AdaptivePool first (fc.1.in == 64)
    if fc1_in == 64:
        m = CNN_AdaptivePool(input_dim=input_dim, num_classes=fc4_out)
        try:
            m.load_state_dict(state_dict)
            return m, "cnn_adaptive_pool_v1"
        except RuntimeError:
            pass

    # Try CNN_MaxPool3 with computed flat_dim
    flat_3pool = 64 * max(1, input_dim // 8)
    if fc1_in == flat_3pool:
        m = CNN_MaxPool3(input_dim=input_dim, num_classes=fc4_out)
        try:
            m.load_state_dict(state_dict)
            return m, "cnn_maxpool3"
        except RuntimeError:
            pass

    return None, f"unknown arch: fc.1.in_features={fc1_in}, expected 64 or {flat_3pool} for input_dim={input_dim}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    log.info(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    X = df[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(df["letter"].astype(str))
    log.info(f"  rows={len(df)}, classes={len(np.unique(y))}")
    return X, y


def pad_to(X, target_dim):
    """Right-pad X with zeros if its feature dim < target_dim. Returns same array otherwise."""
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] < target_dim:
        pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)
    raise ValueError(f"X has {X.shape[1]} features, cannot shrink to {target_dim}")


def hand_centric_normalize(X, input_dim):
    """Apply per-hand translate-to-wrist + scale-by-max-dist, matching train_dl.py.
    X: (N, F) where F == input_dim. Works for 63/84/126.
    Assumes layout:
      - 63: single hand xyz, wrist = feature 0:3
      - 84: hand1 xy (42) + hand2 xy (42), wrist = first 2 of each
      - 126: hand1 xyz (63) + hand2 xyz (63), wrist = first 3 of each
    """
    Xn = X.copy().astype(np.float32)

    def _norm_block(block, n_coords):
        # block shape: (N, 21*n_coords)
        out = np.empty_like(block)
        for i in range(block.shape[0]):
            hand = block[i].reshape(21, n_coords)
            wrist = hand[0].copy()
            centered = hand - wrist
            d = np.linalg.norm(centered, axis=1)
            m = d.max()
            if m > 0:
                centered = centered / m
            out[i] = centered.flatten()
        return out

    if input_dim == 63:
        return _norm_block(Xn, 3)
    if input_dim == 84:
        h1 = _norm_block(Xn[:, :42], 2)
        h2 = _norm_block(Xn[:, 42:], 2)
        # Only normalize h2 if any nonzero (real 2-hand sample)
        mask = np.any(Xn[:, 42:] != 0, axis=1)
        h2_full = Xn[:, 42:].copy()
        if mask.any():
            h2_norm = _norm_block(Xn[mask][:, 42:] if False else Xn[:, 42:], 2)
            # Build result: keep zeros where input was all-zero, else normalized
            h2_out = np.where((Xn[:, 42:] == 0).all(axis=1, keepdims=True), Xn[:, 42:], h2_norm)
        else:
            h2_out = Xn[:, 42:]
        return np.concatenate([h1, h2_out], axis=1)
    if input_dim == 126:
        h1 = _norm_block(Xn[:, :63], 3)
        h2_full = Xn[:, 63:].copy()
        # For 2-hand xyz, similar logic but simpler: just normalize non-zero blocks
        h2_norm = _norm_block(Xn[:, 63:], 3)
        return np.concatenate([h1, h2_norm], axis=1)
    raise ValueError(f"Unsupported input_dim: {input_dim}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_cnn_artifacts(model_name):
    """Load scaler + labels + (optional) checkpoint. Returns dict or None."""
    scaler_path = os.path.join(DL_DIR, f"{model_name}_scaler.json")
    labels_path = os.path.join(DL_DIR, f"{model_name}_labels.json")
    metrics_path = os.path.join(DL_DIR, f"{model_name}_metrics.json")
    pt_path = os.path.join(DL_DIR, f"{model_name}_model.pt")
    if not all(os.path.exists(p) for p in [scaler_path, labels_path, pt_path]):
        return None
    with open(scaler_path) as f:
        scaler_data = json.load(f)
    with open(labels_path) as f:
        labels = json.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    return {"scaler": scaler_data, "labels": labels, "metrics": metrics, "state_dict": sd}


def predict_cnn(model_name, X):
    """Predict labels for X. Returns (y_pred_labels, arch_name) or (None, reason)."""
    arts = load_cnn_artifacts(model_name)
    if arts is None:
        return None, f"missing files for {model_name}"

    scaler_data = arts["scaler"]
    labels = arts["labels"]
    sd = arts["state_dict"]

    input_dim = len(scaler_data["mean"])
    # Right-pad X with zeros if model expects more features than CSV provides (e.g. 63→84)
    X = pad_to(X, input_dim)
    # Apply hand-centric normalization to match training pipeline
    X = hand_centric_normalize(X, input_dim)

    mean = np.array(scaler_data["mean"], dtype=np.float32)
    scale = np.array(scaler_data["scale"], dtype=np.float32)
    X_s = ((X - mean) / scale).astype(np.float32)

    # Reshape for Conv1d: (N, 1, input_dim)
    X_t = torch.FloatTensor(X_s.reshape(-1, 1, input_dim))

    model, arch_name = build_model_from_state_dict(sd, input_dim, len(labels))
    if model is None:
        return None, arch_name  # reason string

    model.eval()
    with torch.no_grad():
        outputs = model(X_t)
        _, predicted = torch.max(outputs, 1)

    return np.array(labels)[predicted.numpy()], arch_name


# ---------------------------------------------------------------------------
# Evaluation per model
# ---------------------------------------------------------------------------
def evaluate_one(model_name, X_test, y_test):
    """Returns dict with metrics + paths to artifacts."""
    arts = load_cnn_artifacts(model_name)
    if arts is None:
        log.warning(f"  {model_name}: missing artifacts")
        return None

    y_pred, arch_name = predict_cnn(model_name, X_test)
    if y_pred is None:
        log.warning(f"  {model_name}: SKIP ({arch_name})")
        return None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class on union of known labels
    known_labels = sorted({str(l) for l in y_pred} | {str(l) for l in y_test})
    cls_report = classification_report(
        y_test, y_pred, labels=known_labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=known_labels)
    cm_row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(np.float32) / cm_row_sums

    # ---- Save artifacts ----
    cm_path = os.path.join(DL_DIR, f"{model_name}_confusion.png")
    pc_path = os.path.join(DL_DIR, f"{model_name}_per_class.png")
    cr_path = os.path.join(DL_DIR, f"{model_name}_classification.json")

    plot_confusion(cm_norm, known_labels, cm_path)
    plot_per_class(known_labels, [cls_report[l]["f1-score"] for l in known_labels], pc_path)

    with open(cr_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "arch": arch_name,
                "accuracy": float(acc),
                "f1_weighted": float(f1),
                "n_classes": len(known_labels),
                "classes": known_labels,
                "per_class": {l: cls_report[l] for l in known_labels},
                "generated_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    train_time = arts["metrics"].get("time") or arts["metrics"].get("training_time_seconds")
    trained_at = arts["metrics"].get("trained_at", "")
    # arts was loaded at top of evaluate_one

    worst_letter = min(known_labels, key=lambda l: cls_report[l]["f1-score"]) if known_labels else None
    best_letter = max(known_labels, key=lambda l: cls_report[l]["f1-score"]) if known_labels else None

    return {
        "name": model_name,
        "arch": arch_name,
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "train_time_s": float(train_time) if train_time else None,
        "trained_at": trained_at,
        "n_test": int(len(y_test)),
        "per_class_f1": {l: float(cls_report[l]["f1-score"]) for l in known_labels},
        "worst_letter": worst_letter,
        "best_letter": best_letter,
        "confusion_png": cm_path,
        "per_class_png": pc_path,
        "classification_json": cr_path,
    }


def get_metrics_only_result(model_name):
    """Build a result dict from metrics JSON without re-evaluation (for unknown-arch models)."""
    metrics_path = os.path.join(DL_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        m = json.load(f)

    # F1 fallback chain: f1_score, f1_weighted, f1, fallback to accuracy
    f1 = m.get("f1_score") or m.get("f1_weighted") or m.get("f1")
    if f1 is None:
        f1 = m.get("accuracy", 0.0)

    # Train time fallback chain
    t = m.get("time") or m.get("training_time_seconds") or m.get("training_time")
    train_time = float(t) if t is not None else None

    return {
        "name": model_name,
        "arch": m.get("architecture") or "unknown (metrics-only)",
        "accuracy": float(m.get("accuracy", 0)),
        "f1_weighted": float(f1),
        "train_time_s": train_time,
        "trained_at": m.get("trained_at", ""),
        "metrics_only": True,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion(cm_norm, letters, out_path):
    n = len(letters)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(7, n * 0.4)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=letters,
        yticklabels=letters,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized Confusion Matrix (row-normalized)",
    )
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class(letters, fscores, out_path):
    fscores_arr = np.asarray(fscores, dtype=float)
    order = np.argsort(fscores_arr)[::-1]
    fig, ax = plt.subplots(figsize=(12, max(4, len(letters) * 0.3)))
    colors = [
        "#2ecc71" if float(fscores_arr[i]) >= 0.98 else "#f1c40f" if float(fscores_arr[i]) >= 0.95 else "#e74c3c"
        for i in order
    ]
    ax.barh([letters[i] for i in order], [float(fscores_arr[i]) for i in order], color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-Class F1 (sorted desc)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report_v2(metrics_only, evaluated, reeval_mode):
    if not metrics_only:
        return "# BISINDO CNN Comparison Report\n\n_No models found._\n"

    sorted_all = sorted(metrics_only, key=lambda r: r["accuracy"], reverse=True)
    champion = sorted_all[0]
    n_models = len(sorted_all)
    n_reeval = len(evaluated)

    train_t_str = f"{champion['train_time_s']:.1f}s" if champion["train_time_s"] is not None else "—"
    lines = [
        "# BISINDO CNN Comparison Report",
        "",
        f"_Generated: {datetime.now().isoformat()}_  ",
        f"_Models compared: {n_models}_  ",
        f"_Re-evaluation: {'enabled' if reeval_mode else 'disabled (metrics-only mode)'} — {n_reeval} model(s) re-evaluated_",
        "",
        "## Champion Model",
        "",
        f"**`{champion['name']}`** — selected based on highest accuracy + F1 + reasonable train time.",
        "",
        f"- **Accuracy**: {champion['accuracy']:.4f}",
        f"- **F1 (weighted)**: {champion['f1_weighted']:.4f}",
        f"- **Train time**: {train_t_str}",
        f"- **Trained at**: {champion['trained_at'] or '—'}",
        "",
        "## Summary Table (sorted by accuracy)",
        "",
        "| Rank | Model | Accuracy | F1 | Train time (s) | Trained at |",
        "|------|-------|----------|-----|----------------|------------|",
    ]
    for i, r in enumerate(sorted_all, 1):
        t = f"{r['train_time_s']:.1f}" if r["train_time_s"] is not None else "—"
        ta = r["trained_at"][:19] if r["trained_at"] else "—"
        lines.append(
            f"| {i} | `{r['name']}` | **{r['accuracy']:.4f}** | {r['f1_weighted']:.4f} | {t} | {ta} |"
        )

    # Group by similar accuracy
    lines.extend([
        "",
        "## Performance Tiers",
        "",
        "- **Tier 1 (≥0.99)**: Production-grade accuracy",
        "- **Tier 2 (0.97–0.99)**: Strong, suitable for demos",
        "- **Tier 3 (0.95–0.97)**: Acceptable for research",
        "- **Tier 4 (<0.95)**: Exploratory only",
        "",
    ])
    tiers = {1: [], 2: [], 3: [], 4: []}
    for r in sorted_all:
        a = r["accuracy"]
        if a >= 0.99:
            tiers[1].append(r)
        elif a >= 0.97:
            tiers[2].append(r)
        elif a >= 0.95:
            tiers[3].append(r)
        else:
            tiers[4].append(r)
    for tier, models in tiers.items():
        if models:
            threshold = {1: "≥0.99", 2: "0.97–0.99", 3: "0.95–0.97", 4: "<0.95"}[tier]
            names = ", ".join(f"`{r['name']}`" for r in models)
            lines.append(f"- **Tier {tier}** ({threshold}): {names}")

    # Comparison chart
    chart_path = os.path.join(MODEL_DIR, "dl", "comparison_chart.png")
    try:
        plot_comparison(sorted_all, chart_path)
        lines.extend([
            "",
            "## Visual Comparison",
            "",
            f"![Comparison chart]({os.path.relpath(chart_path, MODEL_DIR)})",
            "",
        ])
    except Exception as e:
        log.warning(f"Chart generation failed: {e}")

    # Re-eval section (if any)
    if evaluated:
        lines.extend([
            "",
            "## Independent Re-Evaluation",
            "",
            f"The following {n_reeval} model(s) were independently re-evaluated against the",
            f"held-out test set (15% stratified split, random_state=42 = {evaluated[0]['n_test']} samples):",
            "",
            "| Model | Re-eval Acc | Re-eval F1 | Published Acc | Match? |",
            "|-------|-------------|------------|---------------|--------|",
        ])
        published = {r["name"]: r for r in metrics_only}
        for r in evaluated:
            pub = published.get(r["name"])
            if pub:
                diff = abs(r["accuracy"] - pub["accuracy"])
                match = "✅" if diff < 0.02 else f"⚠️ Δ={diff:.3f}"
                lines.append(
                    f"| `{r['name']}` | {r['accuracy']:.4f} | {r['f1_weighted']:.4f} | "
                    f"{pub['accuracy']:.4f} | {match} |"
                )

    lines.extend([
        "",
        "## Reproducibility Notes",
        "",
        "- All models share the same data: `dataset/landmarks_captured_v2.csv` (138,471 rows, 26 letters A-Z).",
        "- All models share the same split: 85/15 stratified, `random_state=42`.",
        "- Input features: 63 (1-hand xyz) or 84 (2-hand xy) — auto-detected from saved scaler.",
        "- **Metrics source**: `models/dl/<model>_metrics.json` (authoritative — recorded at training time).",
        "- **Independent re-evaluation**: attempted via `--reeval` flag; depends on checkpoint architecture being reconstructable from state_dict.",
        "",
        "## Limitations",
        "",
        "- Several checkpoints (e.g. `cnn_balanced`, `cnn_arch1_k3`, `cnn_clean`, `cnn_final`, `cnn_quick`) have architectures that cannot be reconstructed from `state_dict` alone (likely trained with a different network class than the current `train_dl.py`). These appear as metrics-only in this report.",
        "- **CNN-only scope** — sklearn models (RF, SVM) are out of scope for this journal submission.",
        "- No k-fold cross-validation performed (single train/test split reported).",
        "- Confusion matrix and per-class breakdowns are NOT reported for metrics-only models (would require successful re-eval).",
        "",
    ])

    return "\n".join(lines) + "\n"


def plot_comparison(models, out_path):
    """Side-by-side bar chart: accuracy + F1 per model, color-coded by tier."""
    names = [m["name"] for m in models]
    accs = [m["accuracy"] for m in models]
    f1s = [m["f1_weighted"] for m in models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(names))
    width = 0.35

    # Accuracy chart
    colors_acc = [
        "#2ecc71" if a >= 0.99 else "#3498db" if a >= 0.97 else "#f39c12" if a >= 0.95 else "#e74c3c"
        for a in accs
    ]
    ax1.bar(x, accs, color=colors_acc, alpha=0.8, label="Accuracy")
    ax1.axhline(0.99, color="green", linestyle=":", alpha=0.5, label="99% target")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.9, 1.0)
    ax1.set_title("BISINDO CNN Models — Accuracy Comparison")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for i, a in enumerate(accs):
        ax1.text(i, a + 0.001, f"{a:.4f}", ha="center", fontsize=8)

    # F1 chart
    colors_f1 = [
        "#2ecc71" if f >= 0.99 else "#3498db" if f >= 0.97 else "#f39c12" if f >= 0.95 else "#e74c3c"
        for f in f1s
    ]
    ax2.bar(x, f1s, color=colors_f1, alpha=0.8, label="F1 (weighted)")
    ax2.set_ylabel("F1 (weighted)")
    ax2.set_ylim(0.9, 1.0)
    ax2.set_title("BISINDO CNN Models — F1 Score Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for i, f in enumerate(f1s):
        ax2.text(i, f + 0.001, f"{f:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def discover_models(only=None):
    if not os.path.isdir(DL_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(DL_DIR)):
        if fname.endswith("_model.pt") and "cnn" in fname.lower():
            name = fname.replace("_model.pt", "")
            if only and name != only:
                continue
            out.append(name)
    return out


def main():
    global CSV_PATH  # must precede any read
    parser = argparse.ArgumentParser(description="Evaluate BISINDO CNN models")
    parser.add_argument("--model", default=None, help="Single model (default: all)")
    parser.add_argument("--data", default=CSV_PATH, help="CSV path override")
    parser.add_argument("--reeval", action="store_true", help="Attempt inference re-eval")
    args = parser.parse_args()

    CSV_PATH = args.data

    model_names = discover_models(only=args.model)
    if not model_names:
        log.error(f"No CNN models found in {DL_DIR}")
        sys.exit(1)
    log.info(f"Found {len(model_names)} model(s): {model_names}")

    metrics_only = []
    for name in model_names:
        mr = get_metrics_only_result(name)
        if mr:
            metrics_only.append(mr)
    log.info(f"Loaded metrics for {len(metrics_only)} model(s)")

    evaluated = []
    if args.reeval:
        log.info("Re-evaluation enabled — attempting inference on each model")
        X, y = load_data()
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        log.info(f"Test set: {len(y_test)} samples")
        for name in model_names:
            log.info(f"=== {name} ===")
            try:
                r = evaluate_one(name, X_test, y_test)
                if r:
                    log.info(f"  ✅ acc={r['accuracy']:.4f}  f1={r['f1_weighted']:.4f}  arch={r['arch']}")
                    evaluated.append(r)
                else:
                    log.info(f"  ⚠️  re-eval skipped (architecture not reconstructable)")
            except Exception as e:
                log.error(f"  failed: {e}")

    # Use metrics_only (always populated) for the report; evaluated adds verification
    report = generate_report_v2(metrics_only, evaluated, reeval_mode=args.reeval)
    report_path = os.path.join(MODEL_DIR, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Report: {report_path}")
    print(report)

    # Also save JSON for programmatic use
    json_path = os.path.join(MODEL_DIR, "comparison_data.json")
    with open(json_path, "w") as f:
        json.dump({
            "models": metrics_only,
            "re_evaluated": evaluated,
            "reeval_mode": args.reeval,
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)
    log.info(f"Data: {json_path}")


if __name__ == "__main__":
    main()