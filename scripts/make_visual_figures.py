#!/usr/bin/env python3
"""Generate data-driven figures for the BISINDO Bridge journal.

All values plotted come from dataset/landmarks_2hands.csv (the exact
training corpus, 26,192 rows). No invented numbers.
Outputs to docs/jurnal-data/figures/:
  - landmark_extraction.png  (two real hands, 21 landmarks + connections)
  - dataset_composition.png  (samples per letter, A=1192, B-Z=1000)
  - norm_stages.png          (3-stage normalization applied to one real sample)
"""
import csv
import math
import os
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "dataset", "landmarks_2hands.csv")
OUT = os.path.join(ROOT, "docs", "jurnal-data", "figures")
os.makedirs(OUT, exist_ok=True)

# Paper palette (matches TikZ definecolor blocks in main.tex)
C_LANDMARK = (0, 136/255, 136/255)   # cLandmark
C_NORM     = (221/255, 132/255, 31/255)  # cNorm
C_MODEL    = (178/255, 24/255, 43/255)   # cModel
C_CAPTURE  = (33/255, 102/255, 172/255)  # cCapture

GRAY_DARK = "0.30"   # ~TikZ black!70
GRAY_MID  = "0.40"   # ~TikZ black!60
GRAY_LT   = "0.60"   # ~TikZ black!40

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
TIPS = [4, 8, 12, 16, 20]

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})


def load_sample(letter, want_hands=2):
    """Return (h1, h2) as lists of (x, y) for the first matching row."""
    with open(CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        i_num_hands = header.index("num_hands")
        i_letter = header.index("letter")
        # hand 1: lm0_x..lm20_y ; hand 2: h2_lm0_x..h2_lm20_y
        h1_cols = [(header.index(f"lm{i}_x"), header.index(f"lm{i}_y")) for i in range(21)]
        h2_cols = [(header.index(f"h2_lm{i}_x"), header.index(f"h2_lm{i}_y")) for i in range(21)]
        for row in reader:
            if row[i_letter] == letter and row[i_num_hands] == str(want_hands):
                h1 = [(float(row[ix]), float(row[iy])) for ix, iy in h1_cols]
                h2 = [(float(row[ix]), float(row[iy])) for ix, iy in h2_cols]
                return h1, h2
    raise RuntimeError(f"no {want_hands}-hand sample for letter {letter}")


def normalize_stages(h):
    """Apply the paper's stage 1 (wrist translate) and stage 2 (size scale)."""
    wrist = h[0]
    t = [(x - wrist[0], y - wrist[1]) for x, y in h]
    dmax = max(math.hypot(x, y) for x, y in t)
    s = [(x / dmax, y / dmax) for x, y in t]
    return h, t, s


def draw_hand(ax, pts, title, sub=None):
    xs, ys = zip(*pts)
    for a, b in HAND_CONNECTIONS:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "-", color=C_LANDMARK,
                lw=1.4, zorder=2, solid_capstyle="round")
    ax.scatter(xs, ys, s=22, color=C_LANDMARK, zorder=3)
    ax.scatter([xs[i] for i in TIPS], [ys[i] for i in TIPS], s=40,
               color=C_NORM, zorder=4, edgecolors="black", linewidths=0.6)
    ax.scatter([xs[0]], [ys[0]], s=55, color=C_MODEL, zorder=4,
               edgecolors="black", linewidths=0.6)
    ax.set_title(title, fontsize=9.5, pad=3)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color=GRAY_MID)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def pad_to_square(ax, pts, margin=0.08):
    xs, ys = zip(*pts)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)  # inverted y already


# ---------------------------------------------------------------- fig: landmarks
def fig_landmarks():
    h1, h2 = load_sample("B", 2)
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.0))
    for ax, pts, name in ((axes[0], h1, "Hand 1"), (axes[1], h2, "Hand 2")):
        draw_hand(ax, pts, name)
        pad_to_square(ax, pts)
        ax.text(0.02, 0.02, "0", transform=ax.transAxes, fontsize=7,
                color=C_MODEL, fontweight="bold")
    # joint legend rendered as text lines
    axes[0].annotate("wrist (0)", xy=(h1[0][0], h1[0][1]),
                     xytext=(h1[0][0] + 0.12, h1[0][1] - 0.16),
                     fontsize=7.5, color=C_MODEL,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=C_MODEL))
    axes[0].annotate("fingertip (8)", xy=(h1[8][0], h1[8][1]),
                     xytext=(h1[8][0] + 0.10, h1[8][1] + 0.10),
                     fontsize=7.5, color=C_NORM,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=C_NORM))
    fig.suptitle("21 anatomical landmarks per hand, tracked by MediaPipe Hands",
                 fontsize=10, y=0.99)
    fig.text(0.5, 0.005,
             "per hand: 21 landmarks $\\times$ 2 coordinates $(x,y)$ = 42 numbers"
             "   |   two hands: $21 \\times 2 \\times 2$ = 84 features per frame",
             ha="center", fontsize=8.5, color=GRAY_DARK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(OUT, "landmark_extraction.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig: dataset
def fig_dataset():
    with open(CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        i_letter = header.index("letter")
        counts = collections.Counter(row[i_letter] for row in reader)
    letters = [chr(ord("A") + i) for i in range(26)]
    vals = [counts[l] for l in letters]
    colors = [C_MODEL if l == "A" else C_LANDMARK for l in letters]

    fig, ax = plt.subplots(figsize=(6.69, 2.35))
    bars = ax.bar(letters, vals, color=colors, width=0.72)
    ax.axhline(1000, color=GRAY_LT, lw=0.8, ls="--", zorder=0)
    ax.text(25.4, 1012, "1,000", fontsize=7.5, color=GRAY_MID, ha="right")
    ax.annotate("1,192", xy=(0, 1192), xytext=(0.02, 1235),
                fontsize=8, color=C_MODEL, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_MODEL))
    ax.text(2.5, 1240,
            "letter A: first capture batch (slightly larger)",
            fontsize=8, color=GRAY_MID)
    ax.set_ylim(0, 1330)
    ax.set_ylabel("samples", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_composition.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig: norm stages
def fig_norm_stages():
    h1, h2 = load_sample("B", 2)
    stages_h1 = normalize_stages(h1)
    stages_h2 = normalize_stages(h2)

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.6))
    titles = [
        ("1) Raw capture", "position and scale differ"),
        ("2) Wrist translation  $p_i' = p_i - p_0$", "both wrists at origin"),
        ("3) Hand-size scaling  $\\hat{p}_i = p_i' / d_{\\max}$", "both hands unit size"),
    ]
    for k in range(3):
        ax = axes[k]
        for stages, color, ls in ((stages_h1, C_LANDMARK, "-"),
                                  (stages_h2, C_CAPTURE, "-")):
            pts = stages[k]
            xs, ys = zip(*pts)
            for a, b in HAND_CONNECTIONS:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], ls, color=color,
                        lw=1.3, zorder=2, solid_capstyle="round")
            ax.scatter(xs, ys, s=14, color=color, zorder=3)
        draw_axes_off = True
        pad_all = stages_h1[0] + stages_h2[0] if k == 0 else (
            stages_h1[1] + stages_h2[1] if k == 1 else stages_h1[2] + stages_h2[2])
        xs, ys = zip(*pad_all)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 0.10
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(titles[k][0], fontsize=9, pad=3)
        ax.text(0.5, -0.03, titles[k][1], transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color=GRAY_MID)
    fig.text(0.5, 0.015,
             "teal = hand 1, blue = hand 2   |   stage 3 (per-axis z-score) is statistical, not geometric",
             ha="center", fontsize=8, color=GRAY_DARK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "norm_stages.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_landmarks()
    fig_dataset()
    fig_norm_stages()
    for f in ("landmark_extraction.png", "dataset_composition.png", "norm_stages.png"):
        p = os.path.join(OUT, f)
        print(f, os.path.getsize(p), "bytes")
