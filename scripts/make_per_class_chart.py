#!/usr/bin/env python3
"""Regenerate the per-class P/R/F1 chart for the BISINDO Bridge journal.

The old chart was authored ~15.9in wide and shrunk to one IEEE column
(~3.5in) in LaTeX -> all text shrank ~4.5x and became unreadable.
This script authors the figure at FINAL physical size (3.5in) with
true-point fonts, so width=\\columnwidth renders 1:1 and every label
prints at its authored point size.

Data source: docs/jurnal-data/cnn_2hand_per_class.csv (test-set support
150/letter, A=179). Run from repo root:
    python3 scripts/make_per_class_chart.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "jurnal-data" / "cnn_2hand_per_class.csv"
OUT = ROOT / "docs" / "jurnal-data" / "figures" / "cnn_2hand_per_class_chart.png"

rows = []
with CSV.open(newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((r["letter"], float(r["precision"]), float(r["recall"]), float(r["f1"])))

letters = [r[0] for r in rows]
p = np.array([r[1] for r in rows])
rc = np.array([r[2] for r in rows])
f1 = np.array([r[3] for r in rows])

# Final physical size == deployed text block (2cm margins -> 17cm = 6.69in)
# -> width=\textwidth renders 1:1 and every label prints at authored size.
fig, ax = plt.subplots(figsize=(6.69, 2.9), dpi=300)

x = np.arange(len(letters))
w = 0.27
ax.bar(x - w, p, w, label="Precision", color="#4C72B0", linewidth=0)
ax.bar(x, rc, w, label="Recall", color="#DD8452", linewidth=0)
ax.bar(x + w, f1, w, label="F1", color="#55A868", linewidth=0)

ax.axhline(0.97, color="black", linewidth=0.6, linestyle=(0, (4, 3)), alpha=0.55, zorder=0)

# Review instruction: chart scale exactly 0.9 - 1.0.
ax.set_ylim(0.9, 1.0)
ax.set_yticks([0.90, 0.925, 0.95, 0.975, 1.00])
ax.set_yticklabels(["0.90", "0.925", "0.95", "0.975", "1.00"])
ax.set_ylabel("Score", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(letters, fontsize=9)
ax.tick_params(axis="y", labelsize=9, length=2)
ax.tick_params(axis="x", length=0)
ax.set_axisbelow(True)
ax.grid(axis="y", color="black", alpha=0.12, linewidth=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(0.6)

ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
          frameon=False, fontsize=10, handlelength=1.2,
          columnspacing=1.2, borderaxespad=0.0)

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
