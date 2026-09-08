#!/usr/bin/env python3
r"""Clean matplotlib versions of the latency/figure-rate charts (real measured data).

Design rules (so text stays readable and NEVER overlaps when embedded):
- NO in-image "(a)"/"(b)" titles — the LaTeX subcaptions already label them.
- NO decorative vlines/total markers — the xlabel carries the mean.
- Fonts sized so that at the embedded width (0.58 / 0.33 \linewidth) text
  renders at roughly body size (~10-11 pt effective).
- Layout is computed with disjoint regions: in-bar label centered in the
  99% segment, ONNX label in the empty top-right quadrant.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEAL, RED, BLUE = '#008888', '#B2182B', '#2166AC'
OUT = '/home/kevin/bisindo-bridge/docs/jurnal-data/figures/'
plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans',
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ---- (a) per-frame cost: one horizontal stacked bar, 53.4 ms total ----
mp, onx = 52.87, 0.53
fig, ax = plt.subplots(figsize=(4.6, 1.45))
ax.barh([0], [mp], color=TEAL, alpha=0.55, edgecolor=TEAL, height=0.62, zorder=2)
ax.barh([0], [onx], left=[mp], color=RED, height=0.62, zorder=2)
ax.set_xlim(0, 62)
ax.set_ylim(-0.55, 1.5)
ax.set_yticks([])
# single-line label, centered inside the 99% segment (bar spans 0..52.87);
# kept short so it fits INSIDE the bar even at readable font size
ax.text(mp / 2, 0, 'MediaPipe: $\\approx$52.9 ms (99%)',
        ha='center', va='center', fontsize=12.5, color='#004c4c')
# ONNX label in the empty top-right quadrant, straight leader line to the sliver.
# va='top' + straight (arc3) connection: the elbow-style connector produced an
# elbow point INSIDE the text bbox, which stabbed the label with the arrow.
ax.annotate('ONNX 1D-CNN: 0.53 ms (1%)', xy=(mp + onx / 2, 0.28), xytext=(61.5, 1.5),
            ha='right', va='top', fontsize=12, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=RED, lw=1.2, shrinkA=1, shrinkB=2))
ax.set_xlabel('ms per frame  (mean 53.4 ms)', fontsize=13)
ax.tick_params(axis='x', labelsize=12)
fig.tight_layout()
fig.savefig(OUT + 'latency_cost.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---- (b) frame rate: three simple bars ----
names, vals = ['mean', 'median', 'P95'], [18.7, 20.4, 15.2]
fig2, ax2 = plt.subplots(figsize=(2.6, 2.4))
bars = ax2.bar(names, vals, color=BLUE, alpha=0.6, edgecolor=BLUE, width=0.55, zorder=2)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v}', ha='center',
             fontsize=14, fontweight='bold', color='#0f3a5f')
ax2.set_ylim(0, 24)
ax2.set_ylabel('frames per second', fontsize=12)
ax2.grid(axis='y', color='#dddddd', lw=0.7, zorder=0)
ax2.tick_params(axis='x', labelsize=12)
fig2.tight_layout()
fig2.savefig(OUT + 'latency_fps.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print('saved latency_cost.png + latency_fps.png')
