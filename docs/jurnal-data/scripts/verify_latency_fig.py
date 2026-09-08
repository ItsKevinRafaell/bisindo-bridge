#!/usr/bin/env python3
"""Verify figure (a) layout: assert no text-text / text-bar overlaps.

Rebuilds the (a) figure with the exact same code as make_latency_figures.py,
then checks pixel bounding boxes with the real renderer.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEAL, RED = '#008888', '#B2182B'
mp, onx = 52.87, 0.53

fig, ax = plt.subplots(figsize=(4.6, 1.45))
ax.barh([0], [mp], color=TEAL, alpha=0.55, edgecolor=TEAL, height=0.62, zorder=2)
ax.barh([0], [onx], left=[mp], color=RED, height=0.62, zorder=2)
ax.set_xlim(0, 62)
ax.set_ylim(-0.55, 1.6)
ax.set_yticks([])
t_mp = ax.text(mp / 2, 0, 'MediaPipe: $\\approx$52.9 ms (99%)',
               ha='center', va='center', fontsize=12.5, color='#004c4c')
ann = ax.annotate('ONNX 1D-CNN: 0.53 ms (1%)', xy=(mp + onx / 2, 0.28),
                  xytext=(61.5, 1.5), ha='right', va='top', fontsize=12,
                  color=RED, fontweight='bold',
                  arrowprops=dict(arrowstyle='-', color=RED, lw=1.2, shrinkA=1, shrinkB=2))
ax.set_xlabel('ms per frame  (mean 53.4 ms)', fontsize=13)
ax.tick_params(axis='x', labelsize=12)
fig.tight_layout()
fig.canvas.draw()
ren = fig.canvas.get_renderer()

def bbox_text(t):
    return t.get_window_extent(ren)

def inter(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx > 0 and dy > 0

bar = ax.patches[0].get_window_extent(ren)
axes_bb = ax.get_window_extent(ren)
bb_mp, bb_on = bbox_text(t_mp), bbox_text(ann.get_bbox_patch()) if ann.get_bbox_patch() is None and False else bbox_text(ann)

ok = True
# 1. two labels must not overlap each other
if inter(bb_mp, bb_on):
    ok = False; print('FAIL: MediaPipe label vs ONNX label overlap')
else:
    print(f'OK   labels disjoint (gap_x={max(bb_mp.x0,bb_on.x0)-min(bb_mp.x1,bb_on.x1):.0f}px)')
# 2. MediaPipe label fully inside its 99% bar segment
if bb_mp.x0 >= bar.x0 - 1 and bb_mp.x1 <= bar.x1 + 1 and bb_mp.y0 >= bar.y0 - 1 and bb_mp.y1 <= bar.y1 + 1:
    print('OK   MediaPipe label sits inside the bar')
else:
    ok = False; print(f'FAIL: MediaPipe label outside bar: label=({bb_mp.x0:.0f},{bb_mp.x1:.0f}) bar=({bar.x0:.0f},{bar.x1:.0f})')
# 3. ONNX label fully inside the axes area (nothing clipped at the right edge)
if bb_on.x0 >= axes_bb.x0 and bb_on.x1 <= axes_bb.x1 and bb_on.y1 <= axes_bb.y1:
    print('OK   ONNX label inside axes')
else:
    ok = False; print(f'FAIL: ONNX label clipped: label=({bb_on.x0:.0f},{bb_on.x1:.0f}) axes=({axes_bb.x0:.0f},{axes_bb.x1:.0f})')
# 4. leader line must not stab the label: arrow end (xy in display) vs label bbox
xy_disp = ax.transData.transform((mp + onx / 2, 0.28))
seg_hits_label = bb_on.x0 <= xy_disp[0] <= bb_on.x1 and bb_on.y0 <= xy_disp[1] <= bb_on.y1
if not seg_hits_label:
    print('OK   leader line endpoint is outside the ONNX label box')
else:
    ok = False; print('FAIL: arrow endpoint inside label bbox')
print('RESULT:', 'PASS' if ok else 'FAIL')
