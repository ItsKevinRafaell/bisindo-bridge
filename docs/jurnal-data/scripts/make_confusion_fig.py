#!/usr/bin/env python3
"""Clean 26x26 confusion matrix heatmap from real eval JSON."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open('/home/kevin/bisindo-bridge/docs/jurnal-data/cnn_2hand_confusion.json'))
labels, M = d['labels'], np.array(d['matrix'])

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.4, 6.2))
# row-normalized shading so rare letters are readable, diagonal pops
Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
im = ax.imshow(Mn, cmap='BuGn', vmin=0, vmax=1, aspect='equal')

ax.set_xticks(range(26)); ax.set_yticks(range(26))
ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('predicted letter', fontsize=10)
ax.set_ylabel('true letter', fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# annotate every cell with count > 0 except the diagonal (off-diagonal errors only)
for i in range(26):
    for j in range(26):
        if i != j and M[i, j] > 0:
            ax.text(j, i, str(M[i, j]), ha='center', va='center',
                    fontsize=7, color='#B2182B', fontweight='bold')

# red boxes around the top-5 confusable pairs
top = d['top_pairs'][:5]
for tp in top:
    i, j = labels.index(tp['true']), labels.index(tp['pred'])
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor='#B2182B', lw=1.4))

cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('row-normalized rate', fontsize=9)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig('/home/kevin/bisindo-bridge/docs/jurnal-data/figures/confusion_2hand_clean.png',
            dpi=300, bbox_inches='tight')
print('saved confusion_2hand_clean.png;',
      'off-diag cells:', int((M > 0).sum() - np.trace(M > 0)),
      'total errors:', int(M.sum() - np.trace(M)))
