"""Thesis figure for the 48 kHz held-out test of mesh -> latent (FFM_3_8): error and reduction by frequency.

Reads the per-frequency curves already stored in results/mesh2z_48k_v1/{dev,test}_results.json (no model is run and the
test set is not evaluated again): for each row, the RMS error over the 2652 directions at elevation >= -45 deg in each
bin 1-85 (187.5 Hz - 15.94 kHz), averaged over ears. Left: the curves of the held-out subjects. Right: the reduction
relative to the training mean, per bin, for the held-out subjects (solid) and the development subjects (dashed).

    python plot_mesh48k_curves.py <results/mesh2z_48k_v1> <output stem>
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
test = json.load(open(os.path.join(src, 'test_results.json')))['rows']
dev = json.load(open(os.path.join(src, 'dev_results.json')))['rows']
f = np.arange(1, 86) * 48000 / 256 / 1000            # kHz, bins 1-85
c = lambda rows, k: np.asarray(rows[k]['curve_el45'], float)
for rows in (test, dev):
    assert all(len(r['curve_el45']) == 85 for r in rows.values())

ROWS = [('training mean', 'Training mean', 'k', '-', 1.6),
        ('simulated deviation + measured mean', 'Own simulation, used directly', '0.55', '-', 1.3),
        ('geometry', 'Geometry', '#1f77b4', '-', 1.6),
        ('simulation', 'Own simulation, learned', '#ff7f0e', '-', 1.6),
        ('AE reconstruction (reference)', 'Autoencoder reconstruction', '#2ca02c', '--', 1.3)]

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9.5,
                     'legend.fontsize': 7.5, 'savefig.facecolor': 'white'})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), layout='constrained')
for key, name, col, ls, lw in ROWS:
    ax[0].plot(f, c(test, key), color=col, ls=ls, lw=lw, label=name)
ax[0].set_ylabel('RMS error over directions (dB)')
ax[0].set_title('(a) held-out subjects (36)', loc='left')

m_t, m_d = c(test, 'training mean'), c(dev, 'training mean')
for key, name, col in (('geometry', 'Geometry', '#1f77b4'), ('simulation', 'Own simulation, learned', '#ff7f0e')):
    ax[1].plot(f, 100 * (1 - c(test, key) / m_t), color=col, lw=1.4)
    ax[1].plot(f, 100 * (1 - c(dev, key) / m_d), color=col, lw=1.0, ls='--')
ax[1].axhline(0, color='0.4', lw=0.7)
ax[1].set_ylabel('reduction from mean (%)')
ax[1].set_title('(b) reduction by frequency', loc='left')

for a in ax:
    a.axvspan(6, 12, color='0.92', zorder=0)
    a.set_xlim(0, 16); a.set_xlabel('frequency (kHz)'); a.grid(alpha=.3, lw=.5)
h0, l0 = ax[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h0 += [Line2D([], [], color='0.3', lw=1.4), Line2D([], [], color='0.3', lw=1.0, ls='--')]
l0 += ['(b) held-out subjects', '(b) development subjects']
fig.legend(h0, l0, loc='outside lower center', ncol=4, frameon=False)
fig.savefig(out + '.png', dpi=300, bbox_inches='tight', pad_inches=0.04)
fig.savefig(out + '.pdf', bbox_inches='tight', pad_inches=0.04)
print('->', out + '.png / .pdf')

# numbers quoted in the caption and text: reduction of the band-averaged curve, held-out subjects
for lo, hi in ((0.1875, 2.0), (2.0, 5.0), (5.0, 6.0), (6.0, 12.0), (12.0, 16.0)):
    k = (f >= lo) & (f < hi) if hi < 16 else (f >= lo)
    print(f'{lo:5.2f}-{hi:5.2f} kHz  geometry {100 * (1 - c(test, "geometry")[k].mean() / m_t[k].mean()):5.1f}%'
          f'  simulation {100 * (1 - c(test, "simulation")[k].mean() / m_t[k].mean()):5.1f}%')
