"""Thesis figure: what the mesh features of mesh2z see.

Reads one subject's point samples written by sample_meshes.py and draws, in the head frame (mm, origin at the interaural
midpoint, +x forward, +y left, +z up), three side views of the left side:
  (a) the whole head sample above the neck cut used for the head embedding (z > -75 mm), with the ear crop marked;
  (b) the ear crop, all points within 45 mm of the left ear canal, coloured by their lateral offset from the canal;
  (c) the points that protrude more than 3 mm outward from the canal plane, the selection of the pinna-only experiment.
No model is run. Also prints the extent statistics over all subjects that the thesis text quotes.

    python plot_ear_crop.py <mesh2z_samples dir> <output stem> [--subject P0010]
"""
import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

Z_CUT, EAR_R, PINNA_DY = -75.0, 45.0, 3.0     # as in FFM_3_2_1 (Z_CUT, crop radius) and FFM_3_4 (PINNA_DY)


def load(path):
    d = np.load(path)
    return d['head_pts'].astype(float), d['earL_pts'].astype(float), d['ear_left_mm'].astype(float)


def extent_stats(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, 'P*.npz'))):
        d = np.load(f)
        for pts, canal in ((d['earL_pts'], d['ear_left_mm']), (d['earR_pts'], d['ear_right_mm'] * np.array([1, -1, 1]))):
            q = pts.astype(float) - canal.astype(float)
            out = q[q[:, 1] > PINNA_DY]
            rows.append((len(out) / len(q), np.percentile(out[:, 2], 99.5), -np.percentile(out[:, 2], 0.5),
                         -np.percentile(out[:, 0], 0.5)))
    a = np.array(rows)
    names = ('share of crop points protruding > 3 mm', 'top of protruding points above canal (mm)',
             'bottom below canal (mm)', 'back behind canal (mm)')
    print(f'{len(a)} ears')
    for k, n in enumerate(names):
        print(f'  {n:42s} median {np.median(a[:, k]):6.2f}  5-95% [{np.percentile(a[:, k], 5):6.2f}, {np.percentile(a[:, k], 95):6.2f}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--subject', default='P0010')
    a = ap.parse_args()
    head, ear, canal = load(os.path.join(a.root, f'{a.subject}.npz'))
    q = ear - canal
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9.5,
                         'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.1), layout='constrained', gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

    # (a) whole head, left side, above the neck cut; ear crop on top
    h = head[(head[:, 2] > Z_CUT) & (head[:, 1] > 0)]
    ax[0].scatter(h[:, 0], h[:, 2], s=0.15, c='0.7', rasterized=True)
    ax[0].scatter(ear[:, 0], ear[:, 2], s=0.15, c='#1f77b4', rasterized=True)
    ax[0].axhline(Z_CUT, color='k', ls=':', lw=0.8)
    ax[0].text(-100, Z_CUT + 5, 'neck cut', fontsize=7)
    ax[0].plot(*canal[[0, 2]], 'k+', ms=7)
    ax[0].set_title('(a) head and ear crop', loc='left')

    # (b) ear crop coloured by lateral offset from the canal
    t = np.linspace(0, 2 * np.pi, 361)
    sc = ax[1].scatter(q[:, 0], q[:, 2], c=q[:, 1], s=0.3, cmap='viridis', vmin=-10, vmax=25, rasterized=True)
    ax[1].plot(EAR_R * np.cos(t), EAR_R * np.sin(t), 'r--', lw=0.9)
    ax[1].plot(0, 0, 'k+', ms=7)
    ax[1].set_title('(b) ear crop', loc='left')
    cb = fig.colorbar(sc, ax=ax[1], orientation='horizontal', shrink=0.8, pad=0.02, aspect=30)
    cb.set_label('outward from the canal (mm)', fontsize=8); cb.ax.tick_params(labelsize=7)

    # (c) protruding points (pinna-only experiment)
    sel = q[:, 1] > PINNA_DY
    ax[2].scatter(q[~sel, 0], q[~sel, 2], s=0.2, c='0.85', rasterized=True)
    ax[2].scatter(q[sel, 0], q[sel, 2], s=0.2, c='#d62728', rasterized=True)
    ax[2].plot(EAR_R * np.cos(t), EAR_R * np.sin(t), 'r--', lw=0.9)
    ax[2].plot(0, 0, 'k+', ms=7)
    ax[2].set_title('(c) more than 3 mm outward', loc='left')

    for k, axx in enumerate(ax):
        axx.set_aspect('equal'); axx.set_xlabel('forward (mm)', fontsize=8)
        axx.set_ylabel('up (mm)' if k == 0 else 'up from canal (mm)', fontsize=8)
        axx.tick_params(labelsize=7.5)
    ax[1].set_xlabel('forward from canal (mm)', fontsize=8); ax[2].set_xlabel('forward from canal (mm)', fontsize=8)
    fig.savefig(a.out + '.png', dpi=300, bbox_inches='tight', pad_inches=0.04); fig.savefig(a.out + '.pdf', bbox_inches='tight', pad_inches=0.04)
    print('->', a.out + '.png / .pdf')
    extent_stats(a.root)


if __name__ == '__main__':
    main()
