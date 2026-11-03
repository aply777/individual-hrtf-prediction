"""Thesis-size redraw of the Task 2 mode figures (no refitting).

Reads what Task2_ShapeModes.ipynb saved and draws the first two modes of each database at print
size (17 cm wide, 7 pt text), one figure per database:

    results/task2_results.json          shape changes, shares, statistics (committed)
    task2/output/task2_responses.npz    simulated / measured responses per mode (Colab output)
    preprocess_48k/split_48k.json       SYMARE development subjects (for mean landmark positions)
    ../SYMARE/3DShapeData.mat           SYMARE landmarks (for mean landmark positions)

Before drawing, every statistic in the JSON is recomputed from the npz (rms, cosine, ratio, per
band) and must agree; the direction planes and the mean landmark positions are rebuilt with the
notebook's own code.

    python task2/redraw_task2_figures.py      # -> results/task2_modes_{sonicom,symare}_thesis.png
"""
import json
import os

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(f'{REPO}/results/task2_results.json'))
NPZ = np.load(f'{REPO}/task2/output/task2_responses.npz')
SPLIT = f'{REPO}/preprocess_48k/split_48k.json'
LANDMARKS = os.path.join(os.path.dirname(REPO), 'SYMARE', '3DShapeData.mat')
WIDTH_IN = 17 / 2.54

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.dpi': 300})

# ---------------------------------------------------------------- directions (notebook Cell 1, verbatim logic)
_th, _ph = hp.pix2ang(16, np.arange(3072))
EL = 90 - np.degrees(_th)
AZ = (np.degrees(_ph) + 180) % 360 - 180
DIRS = np.flatnonzero(EL >= -45.0)
assert np.array_equal(DIRS, NPZ['dirs'])
POS = {p: j for j, p in enumerate(DIRS)}
_h = DIRS[np.abs(EL[DIRS]) < 1e-9]
H_PLANE = [(float(AZ[p]), POS[p]) for p in _h[np.argsort(AZ[_h])]]
_med = []
for e in np.unique(EL[DIRS]):
    ring = DIRS[EL[DIRS] == e]
    _med.append((float(e), POS[ring[np.argmin(np.abs(AZ[ring]))]]))
    _med.append((float(180 - e), POS[ring[np.argmin(180 - np.abs(AZ[ring]))]]))
M_PLANE = sorted(_med)
assert len(H_PLANE) == 64
F = NPZ['freqs_hz'] / 1000


def plane(r, P):
    return np.array([r[:, j] for _, j in P]).T


# ---------------------------------------------------------------- check the npz against the JSON
BANDS = {'all': (0, 99), '0.2-5 kHz': (0.2, 5), '6-12 kHz': (6, 12), '12-16 kHz': (12, 16)}
for db in ('SONICOM', 'SYMARE'):
    for m in RES['databases'][db]['modes']['modes']:
        s = NPZ[f'{db}_mode{m["mode"]}_sim'].astype(float); me = NPZ[f'{db}_mode{m["mode"]}_meas'].astype(float)
        for b, (lo, hi) in BANDS.items():
            k = (F >= lo) & (F <= hi); a, c = s[k], me[k]
            got = dict(gain_sim_db=np.sqrt((a ** 2).mean()), gain_meas_db=np.sqrt((c ** 2).mean()),
                       cos=(a * c).sum() / np.sqrt((a ** 2).sum() * (c ** 2).sum()))
            got['ratio'] = got['gain_meas_db'] / got['gain_sim_db']
            for key, v in got.items():
                assert abs(v - m['stats'][b][key]) < 1e-6, (db, m['mode'], b, key, v, m['stats'][b][key])
print('npz reproduces every statistic in task2_results.json')

# ---------------------------------------------------------------- SYMARE mean landmark positions (notebook Cells 5-6)
EAR_P, MIDSAG, EAR_N, PAIR_N, PAIR_P = list(range(13)), list(range(13, 18)), list(range(18, 31)), 31, 32
SYM_HEAD = MIDSAG + [PAIR_P, PAIR_N]
HEAD_LABELS = ['13', 'top', 'front', 'chin', 'back', 'eye (ipsi)', 'eye (contra)']
split = json.load(open(SPLIT))
DEV_Y = sorted(split['SYMARE']['train'] + split['SYMARE']['val'])
assert len(DEV_Y) == 46
LM = np.stack(list(sio.loadmat(LANDMARKS, struct_as_record=False, squeeze_me=True)['A'])).astype(np.float64)
assert LM.shape == (61, 33, 3)


def mirror_config(P):
    Q = P.copy(); Q[:, 1] *= -1
    Q[EAR_P], Q[EAR_N] = P[EAR_N] * [1, -1, 1], P[EAR_P] * [1, -1, 1]
    Q[PAIR_P], Q[PAIR_N] = P[PAIR_N] * [1, -1, 1], P[PAIR_P] * [1, -1, 1]
    return Q


Xy = np.array([np.concatenate([c[SYM_HEAD].ravel(), c[EAR_P].ravel()])
               for s in DEV_Y for c in (LM[int(s[-2:]) - 1], mirror_config(LM[int(s[-2:]) - 1]))])
assert Xy.shape == (92, 60)
HEAD_MEAN = Xy[:, :21].mean(0).reshape(7, 3)
EAR_MEAN = Xy[:, 21:].mean(0).reshape(13, 3)


# ---------------------------------------------------------------- panels
def fmt(x, unit):
    if unit == 'mm²':
        t = f'{x:+.0f} mm²'
    else:
        t = '0.0 mm' if abs(x) < 0.05 else f'{x:+.1f} mm'
    return t.replace('-', '−')


def son_shape(fig, cell, m, names):
    ax = fig.add_subplot(cell)
    u = np.array(m['shape_change']); units = ['mm'] * 8 + ['mm²']
    ax.barh(range(9), u, color=['C1'] * 3 + ['C2'] * 6, height=0.7)
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'{n[0].upper() + n[1:]} ({fmt(x, un)})' for n, x, un in zip(names, m['shape_change_units'], units)])
    for t, c in zip(ax.get_yticklabels(), ['C1'] * 3 + ['C2'] * 6):
        t.set_color('#b35900' if c == 'C1' else '#1e7a1e')
    ax.invert_yaxis(); ax.axvline(0, c='k', lw=0.6); ax.grid(axis='x', alpha=0.3)
    ax.set_xlabel('Change at +1 SD of the mode score\n(SD of each measure; orange: head, green: ear)')
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.24, pos.y0, pos.width - 0.24, pos.height])  # room for labels
    return ax


def landmarks(ax, P, d, norm, labels, scale=5, fs=6, offsets=None):
    tips = np.r_[P[:, [0, 2]], P[:, [0, 2]] + scale * d[:, [0, 2]]]
    ax.quiver(P[:, 0], P[:, 2], d[:, 0] * scale, d[:, 2] * scale, color='k', angles='xy', scale_units='xy', scale=1,
              width=0.008, headwidth=4, headlength=5, zorder=2)
    sc = ax.scatter(P[:, 0], P[:, 2], s=16, c=d[:, 1], cmap='coolwarm', norm=norm, edgecolors='k', linewidths=0.4, zorder=3)
    c = P[:, [0, 2]].mean(0)
    for (x, _, z), lab in zip(P, labels):
        if lab is None:
            continue
        v = np.array([x, z]) - c; v = v / (np.linalg.norm(v) + 1e-9)
        off = offsets[lab] if offsets and lab in offsets else tuple(8 * v)
        ax.annotate(lab, (x, z), xytext=off, textcoords='offset points', fontsize=fs, ha='center', va='center')
    lo, hi = tips.min(0), tips.max(0); pad = 0.14 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.tick_params(labelsize=6)
    return sc


def sym_shape(fig, cell, m, names):
    u = np.array(m['shape_change']); h = u[:21].reshape(7, 3); e = u[21:].reshape(13, 3)
    lim = np.abs(np.r_[h[:, 1], e[:, 1]]).max()
    norm = TwoSlopeNorm(0, -lim, lim)
    sub = cell.subgridspec(1, 2, wspace=0.45, width_ratios=[1.15, 1])
    a0 = fig.add_subplot(sub[0]); a1 = fig.add_subplot(sub[1])
    labs = ['13', 'top', 'front', 'chin', 'back', 'eyes', None]   # contralateral eye coincides with the ipsilateral one here
    landmarks(a0, HEAD_MEAN, h, norm, labs,
              offsets={'13': (2, 8), 'top': (11, 3), 'front': (12, -7), 'chin': (0, -8), 'back': (-4, -9), 'eyes': (-14, -6)})
    a0.text(0.03, 0.03, 'Head', transform=a0.transAxes, va='bottom', fontsize=7, fontweight='bold')
    a0.set_xlabel('x (mm, forward)'); a0.set_ylabel('z (mm, up)')
    sc = landmarks(a1, EAR_MEAN, e, norm, [str(i) for i in range(13)], fs=5.5)
    a1.text(0.03, 0.03, 'Pinna', transform=a1.transAxes, va='bottom', fontsize=7, fontweight='bold')
    a1.set_xlabel('x (mm, forward)')
    cb = fig.colorbar(sc, ax=[a0, a1], fraction=0.035, pad=0.03, aspect=18)
    cb.ax.set_title('Lateral\n(mm)', fontsize=6, pad=3); cb.ax.tick_params(labelsize=6)
    return a0


def draw(db, shape_fn, height_in, top_ratio, fname, hspace=0.55, bottom=0.055):
    D = RES['databases'][db]; names = D['input_names']
    fig = plt.figure(figsize=(WIDTH_IN, height_in))
    outer = fig.add_gridspec(4, 1, height_ratios=[top_ratio, 1.1, top_ratio, 1.1], hspace=hspace,
                             left=0.08, right=0.90, top=0.955, bottom=bottom)
    for k in range(2):
        m = D['modes']['modes'][k]; s = NPZ[f'{db}_mode{k + 1}_sim']; me = NPZ[f'{db}_mode{k + 1}_meas']
        a = m['stats']['all']
        top = outer[2 * k].subgridspec(1, 2, wspace=0.40 if db == 'SYMARE' else 0.30, width_ratios=[1.3, 1])
        shape_fn(fig, top[0], m, names)
        fig.text(0.02, outer[2 * k].get_position(fig).y1 + 0.022, f'Mode {k + 1} — {m["share_of_sim_fit"]:.0%} of the simulated fit',
                 fontsize=8, fontweight='bold', ha='left', va='bottom')
        ra = fig.add_subplot(top[1])
        ra.axvspan(6, 12, color='0.92', zorder=0)
        ra.plot(F, np.sqrt((s.astype(float) ** 2).mean(1)), c='C3', lw=1, label='Simulated')
        ra.plot(F, np.sqrt((me.astype(float) ** 2).mean(1)), c='C0', lw=1, label='Measured')
        ra.set_xlabel('Frequency (kHz)'); ra.set_ylabel('RMS over directions (dB)'); ra.set_xlim(0, 16)
        ra.set_xticks([0, 4, 8, 12, 16]); ra.set_ylim(bottom=0); ra.grid(alpha=0.3)
        ra.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
        ra.text(0.98, 0.04, f'cosine {a["cos"]:+.2f}\nmeasured / simulated {a["ratio"]:.2f}'.replace('-', '−'),
                transform=ra.transAxes, ha='right', va='bottom', fontsize=6.5)
        maps = outer[2 * k + 1].subgridspec(1, 5, wspace=0.14, width_ratios=[1, 1, 1, 1, 0.06])
        v = np.percentile(np.abs(np.concatenate([plane(s, H_PLANE), plane(s, M_PLANE)], 1)), 99)
        for j, (dom, P, lab) in enumerate((('Simulated', H_PLANE, 'horizontal'), ('Measured', H_PLANE, 'horizontal'),
                                           ('Simulated', M_PLANE, 'median'), ('Measured', M_PLANE, 'median'))):
            ax = fig.add_subplot(maps[j])
            r = s if dom == 'Simulated' else me
            im = ax.pcolormesh([x for x, _ in P], F, plane(r, P), cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest',
                               rasterized=True)
            ax.set_title(f'{dom}, {lab} plane', fontsize=7)
            ax.set_yticks([2, 6, 10, 14])
            if j == 0:
                ax.set_ylabel('Frequency (kHz)')
            else:
                ax.set_yticklabels([])
            if lab == 'horizontal':
                ax.set_xticks([-90, 0, 90]); ax.set_xticklabels(['−90', '0', '90']); ax.set_xlabel('Azimuth (°)')
            else:
                ax.set_xticks([0, 90, 180]); ax.set_xticklabels(['0\nfront', '90\nup', '180\nback'])
                ax.set_xlabel('Polar angle (°)')
        cb = fig.colorbar(im, cax=fig.add_subplot(maps[4])); cb.set_label('dB per +1 SD'); cb.ax.tick_params(labelsize=6)
    out = f'{REPO}/results/{fname}'
    fig.savefig(out); plt.close(fig)
    print('->', out)


draw('SONICOM', son_shape, 8.7, 1.0, 'task2_modes_sonicom_thesis.png', hspace=0.72, bottom=0.07)
draw('SYMARE', sym_shape, 8.8, 1.3, 'task2_modes_symare_thesis.png')
