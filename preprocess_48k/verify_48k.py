#!/usr/bin/env python3
"""Independent checks of the 48 kHz arrays written by preprocess_48k.py.

Written 2026-10-10, for the 48 kHz version of the data. Run after preprocess_48k.py and before
packing. Every check either passes or stops the script; the results go to
`<out>/verification_report.json` and a figure to `<out>/figures/`.

1. Integrity: every array listed in each manifest exists, has its recorded SHA-256, is
   (129, 3072) float32 and finite; no array is unlisted; ids are unique.
2. Pairing: every SONICOM simulation has a measured counterpart, and SYMARE_BEM pairs 1:1 with
   SYMARE by id.
3. Regression against the 44.1 kHz pipeline: the same spectrum + STPS code, run at a file's native
   44.1 kHz, reproduces the existing 44.1 kHz arrays (SONICOM P0001, HUTUBS pp1).
4. Regression against the old CAR Lab 48 kHz export (`<ELEC5021>/SYMARE/CARLab_left|right`, dB floor
   1e-9): identical for every subject except those changed on purpose (grid fix CL154/CL159, ear swap
   CL003/CL088, repeated directions CL169), whose differences are reported.
5. HUTUBS upsampling: the per-subject resampling errors recorded in the manifest, summarised.
6. HEALPix mirror permutation: a bijection and an involution; saved beside the data.
7. Figure: direction-averaged mean spectrum of every dataset.

    python verify_48k.py [--out ~/Documents/26S2/ELEC5021/data_48k]
"""
import argparse
import glob
import json
import os

import healpy as hp
import numpy as np

import preprocess_48k as P

OLD_SONICOM = os.path.join(P.SONICOM, 'npy_left', 'SON_P0001_left_full.npy')
OLD_HUTUBS = os.path.join(P.DATA_A, 'HUTUBS', 'npy_left', 'HU_pp1_left_full.npy')
OLD_CARLAB = os.path.join(P.CARLAB_ROOT, 'CARLab_left'), os.path.join(P.CARLAB_ROOT, 'CARLab_right')
CARLAB_CHANGED = {'CL154': 'grid fix', 'CL159': 'grid fix', 'CL003': 'ear swap', 'CL088': 'ear swap',
                  'CL169': 'repeated directions merged'}


def check(cond, msg):
    if not cond:
        raise SystemExit(f'FAILED: {msg}')


def integrity(out):
    rep = {}
    for d in sorted(glob.glob(os.path.join(out, '*_48k'))):
        name = os.path.basename(d)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        man = json.load(open(os.path.join(d, 'manifest.json')))
        ids = [m['id'] for m in man]
        check(len(ids) == len(set(ids)) == meta['n_subjects'], f'{name}: ids/count')
        listed = set()
        for m in man:
            for side in ('left', 'right'):
                f = os.path.join(d, f'npy_{side}', m[side])
                listed.add(f)
                check(P.sha256(f) == m[f'{side}_sha256'], f'{name}/{m[side]}: SHA-256')
                a = np.load(f)
                check(a.shape == (P.NBINS, P.NPIX) and a.dtype == np.float32 and np.isfinite(a).all(),
                      f'{name}/{m[side]}: shape/dtype/finite')
        on_disk = set(glob.glob(os.path.join(d, 'npy_*', '*.npy')))
        check(on_disk == listed, f'{name}: unlisted or missing arrays: {sorted(on_disk ^ listed)[:5]}')
        rep[name] = dict(n_subjects=len(ids), arrays=len(listed), distinct_grids=meta['distinct_grids'],
                         excluded=meta.get('excluded', {}),
                         ears_swapped={m['id']: m['ears_swapped'] for m in man if 'ears_swapped' in m},
                         merged_repeats={m['id']: {k: m[k] for k in m if k.startswith(('merged', 'repeat', 'n_dir'))}
                                         for m in man if 'merged_repeated_directions' in m},
                         lab_remarks={m['id']: m['lab_remark'] for m in man if 'lab_remark' in m})
        print(f'  [1] {name:<16}{len(ids):>4} subjects, {len(listed)} arrays: hashes, shapes, finite OK')
    return rep


def pairing(out):
    ids = lambda n: {m['id'] for m in json.load(open(os.path.join(out, f'{n}_48k', 'manifest.json')))}
    son, sim, sym, bem = ids('SONICOM'), ids('SONICOM_SIM'), ids('SYMARE'), ids('SYMARE_BEM')
    check(sim <= son, f'SONICOM_SIM ids without a measured counterpart: {sorted(sim - son)}')
    check(sym == bem, 'SYMARE and SYMARE_BEM ids differ')
    print(f'  [2] pairing OK: {len(sim)} SONICOM simulations all have measurements; SYMARE/BEM {len(sym)} = {len(bem)}')
    return dict(sonicom_pairs=len(sim), symare_pairs=len(sym))


def regression_441(ref_file, sofa, tag):
    hL, hR, az, el, fs = P.read_sofa(sofa, expect_fs=44100.0)
    xyz = P.sph_deg_to_xyz(az, el)
    W = P.build_interpolation_matrix(xyz, P.HP_XYZ, spline_order=P.SPLINE_ORDER, smoothing=P.SMOOTHING)
    new = P.to_healpix(P.spectrum_db(hL, fs=44100), W)
    old = np.load(ref_file)
    d = float(np.abs(new - old).max())
    check(old.shape == new.shape and d < 1e-3, f'{tag}: 44.1 kHz regression max |diff| {d}')
    print(f'  [3] {tag}: new code at 44.1 kHz reproduces the existing array, max |diff| {d:.2e} dB')
    return d


def regression_carlab(out):
    res, same = {}, 0
    for sid, path in P.items_carlab():
        hL, hR, xyz, fs, extra = P.load_carlab(sid, path)
        swapped = 'ears_swapped' in extra
        mL, mR = P.spectrum_db(hL, floor=1e-9), P.spectrum_db(hR, floor=1e-9)
        xyz, hL, hR, mL, mR, _ = P.merge_repeated(xyz, hL, hR, mL, mR)
        W = P.build_interpolation_matrix(xyz, P.HP_XYZ, spline_order=P.SPLINE_ORDER, smoothing=P.SMOOTHING)
        newL = P.to_healpix(mL, W)
        oldL = np.load(os.path.join(OLD_CARLAB[1 if swapped else 0], f'{sid}_{"right" if swapped else "left"}_full.npy'))
        d = np.abs(newL - oldL)[:86]
        if sid in CARLAB_CHANGED:
            res[sid] = dict(reason=CARLAB_CHANGED[sid], max_db=round(float(d.max()), 3),
                            rms_db=round(float(np.sqrt((d ** 2).mean())), 3), compared_with_old='right ear' if swapped else 'left ear')
        else:
            check(d.max() < 1e-3, f'CARLab {sid}: differs from the old export, max {d.max():.4f} dB')
            same += 1
    print(f'  [4] CARLab: {same} subjects identical to the old 48 kHz export (< 1e-3 dB); changed on purpose:')
    for k, v in res.items():
        print(f'        {k}: {v}')
    check(same == 218 - len(CARLAB_CHANGED), 'CARLab: unexpected number of identical subjects')
    return dict(identical=same, changed=res)


def hutubs_resampling(out):
    man = json.load(open(os.path.join(out, 'HUTUBS_48k', 'manifest.json')))
    s = {k: max(m[k] for m in man) for k in man[0] if k.startswith('resample_') and not k.endswith('mean_db')}
    s.update({k: max(abs(m[k]) for m in man) for k in man[0] if k.endswith('mean_db')})
    print(f'  [5] HUTUBS upsampling, worst subject: ' + ', '.join(f'{k} {v:.4f}' for k, v in sorted(s.items())))
    return s


def mirror(out):
    th, ph = hp.pix2ang(P.NSIDE, np.arange(P.NPIX))
    idx = hp.ang2pix(P.NSIDE, th, (-ph) % (2 * np.pi))
    check(len(np.unique(idx)) == P.NPIX and np.array_equal(idx[idx], np.arange(P.NPIX)), 'mirror permutation')
    np.save(os.path.join(out, 'healpix_mirror_idx.npy'), idx)
    old = os.path.join(P.CARLAB_ROOT, 'healpix_mirror_idx.npy')
    same = bool(np.array_equal(np.load(old), idx)) if os.path.exists(old) else None
    print(f'  [6] mirror permutation: bijection and involution; equal to the existing file: {same}')
    return dict(equal_to_existing=same)


def figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d in sorted(glob.glob(os.path.join(out, '*_48k'))):
        arrs = [np.load(f).mean(1) for f in sorted(glob.glob(os.path.join(d, 'npy_*', '*.npy')))]
        ax.plot(P.FREQS / 1000, np.mean(arrs, 0), label=f'{os.path.basename(d)} (n={len(arrs) // 2})')
    ax.axvline(15.9375, color='k', lw=0.6, ls=':')
    ax.set(xlabel='kHz', ylabel='dB (mean over directions, ears, subjects)', title='48 kHz arrays: mean spectra')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    f = os.path.join(out, 'figures', 'mean_spectra_48k.png')
    fig.tight_layout()
    fig.savefig(f, dpi=130)
    print(f'  [7] figure: {f}')
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=P.DEFAULT_OUT)
    a = ap.parse_args()
    print(f'Verifying {a.out}')
    rep = dict(integrity=integrity(a.out), pairing=pairing(a.out),
               regression_44k1=dict(
                   sonicom_P0001=regression_441(OLD_SONICOM, os.path.join(P.SONICOM, 'dataset', 'P0001', 'HRTF', 'HRTF', '44kHz',
                                                                          'P0001_FreeFieldCompMinPhase_44kHz.sofa'), 'SONICOM P0001'),
                   hutubs_pp1=regression_441(OLD_HUTUBS, os.path.join(P.HUTUBS, 'pp1_HRIRs_measured.sofa'), 'HUTUBS pp1')),
               regression_carlab_48k=regression_carlab(a.out), hutubs_resampling=hutubs_resampling(a.out),
               mirror=mirror(a.out), figure=figure(a.out))
    json.dump(rep, open(os.path.join(a.out, 'verification_report.json'), 'w'), indent=1)
    print(f'\nAll checks passed. Report: {os.path.join(a.out, "verification_report.json")}')


if __name__ == '__main__':
    main()
