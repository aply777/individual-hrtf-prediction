#!/usr/bin/env python3
"""Check and pack the SYMARE head-torso-ears BEM arrays (SYMARE_HTE) for the Task 1 torso check.

Written 2026-10-11. Run after

    python preprocess_48k.py --only SYMARE_HTE --out ~/Documents/26S2/ELEC5021/data_48k_hte

Checks SYMARE_HTE against SYMARE_BEM (head and ears, from the frozen 48 kHz set): the same 61 subject
ids and direction grid, different source files; every array's SHA-256, shape, dtype and finiteness, and
no unlisted files; below 5 kHz the two simulations of one subject agree to within a few dB; pairing:
after removing each set's population mean, a subject's HTE deviation correlates with its own HE
deviation and not with anyone else's (raw spectra cannot show this: the torso shifts every subject by
about the same 2-3 dB, more than subjects differ); above 5.3 kHz HTE rolls off.
Writes verification_hte.json into the folder, then `<hte>_<date>.zip` beside it and its SHA-256.

    python pack_hte.py [--hte ~/Documents/26S2/ELEC5021/data_48k_hte] [--ref ~/Documents/26S2/ELEC5021/data_48k]
"""
import argparse
import datetime
import glob
import json
import os
import zipfile

import numpy as np

import preprocess_48k as P

BAND = (P.FREQS >= 200) & (P.FREQS <= 5000)          # Task 1's 0.2-5 kHz band: bins 2-26
ABOVE = P.FREQS >= 6000


def load(d, man, side):
    out = []
    for m in man:
        f = os.path.join(d, f'npy_{side}', m[side])
        assert P.sha256(f) == m[f'{side}_sha256'], f
        a = np.load(f)
        assert a.shape == (P.NBINS, P.NPIX) and a.dtype == np.float32 and np.isfinite(a).all(), f
        out.append(a)
    return np.stack(out).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hte', default=os.path.expanduser('~/Documents/26S2/ELEC5021/data_48k_hte'))
    ap.add_argument('--ref', default=P.DEFAULT_OUT)
    a = ap.parse_args()
    hte, ref = os.path.join(a.hte, 'SYMARE_HTE_48k'), os.path.join(a.ref, 'SYMARE_BEM_48k')
    mh, mr = (json.load(open(os.path.join(d, 'manifest.json'))) for d in (hte, ref))
    meta = json.load(open(os.path.join(hte, 'meta.json')))
    assert meta['git'].get('preprocess_48k_uncommitted') is False, 'SYMARE_HTE was not written by committed code'
    assert [m['id'] for m in mh] == [m['id'] for m in mr] and len(mh) == 61, 'subject ids differ'
    assert all(x['grid'] == y['grid'] for x, y in zip(mh, mr)), 'direction grids differ'
    assert all(x['source_sha256'] != y['source_sha256'] for x, y in zip(mh, mr)), 'an HTE source equals its HE source'
    assert all('/HTE05kHz/' in m['source'] for m in mh)
    listed = {os.path.join(hte, f'npy_{s}', m[s]) for m in mh for s in ('left', 'right')}
    found = set(glob.glob(os.path.join(hte, 'npy_*', '*')))
    assert found == listed, f'unlisted or missing files: {sorted(found ^ listed)[:5]}'

    rep = dict(created=datetime.datetime.now().isoformat(timespec='seconds'), n_subjects=len(mh),
               reference='SYMARE_BEM_48k (HE16kHz) in ' + a.ref, ild_band_hz=meta['ild_band_hz'])
    for side in ('left', 'right'):
        H, E = load(hte, mh, side), load(ref, mr, side)
        rms = np.sqrt(((H - E)[:, BAND] ** 2).mean((1, 2)))                       # per subject, own HE
        dev = lambda X: (lambda D: D / np.linalg.norm(D, axis=1, keepdims=True))((X[:, BAND] - X[:, BAND].mean(0)).reshape(len(X), -1))
        C = dev(H) @ dev(E).T                                                       # corr(HTE dev i, HE dev j)
        own = C.argmax(1) == np.arange(len(mh))
        off = C[~np.eye(len(mh), dtype=bool)]
        drop = (H - E)[:, ABOVE].mean((1, 2))
        rep[side] = dict(rms_hte_minus_he_0p2_5khz_db=dict(min=float(rms.min()), median=float(np.median(rms)), max=float(rms.max())),
                         deviation_corr_own_median=float(np.median(np.diag(C))), deviation_corr_others_median=float(np.median(off)),
                         nearest_he_is_own=int(own.sum()), not_own=[mh[i]['id'] for i in np.flatnonzero(~own)],
                         mean_hte_minus_he_above_6khz_db=dict(min=float(drop.min()), max=float(drop.max())))
        print(f"{side:>5}: rms(HTE - own HE) 0.2-5 kHz {rms.min():.2f}-{rms.max():.2f} dB (median {np.median(rms):.2f}); "
              f"deviation corr own {np.median(np.diag(C)):.2f} vs others {np.median(off):+.2f}; own HE nearest for "
              f"{own.sum()}/{len(mh)}; HTE - HE above 6 kHz {drop.max():+.1f} dB or lower")
        assert rms.max() < 6.0, 'HTE and HE disagree below 5 kHz'
        assert np.median(np.diag(C)) > 0.3 > abs(np.median(off)) and own.sum() >= 2 * len(mh) // 3, 'HTE/HE pairing'
        assert drop.max() < -10.0, 'HTE does not roll off above 6 kHz: wrong folder?'
    json.dump(rep, open(os.path.join(hte, 'verification_hte.json'), 'w'), indent=1)

    out = os.path.abspath(a.hte)
    zpath = f'{out}_{datetime.date.today().isoformat()}.zip'
    root = os.path.basename(out)
    files = sorted(f for f in glob.glob(os.path.join(out, '**', '*'), recursive=True) if os.path.isfile(f))
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in files:
            z.write(f, os.path.join(root, os.path.relpath(f, out)))
    with zipfile.ZipFile(zpath) as z:
        bad, n_npy = z.testzip(), sum(n.endswith('.npy') for n in z.namelist())
    if bad is not None or n_npy != 2 * len(mh):
        raise SystemExit(f'zip check failed: bad member {bad}, {n_npy} arrays')
    digest = P.sha256(zpath)
    open(zpath + '.sha256', 'w').write(f'{digest}  {os.path.basename(zpath)}\n')
    print(f'{zpath}\n  {len(files)} files ({n_npy} arrays), {os.path.getsize(zpath) / 2**20:.0f} MiB, CRC OK\n  SHA-256 {digest}')


if __name__ == '__main__':
    main()
