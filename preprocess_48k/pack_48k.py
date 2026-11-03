#!/usr/bin/env python3
"""Pack the verified 48 kHz arrays into one zip for Google Drive.

Written 2026-10-10, for the 48 kHz version of the data. Run after preprocess_48k.py and
verify_48k.py. Refuses to pack unless verification_report.json is present and newer than every
manifest. Writes README_DATA.md into the data folder, then `<out>_<date>.zip` beside it, tests every
member's CRC, checks the member count against the manifests, and writes the zip's SHA-256 to
`<zip>.sha256`.

    python pack_48k.py [--out ~/Documents/26S2/ELEC5021/data_48k]
"""
import argparse
import datetime
import glob
import json
import os
import zipfile

import preprocess_48k as P


def readme(out):
    rep = json.load(open(os.path.join(out, 'verification_report.json')))
    rows, notes = [], []
    for d in sorted(glob.glob(os.path.join(out, '*_48k'))):
        m = json.load(open(os.path.join(d, 'meta.json')))
        grids = list(m['distinct_grids'].values())
        rows.append(f"| `{os.path.basename(d)}` | {m['n_subjects']} | {len(grids)} ({', '.join(map(str, grids))}) | {m['source']} |")
    integ = rep['integrity']
    for name, r in integ.items():
        for k, v in r['ears_swapped'].items():
            notes.append(f'- `{name}` {k}: ear channels swapped back ({v}).')
        for k, v in r['excluded'].items():
            notes.append(f'- `{name}` {k}: excluded ({v}).')
        for k, v in r['merged_repeats'].items():
            notes.append(f'- `{name}` {k}: repeated directions merged ({v}).')
        for k, v in r['lab_remarks'].items():
            notes.append(f'- `{name}` {k}: lab remark "{v}" (kept).')
    carlab = rep['regression_carlab_48k']
    for k in ('CL154', 'CL159'):
        v = carlab['changed'][k]
        notes.append(f'- `CARLab_48k` {k}: interpolated from its own direction grid. The earlier CAR Lab exports '
                     f'(44.1 and 48 kHz) keyed the STPS cache on grid size and first direction and used CL126\'s '
                     f'grid for this subject (difference here: rms {v["rms_db"]} dB, max {v["max_db"]} dB).')
    text = f"""# HRTF arrays at 48 kHz

Created {datetime.date.today().isoformat()} by `hrtf-GitHub/preprocess_48k/` (preprocess_48k.py, verify_48k.py,
pack_48k.py). This is the **48 kHz version** of the data. It does not replace the 44.1 kHz arrays used for FFM26.

## Contents

| Folder | Subjects | Direction grids (subjects per grid) | Source |
|---|---:|---|---|
{chr(10).join(rows)}

Each folder holds `npy_left/<id>_left_full.npy` and `npy_right/<id>_right_full.npy`, plus `meta.json`
(settings) and `manifest.json` (per subject: source file and its SHA-256, grid hash, ILD and ITD at six
azimuths, output SHA-256s). `healpix_mirror_idx.npy` maps each pixel to its left-right mirror image.
`verification_report.json` records the checks below, and `figures/mean_spectra_48k.png` the mean spectra.

## Array convention

- (129, 3072) float32 per ear: dB magnitude, 20 log10(|H| + 1e-6), at f_k = k x 187.5 Hz (k = 0..128,
  DC-24 kHz), on HEALPix nside 16 (RING order). Right ears are **not** mirrored.
- Spectrum = DTFT of the complete impulse response at those frequencies; nothing truncated or windowed.
- STPS (order 2, lambda 1e-5), one interpolation matrix per distinct direction grid; every subject is
  interpolated from its own grid.
- HUTUBS was upsampled from 44.1 kHz (resample_poly 160/147, Kaiser beta 10, gain 44100/48000); all
  other databases are native 48 kHz.
- **Use only up to 15.94 kHz (bins 0-85).** Above 16 kHz: CAR Lab has a 16.3 kHz filter, the SYMARE BEM is
  valid to ~16 kHz, and HUTUBS has resampling images between 22.05 and 24 kHz.
- SONICOM directions below -45 deg elevation are extrapolated by the STPS, not measured.

## Changes from the data as delivered

{chr(10).join(notes)}

## Checks passed

- Every input file's sampling rate, coordinate type and units, receiver order and listener view asserted.
- Per subject: left ear louder (and, where the ITD is kept, earlier) at azimuth +45, +90, +135 deg, and
  the reverse at -45, -90, -135 deg.
- The same code at 44.1 kHz reproduces the existing 44.1 kHz arrays (max |diff| {rep['regression_44k1']['sonicom_P0001']:.1e} dB SONICOM P0001,
  {rep['regression_44k1']['hutubs_pp1']:.1e} dB HUTUBS pp1).
- {carlab['identical']} of 218 CAR Lab subjects are identical to the earlier 48 kHz export; the other five
  were changed on purpose: {', '.join(f"{k} ({v['reason']})" for k, v in carlab['changed'].items())}.
- HUTUBS upsampling: worst subject rms {max(rep['hutubs_resampling']['resample_L_rms_db'], rep['hutubs_resampling']['resample_R_rms_db']):.4f} dB,
  99.9th percentile {max(rep['hutubs_resampling']['resample_L_p999_db'], rep['hutubs_resampling']['resample_R_p999_db']):.4f} dB, up to 16 kHz.
- Every array's SHA-256 matches its manifest; shapes, dtype and finiteness checked; no unlisted files.
"""
    open(os.path.join(out, 'README_DATA.md'), 'w').write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=P.DEFAULT_OUT)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    vr = os.path.join(out, 'verification_report.json')
    mans = glob.glob(os.path.join(out, '*_48k', 'manifest.json'))
    if not os.path.exists(vr) or any(os.path.getmtime(m) > os.path.getmtime(vr) for m in mans):
        raise SystemExit('Run verify_48k.py first (verification_report.json missing or older than a manifest).')
    readme(out)
    n_expected = sum(2 * len(json.load(open(m))) for m in mans)
    zpath = f'{out}_{datetime.date.today().isoformat()}.zip'
    root = os.path.basename(out)
    files = sorted(f for f in glob.glob(os.path.join(out, '**', '*'), recursive=True) if os.path.isfile(f))
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in files:
            z.write(f, os.path.join(root, os.path.relpath(f, out)))
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        names = z.namelist()
    n_npy = sum(n.endswith('.npy') and '/npy_' in n for n in names)
    if bad is not None or n_npy != n_expected:
        raise SystemExit(f'zip check failed: bad member {bad}, {n_npy} arrays vs {n_expected} expected')
    digest = P.sha256(zpath)
    open(zpath + '.sha256', 'w').write(f'{digest}  {os.path.basename(zpath)}\n')
    print(f'{zpath}\n  {len(names)} files ({n_npy} arrays), {os.path.getsize(zpath) / 2**30:.2f} GiB, CRC OK\n  SHA-256 {digest}')


if __name__ == '__main__':
    main()
