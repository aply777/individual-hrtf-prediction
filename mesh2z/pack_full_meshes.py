"""Pack SONICOM's complete preprocessed head meshes, byte for byte, for upload to Google Drive.

FFM_3_2_1 ... FFM_3_5 read `mesh2z_samples` (20 000 area-uniform points per ear crop, drawn
from these meshes). FFM_3_6 reads the meshes themselves. This script packs them:

    sonicom_full_meshes.zip
        sonicom_full_meshes/PXXXX_preprocessed.stl   original files, unmodified (deflate level 1)
        sonicom_full_meshes/manifest_full.csv        one row per subject, see FIELDS below
        sonicom_full_meshes/PACK_INFO.json
    sonicom_full_meshes.zip.sha256                   beside the zip; FFM_3_6 checks the zip against it

Before writing, for every subject:
  - the 200 subjects are exactly FFM_3_1's 164 development + 36 locked subjects;
  - the STL is a well-formed binary STL (header triangle count matches the body);
  - its SHA-1 equals `stl_sha1` in mesh2z_samples/PXXXX.npz, i.e. the same source file that the
    earlier notebooks were sampled from;
  - the ear-canal positions read from the Mesh2HRTF SOFA equal those in mesh2z_samples/manifest.csv.
After writing, every STL is read back from the zip and its SHA-256 compared with the manifest.

The zip is deterministic (fixed entry order, timestamps and compression level): packing the same
files again gives the same bytes and therefore the same SHA-256.

    python pack_full_meshes.py
    python pack_full_meshes.py --out /some/where/sonicom_full_meshes.zip
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile

import h5py
import numpy as np

SON = os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM')
DEFAULTS = dict(
    root=os.environ.get('SONICOM_SYNTH_ROOT', f'{SON}/synthetic'),
    samples=f'{SON}/mesh2z_samples',
    latents=f'{SON}/mesh2z_sonicom_latents_v1.npz',
    out=f'{SON}/sonicom_full_meshes.zip',
)
TOP = 'sonicom_full_meshes'
EAR_R = 45.0                     # mm; the crop radius of sample_meshes.py, recorded for FFM_3_6
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FIELDS = ['subject', 'partition', 'stl_file', 'stl_bytes', 'stl_sha256', 'stl_sha1', 'n_tris',
          'ear_left_x_mm', 'ear_left_y_mm', 'ear_left_z_mm',
          'ear_right_x_mm', 'ear_right_y_mm', 'ear_right_z_mm',
          'sofa_file', 'sofa_sha256', 'crop_area_L_mm2_samples', 'crop_area_R_mm2_samples']


def stl_triangles(b):
    n = int(np.frombuffer(b, '<u4', count=1, offset=80)[0])
    if len(b) != 84 + 50 * n:
        raise ValueError(f'not a binary STL: header says {n} triangles, file has {len(b)} bytes')
    return n


def ear_positions_mm(sofa_path):
    """Same reading as sample_meshes.py: ReceiverPosition in metres -> mm; row 0 = left."""
    with h5py.File(sofa_path, 'r') as f:
        rp = np.array(f['ReceiverPosition'], dtype=np.float64).reshape(-1, 3)
        unit = f['ReceiverPosition'].attrs.get('Units', b'')
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    if rp.shape != (2, 3) or unit.lower() not in ('meter', 'metre', 'm'):
        raise ValueError(f'ReceiverPosition {rp.shape} in {unit!r}')
    return rp[0] * 1000.0, rp[1] * 1000.0


def entry(name):
    zi = zipfile.ZipInfo(f'{TOP}/{name}', date_time=FIXED_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    zi.create_system = 3
    return zi


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', default=v)
    args = ap.parse_args()

    L = np.load(args.latents)
    dev = [str(s)[4:] for s in L['eval_sids']]
    locked = [str(s)[4:] for s in L['locked_sids']]
    part = {**{s: 'dev' for s in dev}, **{s: 'locked' for s in locked}}
    with open(os.path.join(args.samples, 'manifest.csv'), newline='') as f:
        old = {r['subject']: r for r in csv.DictReader(f)}
    subjects = sorted(d for d in os.listdir(args.root)
                      if os.path.exists(os.path.join(args.root, d, 'SYNTHETIC_HRTF', f'{d}_preprocessed.stl')))
    if set(subjects) != set(part) or len(subjects) != 200:
        sys.exit(f'{len(subjects)} meshes on disk; expected the 164 + 36 subjects of FFM_3_1 '
                 f'(missing {sorted(set(part) - set(subjects))[:5]}, extra {sorted(set(subjects) - set(part))[:5]})')

    tmp = args.out + '.part'
    rows, t0, total = [], time.time(), 0
    with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
        for i, s in enumerate(subjects, 1):
            d = os.path.join(args.root, s, 'SYNTHETIC_HRTF')
            stl, sofa = os.path.join(d, f'{s}_preprocessed.stl'), os.path.join(d, 'HRIR_SONICOM_44100.sofa')
            b = open(stl, 'rb').read()
            n = stl_triangles(b)
            sha1 = hashlib.sha1(b).hexdigest()
            ref = str(np.load(os.path.join(args.samples, f'{s}.npz'))['stl_sha1'])
            if sha1 != ref:
                sys.exit(f'{s}: STL SHA-1 {sha1} differs from mesh2z_samples ({ref}); not the sampled source')
            eL, eR = ear_positions_mm(sofa)
            o = old[s]
            got = np.r_[eL, eR]
            want = np.array([float(o[k]) for k in ('earL_x', 'earL_y', 'earL_z', 'earR_x', 'earR_y', 'earR_z')])
            if not np.allclose(got, want, rtol=0, atol=1e-9):
                sys.exit(f'{s}: ear canals {got} differ from mesh2z_samples/manifest.csv {want}')
            z.writestr(entry(f'{s}_preprocessed.stl'), b, compresslevel=1)
            rows.append(dict(subject=s, partition=part[s], stl_file=f'{s}_preprocessed.stl', stl_bytes=len(b),
                             stl_sha256=hashlib.sha256(b).hexdigest(), stl_sha1=sha1, n_tris=n,
                             **{f'ear_{side}_{ax}_mm': repr(float(v))
                                for side, e in (('left', eL), ('right', eR)) for ax, v in zip('xyz', e)},
                             sofa_file=f'{s}/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa', sofa_sha256=sha256_file(sofa),
                             crop_area_L_mm2_samples=o['crop_area_L_mm2'], crop_area_R_mm2_samples=o['crop_area_R_mm2']))
            total += len(b)
            if i == 1 or i % 25 == 0 or i == len(subjects):
                print(f'[{i:3d}/{len(subjects)}] {s} {n:,} triangles | {total / 1e9:.2f} GB read | {time.time() - t0:.0f} s',
                      flush=True)

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
        z.writestr(entry('manifest_full.csv'), buf.getvalue().encode(), compresslevel=1)
        info = dict(
            what='SONICOM PXXXX_preprocessed.stl, unmodified bytes, for mesh2z FFM_3_6',
            source='SONICOM synthetic/PXXXX/SYNTHETIC_HRTF/PXXXX_preprocessed.stl (Frankfurt-aligned, '
                   'beheaded, smoothed, hair removed by the SONICOM curators; not the raw 3DSCAN)',
            n_subjects=len(rows), n_dev=sum(r['partition'] == 'dev' for r in rows),
            n_locked=sum(r['partition'] == 'locked' for r in rows),
            stl_bytes_total=total, n_tris_total=int(sum(r['n_tris'] for r in rows)),
            units='mm', frame='origin interaural midpoint; +x forward, +y left, +z up',
            ear_canals='Mesh2HRTF ReceiverPosition from HRIR_SONICOM_44100.sofa, metres x 1000, right NOT mirrored',
            crop_definition_in_sample_meshes=f'faces whose centroid lies within {EAR_R} mm of the canal',
            checked='STL SHA-1 == mesh2z_samples stl_sha1; canals == mesh2z_samples/manifest.csv',
            packer='mesh2z/pack_full_meshes.py')
        z.writestr(entry('PACK_INFO.json'), json.dumps(info, indent=1).encode(), compresslevel=1)

    # read every STL back from the zip and compare with the manifest
    with zipfile.ZipFile(tmp) as z:
        for r in rows:
            if hashlib.sha256(z.read(f"{TOP}/{r['stl_file']}")).hexdigest() != r['stl_sha256']:
                sys.exit(f"{r['subject']}: SHA-256 read back from the zip differs")
    os.replace(tmp, args.out)
    digest = sha256_file(args.out)
    with open(args.out + '.sha256', 'w') as f:
        f.write(f'{digest}  {os.path.basename(args.out)}\n')
    print(f'\n{len(rows)} meshes ({sum(r["partition"] == "dev" for r in rows)} dev + '
          f'{sum(r["partition"] == "locked" for r in rows)} locked), {total / 1e9:.2f} GB of STL '
          f'-> {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB, {time.time() - t0:.0f} s)')
    print(f'every STL read back from the zip matches its SHA-256\nzip SHA-256 {digest}')


if __name__ == '__main__':
    main()
