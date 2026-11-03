"""Step 1 of mesh2z: SONICOM head meshes -> fixed-size, area-uniform point samples.

Reads, per subject, `PXXXX/SYNTHETIC_HRTF/PXXXX_preprocessed.stl` (the scan with minimal
modification) and the ear-canal positions stored in the Mesh2HRTF simulation
`HRIR_SONICOM_44100.sofa` beside it, and writes one `PXXXX.npz` holding

    head_pts, head_nrm   (N_HEAD, 3)  area-uniform sample of the whole mesh
    earL_pts, earL_nrm   (N_EAR, 3)   area-uniform sample of the faces within EAR_R of the
                                      left ear canal, raw coordinates
    earR_pts, earR_nrm   (N_EAR, 3)   same for the right ear, ALREADY MIRRORED (y -> -y) into
                                      the left-ear frame, matching the right-ear mirroring
                                      of the HRTFs in MonauralHRTFDataset
    ear_left_mm, ear_right_mm         ear-canal positions from the SOFA (right NOT mirrored)

Every sample array is stored in a random order, so its first k rows are themselves an
area-uniform sample of size k. Subsample by slicing; do not resample.

Frame, verified on P0001-P0003 and asserted for every subject: millimetres, origin at the
interaural midpoint, +x forward (nose), +y left, +z up. Scale is NOT normalised: head and
ear size are signal.

Mesh resolution differs between subjects (P0001 375k triangles, P0003 778k), which is why
everything here is sampled by area and nothing uses vertices directly.

Every check fails loudly. A subject that fails any check is listed in `failures.json` and
the script exits non-zero; nothing is silently skipped.

    python sample_meshes.py                      # all subjects found
    python sample_meshes.py --subjects P0001 P0002
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_ROOT = os.environ.get(
    'SONICOM_SYNTH_ROOT', os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM/synthetic'))
MEASURED_ROOT = os.environ.get(
    'SONICOM_MEASURED_ROOT', os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM/dataset'))

N_HEAD = 50_000
N_EAR = 20_000
EAR_R = 45.0            # mm; crop radius around the ear canal, pinna plus surrounding skin
BASE_SEED = 5021


# ---------------------------------------------------------------- reading
def read_binary_stl(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), '<u4')[0])
        body = f.read()
    if len(body) != 50 * n:
        raise ValueError(f'not a binary STL: header says {n} triangles, body has {len(body)} bytes')
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(body, dt, count=n)['v']          # (T, 3, 3) float32


def ear_positions_mm(sofa_path):
    """Ear-canal positions written by Mesh2HRTF, converted to mm. Row 0 = left receiver."""
    with h5py.File(sofa_path, 'r') as f:
        rp = np.array(f['ReceiverPosition'], dtype=np.float64).reshape(-1, 3)
        unit = f['ReceiverPosition'].attrs.get('Units', b'')
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    if rp.shape != (2, 3):
        raise ValueError(f'ReceiverPosition shape {rp.shape}, expected (2, 3)')
    if unit.lower() not in ('meter', 'metre', 'm'):
        raise ValueError(f'ReceiverPosition units {unit!r}, expected metres')
    return rp[0] * 1000.0, rp[1] * 1000.0


def energy_direction(sofa_path):
    """Broadband-energy-weighted mean source direction for each receiver, (2, 3) unit-ish."""
    with h5py.File(sofa_path, 'r') as f:
        ir = np.array(f['Data.IR'], dtype=np.float64)            # (M, 2, N)
        pos = np.array(f['SourcePosition'], dtype=np.float64)
        ptype = f['SourcePosition'].attrs.get('Type', b'spherical')
        ptype = ptype.decode() if isinstance(ptype, bytes) else str(ptype)
    if 'spher' not in ptype.lower():
        raise ValueError(f'SourcePosition type {ptype!r}, expected spherical')
    az, el = np.deg2rad(pos[:, 0]), np.deg2rad(pos[:, 1])
    u = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    e = (ir ** 2).sum(-1)                                        # (M, 2)
    return (e.T @ u) / e.sum(0)[:, None]


# ---------------------------------------------------------------- geometry
def mesh_stats(V):
    """Closure, orientation and size of a triangle soup. V: (T, 3, 3) float32."""
    flat = np.ascontiguousarray(V.reshape(-1, 3))
    key = flat.view(np.dtype((np.void, flat.dtype.itemsize * 3))).ravel()
    _, inv = np.unique(key, return_inverse=True)                 # weld on exact float32 bits
    F = inv.reshape(-1, 3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    V64 = V.astype(np.float64)
    vol = np.einsum('ij,ij->i', V64[:, 0], np.cross(V64[:, 1], V64[:, 2])).sum() / 6.0
    edges = np.linalg.norm(np.concatenate([V64[:, 1] - V64[:, 0], V64[:, 2] - V64[:, 1],
                                           V64[:, 0] - V64[:, 2]]), axis=1)
    U = np.zeros((inv.max() + 1, 3))
    U[inv] = flat
    bad = uE[cnt != 2]                       # boundary (1 face) or non-manifold (>2 faces)
    st = dict(n_tris=len(V), n_verts=int(inv.max()) + 1, n_open_edges=len(bad),
              volume_mm3=float(vol), edge_median_mm=float(np.median(edges)))
    return st, (U[bad[:, 0]] + U[bad[:, 1]]) / 2, np.linalg.norm(U[bad[:, 0]] - U[bad[:, 1]], axis=1)


def face_area_normal(V):
    V = V.astype(np.float64)
    cr = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    a2 = np.linalg.norm(cr, axis=1)
    return 0.5 * a2, cr / np.maximum(a2, 1e-300)[:, None]


def sample_faces(V, area, normal, idx, n, rng):
    """Area-uniform sample of n points on the faces V[idx]; returns points, normals (float32)."""
    w = area[idx] / area[idx].sum()
    f = idx[rng.choice(len(idx), size=n, p=w)]
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.c_[1 - s, s * (1 - r2), s * r2]
    pts = np.einsum('nk,nkd->nd', bary, V[f].astype(np.float64))
    return pts.astype(np.float32), normal[f].astype(np.float32)


def mirror_y(a):
    out = np.array(a, copy=True)
    out[..., 1] *= -1
    return out


# ---------------------------------------------------------------- checks
def check(cond, msg, errors):
    if not cond:
        errors.append(msg)


def process(sid, root, out_dir, overwrite):
    d = os.path.join(root, sid, 'SYNTHETIC_HRTF')
    stl = os.path.join(d, f'{sid}_preprocessed.stl')
    sofa = os.path.join(d, 'HRIR_SONICOM_44100.sofa')
    out = os.path.join(out_dir, f'{sid}.npz')
    if os.path.exists(out) and not overwrite:
        return 'exists', None
    for p in (stl, sofa):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    V = read_binary_stl(stl)
    st, bad_mid, bad_len = mesh_stats(V)
    area, normal = face_area_normal(V)
    lo, hi = V.reshape(-1, 3).min(0), V.reshape(-1, 3).max(0)
    ext = hi - lo
    earL, earR = ear_positions_mm(sofa)
    edir = energy_direction(sofa)

    err = []
    # scale: millimetres, a human head truncated at the neck
    check(np.all((ext > 120) & (ext < 350)), f'bbox extent {np.round(ext, 1)} mm outside 120-350 mm', err)
    # 3.5-5.9 L on 166 subjects; P0023 is 10.4 L (larger head, neck cut 22 mm lower)
    check(2e6 < st['volume_mm3'] < 12e6, f"volume {st['volume_mm3']:.3g} mm^3 outside 2-12 L "
                                           '(negative => inward normals)', err)
    # The scans are NOT all watertight: 24/200 have open or non-manifold edges, almost all far
    # from the ears and at most 5.2 mm of edge inside the ear crop. Irrelevant to area
    # sampling; recorded, and fatal only where it could change the ear crop or the head.
    d_ear = np.minimum(np.linalg.norm(bad_mid - earL, axis=1), np.linalg.norm(bad_mid - earR, axis=1))
    st['open_len_mm'] = float(bad_len.sum())
    st['open_len_in_crop_mm'] = float(bad_len[d_ear < EAR_R].sum())
    check(st['open_len_in_crop_mm'] < 10, f"{st['open_len_in_crop_mm']:.1f} mm of open/non-manifold "
                                          f'edge within {EAR_R} mm of an ear canal', err)
    check(st['open_len_mm'] < 200, f"{st['open_len_mm']:.1f} mm of open/non-manifold edge in total", err)
    # frame: nose forward on the midline, crown up
    nose = V.reshape(-1, 3)[V.reshape(-1, 3)[:, 0].argmax()]
    check(nose[0] > 90 and abs(nose[1]) < 20, f'max-x point {np.round(nose, 1)} is not a midline nose', err)
    # nose tip height on all 200 subjects: z = -22.6 .. -3.2 mm. A head turned 180 degrees puts
    # the occiput here instead (z = +36 .. +72 on the subjects checked); the ear-crop check
    # below catches the rest.
    check(-40 < nose[2] < 10, f'max-x point at z={nose[2]:.1f} mm, a nose tip sits at -40..10', err)
    check(hi[2] > 100, f'max z {hi[2]:.1f} mm: crown not up', err)
    # ears: origin at the interaural midpoint, left on +y
    for nm, e, sgn in (('left', earL, 1), ('right', earR, -1)):
        check(50 < sgn * e[1] < 95 and abs(e[0]) < 15 and abs(e[2]) < 15,
              f'{nm} ear canal at {np.round(e, 1)} mm, expected (|x|<15, {sgn:+d}*y in 50-95, |z|<15)', err)
    # 166 subjects up to 9.1 mm, three more at 10.0-10.7: a continuum, not outliers
    check(abs(earL[1] + earR[1]) < 15, f'ears asymmetric about origin by {earL[1] + earR[1]:.1f} mm', err)
    # laterality, acoustically: receiver 0 must hear the +y hemisphere louder
    check(edir[0, 1] > 0.2 and edir[1, 1] < -0.2,
          f'energy-weighted mean direction y: receiver0 {edir[0, 1]:+.3f}, receiver1 {edir[1, 1]:+.3f}', err)

    rng = np.random.default_rng(BASE_SEED + int(sid[1:]))
    allf = np.arange(len(V))
    head_pts, head_nrm = sample_faces(V, area, normal, allf, N_HEAD, rng)
    cen = V.astype(np.float64).mean(1)
    ears = {}
    for nm, e in (('L', earL), ('R', earR)):
        idx = np.flatnonzero(np.linalg.norm(cen - e, axis=1) < EAR_R)
        check(len(idx) > 1000, f'only {len(idx)} faces within {EAR_R} mm of ear {nm}', err)
        if len(idx) == 0:
            break
        p, n = sample_faces(V, area, normal, idx, N_EAR, rng)
        ears[nm] = (p, n, float(area[idx].sum()))
        # the pinna sits behind the canal: crop centroid 4.5-6.4 mm behind it on P0001-P0003
        dx = p[:, 0].mean() - e[0]
        check(dx < 0, f'ear {nm} crop centroid {dx:+.1f} mm in x from the canal; expected behind it', err)
    # the SOFA ear positions must sit on this mesh, not merely in the same box
    tree = cKDTree(head_pts)
    for nm, e in (('left', earL), ('right', earR)):
        dist = tree.query(e)[0]
        check(dist < 6.0, f'{nm} ear canal is {dist:.1f} mm from the nearest surface sample', err)
    if err:
        return 'failed', err

    np.savez_compressed(
        out,
        head_pts=head_pts, head_nrm=head_nrm,
        earL_pts=ears['L'][0], earL_nrm=ears['L'][1],
        earR_pts=mirror_y(ears['R'][0]), earR_nrm=mirror_y(ears['R'][1]),
        ear_left_mm=earL.astype(np.float32), ear_right_mm=earR.astype(np.float32),
        units='mm', frame='origin interaural midpoint; +x forward, +y left, +z up',
        earR_note='earR_* mirrored y->-y into the left-ear frame; ear_right_mm is NOT mirrored',
        ear_radius_mm=EAR_R, seed=BASE_SEED + int(sid[1:]),
        stl_sha1=hashlib.sha1(open(stl, 'rb').read()).hexdigest(),
    )
    row = dict(subject=sid, **st, area_mm2=float(area.sum()),
               ext_x=float(ext[0]), ext_y=float(ext[1]), ext_z=float(ext[2]),
               z_min=float(lo[2]), nose_z=float(nose[2]),
               earL_x=earL[0], earL_y=earL[1], earL_z=earL[2],
               earR_x=earR[0], earR_y=earR[1], earR_z=earR[2],
               crop_area_L_mm2=ears['L'][2], crop_area_R_mm2=ears['R'][2],
               edir_y_L=float(edir[0, 1]), edir_y_R=float(edir[1, 1]))
    return 'ok', row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--out', default=None, help='default: <root>/../mesh2z_samples')
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out or os.path.join(root, '..', 'mesh2z_samples'))
    os.makedirs(out_dir, exist_ok=True)
    subjects = args.subjects or sorted(
        os.path.basename(os.path.dirname(os.path.dirname(p)))
        for p in glob.glob(os.path.join(root, 'P*', 'SYNTHETIC_HRTF', '*_preprocessed.stl')))
    if not subjects:
        raise SystemExit(f'no P*/SYNTHETIC_HRTF/*_preprocessed.stl under {root}')
    print(f'{len(subjects)} subjects | root {root}\n-> {out_dir}')

    rows, failures, skipped = [], {}, 0
    for i, sid in enumerate(subjects, 1):
        try:
            status, info = process(sid, root, out_dir, args.overwrite)
        except Exception as e:                                   # report, never swallow
            status, info = 'failed', [f'{type(e).__name__}: {e}']
        if status == 'ok':
            rows.append(info)
            print(f"[{i:3d}/{len(subjects)}] {sid} ok  tris {info['n_tris']:,}  "
                  f"earL y {info['earL_y']:+.1f}  earR y {info['earR_y']:+.1f}")
        elif status == 'exists':
            skipped += 1
        else:
            failures[sid] = info
            print(f'[{i:3d}/{len(subjects)}] {sid} FAILED')
            for m in info:
                print(f'      - {m}')

    if rows:                                                     # merge by subject; reruns replace
        man = os.path.join(out_dir, 'manifest.csv')
        old = {}
        if os.path.exists(man):
            with open(man, newline='') as f:
                old = {r['subject']: r for r in csv.DictReader(f)}
        old.update({r['subject']: r for r in rows})
        with open(man, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(old[k] for k in sorted(old))
    with open(os.path.join(out_dir, 'failures.json'), 'w') as f:
        json.dump(failures, f, indent=2)

    print(f'\nok {len(rows)} | already done {skipped} | failed {len(failures)}')
    if failures:
        sys.exit(f'{len(failures)} subject(s) failed; see {out_dir}/failures.json')


if __name__ == '__main__':
    main()
