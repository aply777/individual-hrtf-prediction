#!/usr/bin/env python3
"""48 kHz preprocessing: every database -> HEALPix dB-magnitude arrays at 48 kHz.

Written 2026-10-10. This is the **48 kHz version** of the data pipeline. It replaces, for new
work, `analysis/preprocess_multidataset_stps.py` (44.1 kHz) and `analysis/export_carlab_npy.py`
/ `export_carlab_44k1.py`; it does not modify or overwrite their outputs.

Conventions, fixed before the run (see README.md for the reasons):

* **48 kHz everywhere.** SONICOM (measured and simulated), SYMARE (acoustic and BEM) and CAR Lab
  are native 48 kHz and are not resampled. HUTUBS (44.1 kHz) is upsampled in the time domain with
  `resample_poly(160, 147)` (Kaiser beta 10) and scaled by 44100/48000, so the filter's frequency response is
  unchanged (checked per subject against the DTFT of the original).
* **Spectrum** = DTFT of the complete impulse response at f_k = k * 48000/256, k = 0..128
  (187.5 Hz apart). No impulse response is truncated or windowed. For a 256-sample IR this is
  exactly `rfft(h, n=256)`, the rule of the 44.1 kHz pipeline.
* **dB** = 20 log10(|H| + 1e-6), then a spherical thin-plate spline (order 2, lambda 1e-5) onto
  HEALPix nside 16, i.e. the 44.1 kHz pipeline's settings.
* **Every subject is interpolated from its own direction grid.** The STPS matrix is cached under a
  hash of the subject's complete grid, never under its size or first direction. (The CAR Lab
  exporters keyed the cache on size + first direction; two CAR Lab subjects, CL154 and CL159,
  were interpolated with CL126's grid. See README.md.)
* Coordinates, sampling rate, receiver order and listener orientation are read from every file and
  asserted; a per-subject acoustic check requires the left ear to be louder (and, where the ITD is
  kept, earlier) for a source at azimuth +90 deg. Any failure stops the run.
* Ear channels exchanged in the delivered data are swapped back and flagged in the manifest: CAR Lab
  CL003 (lab remark 'left and right mics reversed') and CL088, SONICOM P0339. SONICOM P0209 is
  excluded: its left/right cues are inconsistent rather than swapped.
* Directions recorded twice (CAR Lab CL169: azimuth +180 and -180 deg at six elevations) are merged by
  averaging their dB magnitudes; the difference between the copies is recorded.
* Left and right ears are saved in their native frames (no mirroring), as in the 44.1 kHz pipeline.

Added 2026-10-11: `SYMARE_HTE`, the SYMARE BEM on head, torso and ears (`Bem/HTE05kHz`), for the
Task 1 torso check. Valid to ~5 kHz only (it rolls off above 5.3 kHz), so its left/right check uses
the ILD over 1.5-5 kHz. It is written only when asked for (`--only SYMARE_HTE`), to its own folder, and
is not part of `data_48k_2026-10-10.zip`; running with no `--only` still writes the six original sets.

Output: one (129, 3072) float32 array per ear, `<id>_left_full.npy` / `<id>_right_full.npy`, in
`<out>/<DATASET>_48k/npy_left|npy_right`, with `meta.json` and `manifest.json` (SHA-256 of every
input and output, grid hash, ILD, ITD) in each dataset folder.

    python preprocess_48k.py                       # all datasets
    python preprocess_48k.py --only HUTUBS SYMARE  # a subset
    python preprocess_48k.py --overwrite           # replace existing outputs
    python preprocess_48k.py --only SYMARE_HTE --out ~/Documents/26S2/ELEC5021/data_48k_hte
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import h5py
import healpy as hp
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'analysis'))
from spherical_spline import build_interpolation_matrix  # noqa: E402  (the one STPS implementation)
from _paths import symare_root                            # noqa: E402

# ----------------------------------------------------------------------------- conventions
FS = 48000
NFFT_REF = 256                       # sets the frequency grid: f_k = k * FS / NFFT_REF
NBINS = NFFT_REF // 2 + 1            # 129, DC .. 24 kHz
FREQS = np.arange(NBINS) * FS / NFFT_REF
NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
DB_FLOOR = 1e-6
SPLINE_ORDER, SMOOTHING = 2, 1e-5
HUTUBS_UP, HUTUBS_DOWN = 160, 147    # 44100 * 160 / 147 = 48000
RESAMPLE_WINDOW = ('kaiser', 10.0)   # scipy's default beta 5 leaves a 0.006 dB mean / 0.018 dB rms passband error
ILD_BAND = (FREQS >= 1500) & (FREQS <= 16000)
ITD_RANGE_US = (300.0, 1100.0)       # plausible |ITD| at azimuth +-90 deg

# ----------------------------------------------------------------------------- locations
DATA_A = os.path.expanduser(os.environ.get('ELEC5020_DATA', '~/Documents/26S1/ELEC5020'))
SONICOM = os.path.join(DATA_A, 'SONICOM')
HUTUBS = os.path.join(DATA_A, 'HUTUBS', 'HRIRs')
SYMARE = os.path.join(DATA_A, 'SYMARE', 'HRIRs')
CARLAB_ROOT = symare_root()          # <ELEC5021>/SYMARE, holds the CAR Lab archive
DEFAULT_OUT = os.path.expanduser('~/Documents/26S2/ELEC5021/data_48k')


# ============================================================================= helpers
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def sph_deg_to_xyz(az_deg, el_deg):
    """SOFA convention: azimuth counter-clockwise from +x (front) towards +y (left); elevation up."""
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    return np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))


HP_XYZ = np.column_stack(hp.pix2vec(NSIDE, np.arange(NPIX)))


class GridCache:
    """STPS matrices keyed by a hash of the complete direction grid of each subject."""

    def __init__(self):
        self.W, self.count = {}, {}

    def get(self, xyz):
        key = hashlib.sha1(np.ascontiguousarray(np.round(xyz, 9)).tobytes()).hexdigest()[:16]
        if key not in self.W:
            self.W[key] = build_interpolation_matrix(xyz, HP_XYZ, spline_order=SPLINE_ORDER,
                                                     smoothing=SMOOTHING)
            self.count[key] = 0
            print(f'    [STPS] new grid {key}: {len(xyz)} directions', flush=True)
        self.count[key] += 1
        return key, self.W[key]


def spectrum_db(h, fs=FS, floor=DB_FLOOR):
    """h: (n_dir, L) impulse responses at `fs` -> (n_dir, 129) dB magnitude at k * fs / 256.

    The complete IR is transformed with a length that is a multiple of 256 and the result
    decimated, which samples its DTFT at exactly the 129 frequencies without truncating it.
    """
    L = h.shape[1]
    npad = NFFT_REF * int(np.ceil(L / NFFT_REF))
    H = np.fft.rfft(h, n=npad, axis=1)[:, ::npad // NFFT_REF]
    assert H.shape[1] == NBINS, H.shape
    return 20.0 * np.log10(np.abs(H) + floor)


def to_healpix(mag_db, W):
    """(n_dir, 129) dB -> (129, 3072) float32, as in the 44.1 kHz pipeline."""
    out = (W @ mag_db).T.astype(np.float32)
    assert out.shape == (NBINS, NPIX) and np.isfinite(out).all()
    return out


def merge_repeated(xyz, hL, hR, magL, magR):
    """Directions recorded more than once (e.g. azimuth +180 and -180 deg) are merged: their dB
    magnitudes are averaged and the first copy's impulse response is kept for the ITD check. Without
    this, duplicate points enter the STPS system and are averaged implicitly through the smoothing."""
    _, first, inv, cnt = np.unique(np.round(xyz, 9), axis=0, return_index=True,
                                   return_inverse=True, return_counts=True)
    if (cnt == 1).all():
        return xyz, hL, hR, magL, magR, {}
    inv = inv.ravel()
    keep = np.sort(first)
    groups = [np.where(inv == inv[k])[0] for k in keep]
    band = (FREQS >= 200) & (FREQS <= 16000)
    info = {'merged_repeated_directions': int((cnt > 1).sum()), 'n_dir_recorded': int(len(xyz))}
    for ear, m in (('L', magL), ('R', magR)):
        d = np.concatenate([m[g[0], band] - m[j, band] for g in groups if len(g) > 1 for j in g[1:]])
        info[f'repeat_rms_db_{ear}'] = round(float(np.sqrt((d ** 2).mean())), 3)
    mL = np.stack([magL[g].mean(0) for g in groups])
    mR = np.stack([magR[g].mean(0) for g in groups])
    return xyz[keep], hL[keep], hR[keep], mL, mR, info


def nearest(xyz, az_deg, el_deg):
    return int(np.argmax(xyz @ sph_deg_to_xyz([az_deg], [el_deg])[0]))


LAT_AZ = (45.0, 90.0, 135.0, -45.0, -90.0, -135.0)   # horizontal plane; + = left in the SOFA convention


def laterality(hL, hR, magL, magR, xyz, fs, itd_present, ild_band=ILD_BAND):
    """ILD (dB, 1.5-16 kHz) and ITD (us) at six horizontal azimuths; assert the left/right sense.

    Every one of the six must have the expected sign (left louder, and earlier, for a source on the
    left); |ILD| > 3 dB at +-90 deg and > 1 dB at +-45 and +-135 deg; and, where the ITD is kept, a
    plausible ITD (300-1100 us at +-90 deg, 100-1100 us elsewhere). A channel swap or a mirrored
    azimuth fails all six; a corrupted measurement typically fails some. (The 1 dB bound off the
    interaural axis was set after a first scan with 3 dB everywhere flagged SONICOM P0220 and HUTUBS
    pp33 for a correct-sign but weak ILD at -135 deg (-2.0, -1.8 dB); both are normal at the other
    five azimuths, ITDs included.)
    """
    out, errs = {}, []
    for az in LAT_AZ:
        tag = f"{'p' if az > 0 else 'm'}{int(abs(az))}"
        i = nearest(xyz, az, 0.0)
        ild = float(magL[i, ild_band].mean() - magR[i, ild_band].mean())
        out[f'ild_{tag}_db'] = round(ild, 2)
        if not (np.sign(ild) == np.sign(az) and abs(ild) > (3.0 if abs(az) == 90 else 1.0)):
            errs.append(f'ILD {ild:+.1f} dB at az {az:+.0f}')
        if itd_present:
            c = np.correlate(hR[i], hL[i], mode='full')         # > 0 lag: right ear later
            itd = (int(np.argmax(np.abs(c))) - (hL.shape[1] - 1)) / fs * 1e6
            out[f'itd_{tag}_us'] = round(itd, 1)
            lo = ITD_RANGE_US[0] if abs(az) == 90 else 100.0
            if not (np.sign(itd) == np.sign(az) and lo <= abs(itd) <= ITD_RANGE_US[1]):
                errs.append(f'ITD {itd:+.0f} us at az {az:+.0f}')
    assert not errs, f'left/right sense: {errs}'
    return out


def _attr(obj, key):
    v = obj.attrs.get(key, b'')
    return v.decode() if isinstance(v, bytes) else str(v)


def read_sofa(path, expect_fs):
    """Read one SOFA file and assert everything the processing relies on."""
    with h5py.File(path, 'r') as f:
        ir = np.asarray(f['Data.IR'][()], dtype=np.float64)
        fs = float(np.asarray(f['Data.SamplingRate'][()]).ravel()[0])
        sp = np.asarray(f['SourcePosition'][()], dtype=np.float64)
        sp_type, sp_units = _attr(f['SourcePosition'], 'Type'), _attr(f['SourcePosition'], 'Units')
        rp = np.asarray(f['ReceiverPosition'][()], dtype=np.float64)
        lv = np.asarray(f['ListenerView'][()], dtype=np.float64).ravel()[:3]
        lv_type = _attr(f['ListenerView'], 'Type')
    assert fs == expect_fs, f'{path}: fs {fs} != {expect_fs}'
    assert ir.ndim == 3 and ir.shape[1] == 2 and ir.shape[0] == sp.shape[0], f'{path}: IR {ir.shape}'
    assert np.isfinite(ir).all(), f'{path}: non-finite IR'
    assert sp_type.lower() == 'spherical', f'{path}: SourcePosition type {sp_type}'
    assert sp_units.lower().replace(' ', '').startswith('degree,degree'), f'{path}: units {sp_units}'
    rp = rp.reshape(rp.shape[0], 3, -1)[:, :, 0]
    assert rp.shape[0] == 2 and rp[0, 1] > 0 > rp[1, 1], f'{path}: receiver 0 must be left (+y): {rp}'
    assert lv_type.lower() == 'cartesian' and np.allclose(lv, [1, 0, 0], atol=1e-9), f'{path}: ListenerView {lv}'
    return ir[:, 0, :], ir[:, 1, :], sp[:, 0], sp[:, 1], fs


# ============================================================================= datasets
def items_sonicom(kind):
    if kind == 'measured':
        pat = os.path.join(SONICOM, 'dataset', 'P*', 'HRTF', 'HRTF', '48kHz', 'P*_FreeFieldCompMinPhase_48kHz.sofa')
    else:
        pat = os.path.join(SONICOM, 'synthetic', 'P*', 'SYNTHETIC_HRTF', 'HRIR_SONICOM_48000.sofa')
    for p in sorted(glob.glob(pat)):
        pid = p.split(os.sep + 'dataset' + os.sep)[-1].split(os.sep)[0] if kind == 'measured' \
            else p.split(os.sep + 'synthetic' + os.sep)[-1].split(os.sep)[0]
        assert pid.startswith('P') and len(pid) == 5, p
        yield f'SON_{pid}', p


# Measured SONICOM subjects that fail the laterality check in the delivered files (44.1 and 48 kHz alike).
SONICOM_EAR_SWAP = {
    'SON_P0339': 'ILD and ITD reversed at all six azimuths in the measured files (44.1 and 48 kHz); no simulation exists',
}
SONICOM_EXCLUDE = {
    'SON_P0209': 'left/right inconsistent, not a channel swap: ILD reversed but weak (-11/+7 dB at +-90 deg), '
                 'ITD erratic (+396, -375, +1458 us at +90, +135, -45 deg in FreeFieldComp_48kHz)',
}


def load_sonicom(sid, path):
    hL, hR, az, el, fs = read_sofa(path, expect_fs=48000.0)
    extra = {}
    if sid in SONICOM_EAR_SWAP and '/dataset/' in path:
        hL, hR = hR, hL
        extra['ears_swapped'] = SONICOM_EAR_SWAP[sid]
    return hL, hR, sph_deg_to_xyz(az, el), fs, extra


def items_hutubs():
    for i in range(1, 97):
        p = os.path.join(HUTUBS, f'pp{i}_HRIRs_measured.sofa')
        assert os.path.exists(p), p
        yield f'HU_pp{i}', p


def dtft(h, f, fs):
    """Direct DTFT of h (n_dir, L) at frequencies f (Hz)."""
    n = np.arange(h.shape[1])
    return h @ np.exp(-2j * np.pi * np.outer(n, f) / fs)


def load_hutubs(sid, path):
    hL0, hR0, az, el, fs0 = read_sofa(path, expect_fs=44100.0)
    gain = 44100.0 / 48000.0                     # keeps the filter's frequency response
    hL = resample_poly(hL0, HUTUBS_UP, HUTUBS_DOWN, axis=1, window=RESAMPLE_WINDOW) * gain
    hR = resample_poly(hR0, HUTUBS_UP, HUTUBS_DOWN, axis=1, window=RESAMPLE_WINDOW) * gain
    # check: same physical frequencies, before and after, up to 16 kHz
    f = FREQS[FREQS <= 16000]
    check = {}
    for ear, a, b in (('L', hL0, hL), ('R', hR0, hR)):
        d = 20 * np.log10(np.abs(dtft(b, f, 48000.0)) + DB_FLOOR) - 20 * np.log10(np.abs(dtft(a, f, 44100.0)) + DB_FLOOR)
        check[f'resample_{ear}_mean_db'] = round(float(d.mean()), 5)
        check[f'resample_{ear}_rms_db'] = round(float(np.sqrt((d ** 2).mean())), 5)
        check[f'resample_{ear}_p999_db'] = round(float(np.percentile(np.abs(d), 99.9)), 5)
        assert abs(d.mean()) < 0.005 and np.sqrt((d ** 2).mean()) < 0.005 and np.percentile(np.abs(d), 99.9) < 0.05, \
            f'{path}: upsampling changed the response: {check}'
    check['len_in'], check['len_out'] = hL0.shape[1], hL.shape[1]
    return hL, hR, sph_deg_to_xyz(az, el), 48000.0, check


def items_symare(kind):
    sub = {'acoustic': 'Acoustic', 'bem': os.path.join('Bem', 'HE16kHz'), 'hte': os.path.join('Bem', 'HTE05kHz')}[kind]
    pre = 'HA' if kind == 'acoustic' else 'HB'
    for i in range(1, 62):
        p = os.path.join(SYMARE, sub, f'{pre}{i:02d}.mat')
        assert os.path.exists(p), p
        yield f'SYM_HA{i:02d}', p           # BEM files are paired to the acoustic subject id


_SYM = {}


def symare_grid():
    if not _SYM:
        P = os.path.join(SYMARE, 'Parameters')
        fs = float(np.squeeze(sio.loadmat(os.path.join(P, 'fs.mat'))['fs']))
        az = np.degrees(np.squeeze(sio.loadmat(os.path.join(P, 'azim.mat'))['azim']))
        el = np.degrees(np.squeeze(sio.loadmat(os.path.join(P, 'elev.mat'))['elev']))
        assert fs == 48000.0, f'SYMARE fs {fs}'
        assert az.shape == el.shape == (393,) and np.abs(el).max() <= 90.0
        _SYM.update(fs=fs, xyz=sph_deg_to_xyz(az, el))
    return _SYM


def load_symare(sid, path):
    g = symare_grid()
    m = sio.loadmat(path)
    hL, hR = np.asarray(m['hL'], float).T, np.asarray(m['hR'], float).T   # (393, 256)
    assert hL.shape == hR.shape == (393, 256) and np.isfinite(hL).all() and np.isfinite(hR).all(), path
    return hL, hR, g['xyz'], g['fs'], {}


CARLAB_REC = 2052


def items_carlab():
    man = json.load(open(os.path.join(CARLAB_ROOT, 'CARLab_subjects.json')))
    assert len(man) == 218 and len({r['id'] for r in man}) == 218
    for r in sorted(man, key=lambda r: r['id']):
        p = os.path.join(CARLAB_ROOT, 'New_hrtfs', 'CARLabHrirDataBaseFixedByPierre', f"hrirfilt{r['carlab_idx']}")
        assert os.path.exists(p), p
        yield r['id'], p


def read_hrirfilt(path):
    """CAR Lab binary: int16 L, then records of int16 az, int16 el, 2 x L float32 (48 kHz)."""
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), '<i2')[0])
    assert (len(b) - 2) % CARLAB_REC == 0 and 4 + 2 * 4 * L == CARLAB_REC, f'{path}: record size'
    body = b[2:].reshape(-1, CARLAB_REC)
    pos = np.frombuffer(body[:, :4].tobytes(), '<i2').reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), '<f4').reshape(-1, 2, L).astype(np.float64)
    assert np.isfinite(sig).all(), path
    return -pos[:, 0], pos[:, 1], sig          # azimuth sign flipped to the SOFA convention


# Subjects whose two ear channels are exchanged in the archive. Found by the per-subject laterality
# check (left ear quieter and later for a source at azimuth +90 deg, and the reverse at -90 deg, with
# front/back normal); 216 of 218 subjects pass it as delivered.
CARLAB_EAR_SWAP = {
    'CL003': "lab remark in carlabdb.mat: 'left and right mics reversed'",
    'CL088': 'no lab remark; ILD and ITD reversed at +-90 deg (-25/+18 dB), front/back normal',
}
_REMARKS = {}


def carlab_remark(idx):
    if not _REMARKS:
        d = sio.loadmat(os.path.join(CARLAB_ROOT, 'New_hrtfs', 'carlabdb.mat'))['carlab_hrir'][0]
        for i, e in enumerate(d, 1):
            v = np.array(e['endremark']).ravel()
            s = str(v[0]).strip() if v.size else ''
            _REMARKS[i] = '' if s.lower() in ('', 'none') else s
    return _REMARKS.get(idx, '')


def load_carlab(sid, path):
    az, el, sig = read_hrirfilt(path)
    idx = int(os.path.basename(path).replace('hrirfilt', ''))
    extra = {'carlab_idx': idx}
    if carlab_remark(idx):
        extra['lab_remark'] = carlab_remark(idx)
    left, right = 0, 1                                       # ear 0 = left, as delivered
    if sid in CARLAB_EAR_SWAP:
        left, right = 1, 0
        extra['ears_swapped'] = CARLAB_EAR_SWAP[sid]
    return sig[:, left, :], sig[:, right, :], sph_deg_to_xyz(az, el), 48000.0, extra


DATASETS = {
    'SONICOM': dict(items=lambda: items_sonicom('measured'), load=load_sonicom, itd=False, exclude=SONICOM_EXCLUDE,
                    source='SONICOM measured, P*_FreeFieldCompMinPhase_48kHz.sofa (minimum phase: no ITD)'),
    'SONICOM_SIM': dict(items=lambda: items_sonicom('synthetic'), load=load_sonicom, itd=True,
                        source='SONICOM Mesh2HRTF simulation, HRIR_SONICOM_48000.sofa (320 samples)'),
    'HUTUBS': dict(items=items_hutubs, load=load_hutubs, itd=True,
                   source='HUTUBS measured, pp*_HRIRs_measured.sofa (44.1 kHz), upsampled resample_poly 160/147 Kaiser beta 10, gain 44100/48000'),
    'SYMARE': dict(items=lambda: items_symare('acoustic'), load=load_symare, itd=True,
                   source='SYMARE HRIRs/Acoustic/HA*.mat (48 kHz)'),
    'SYMARE_BEM': dict(items=lambda: items_symare('bem'), load=load_symare, itd=True,
                       source='SYMARE HRIRs/Bem/HE16kHz/HB*.mat (FM-BEM, head and ears, valid to ~16 kHz); id = paired HA subject'),
    'CARLab': dict(items=items_carlab, load=load_carlab, itd=True,
                   source='CAR Lab hrirfilt (CARLabHrirDataBaseFixedByPierre, 48 kHz); ids from CARLab_subjects.json'),
    'SYMARE_HTE': dict(items=lambda: items_symare('hte'), load=load_symare, itd=True, ild_hz=(1500.0, 5000.0),
                       source='SYMARE HRIRs/Bem/HTE05kHz/HB*.mat (FM-BEM, head, torso and ears, valid to ~5 kHz; '
                              'rolls off above 5.3 kHz: use bins <= 26, i.e. <= 4.875 kHz); id = paired HA subject'),
}
ON_REQUEST = {'SYMARE_HTE'}                  # not written unless named in --only


# ============================================================================= main
def git_state():
    try:
        rev = subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain', '--', 'preprocess_48k'],
                                             text=True).strip())
        return {'commit': rev, 'preprocess_48k_uncommitted': dirty}
    except Exception as e:                   # noqa: BLE001
        return {'commit': None, 'error': str(e)}


def run_dataset(name, cfg, out_root, overwrite):
    d = os.path.join(out_root, f'{name}_48k')
    dl, dr = os.path.join(d, 'npy_left'), os.path.join(d, 'npy_right')
    if os.path.exists(d):
        if not overwrite:
            raise SystemExit(f'{d} exists; use --overwrite to replace it')
        shutil.rmtree(d)
    os.makedirs(dl), os.makedirs(dr)
    cache, manifest = GridCache(), []
    items = list(cfg['items']())
    assert items and len({i for i, _ in items}) == len(items), f'{name}: duplicate ids'
    excl = cfg.get('exclude', {})
    assert set(excl) <= {i for i, _ in items}, f'{name}: unknown excluded ids'
    items = [(i, p) for i, p in items if i not in excl]
    print(f'\n=== {name}: {len(items)} subjects ===', flush=True)
    for n, (sid, path) in enumerate(items, 1):
        hL, hR, xyz, fs, extra = cfg['load'](sid, path)
        assert fs == FS and hL.shape == hR.shape and hL.shape[0] == xyz.shape[0], f'{sid}: shapes {hL.shape} {xyz.shape}'
        assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0)
        magL, magR = spectrum_db(hL), spectrum_db(hR)
        xyz, hL, hR, magL, magR, rep = merge_repeated(xyz, hL, hR, magL, magR)
        extra = {**extra, **rep}
        assert len(np.unique(np.round(xyz, 9), axis=0)) == len(xyz), f'{sid}: repeated directions remain'
        key, W = cache.get(xyz)
        band = ILD_BAND if 'ild_hz' not in cfg else (FREQS >= cfg['ild_hz'][0]) & (FREQS <= cfg['ild_hz'][1])
        lat = laterality(hL, hR, magL, magR, xyz, fs, cfg['itd'], band)
        aL, aR = to_healpix(magL, W), to_healpix(magR, W)
        fL, fR = os.path.join(dl, f'{sid}_left_full.npy'), os.path.join(dr, f'{sid}_right_full.npy')
        np.save(fL, aL), np.save(fR, aR)
        manifest.append(dict(id=sid, source=os.path.relpath(path, os.path.expanduser('~')), source_sha256=sha256(path),
                             n_dir=int(len(xyz)), ir_len=int(hL.shape[1]), grid=key, **lat, **extra,
                             left=os.path.basename(fL), left_sha256=sha256(fL),
                             right=os.path.basename(fR), right_sha256=sha256(fR)))
        if n % 50 == 0 or n == len(items):
            print(f'  {n}/{len(items)}', flush=True)
    meta = dict(dataset=name, version='48k', created=datetime.datetime.now().isoformat(timespec='seconds'),
                script='preprocess_48k/preprocess_48k.py', git=git_state(), source=cfg['source'],
                sample_rate=FS, freq_hz='k * 48000 / 256, k = 0..128 (187.5 Hz apart)', n_bins=NBINS,
                spectrum='DTFT of the complete IR at the 129 frequencies; no truncation or windowing',
                units='dB = 20 log10(|H| + 1e-6)', db_floor=DB_FLOOR,
                interpolation=f'STPS order {SPLINE_ORDER}, lambda {SMOOTHING}, onto HEALPix nside {NSIDE} '
                              '(RING order); one matrix per distinct subject grid',
                array='(129, 3072) float32 per ear; right ears NOT mirrored',
                n_subjects=len(manifest), distinct_grids={k: v for k, v in cache.count.items()},
                min_abs_ild_db=min(abs(v) for m in manifest for k, v in m.items() if k.startswith('ild_')),
                ild_band_hz=list(cfg.get('ild_hz', (1500.0, 16000.0))),
                excluded=cfg.get('exclude', {}))
    if cfg['itd']:
        meta['itd_p90_us_range'] = [min(m['itd_p90_us'] for m in manifest), max(m['itd_p90_us'] for m in manifest)]
        meta['itd_m90_us_range'] = [min(m['itd_m90_us'] for m in manifest), max(m['itd_m90_us'] for m in manifest)]
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    json.dump(manifest, open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
    print(f'  {name}: {len(manifest)} subjects, {len(cache.count)} distinct grid(s) '
          f'{list(cache.count.values())}, min |ILD| {meta["min_abs_ild_db"]:.1f} dB, excluded {list(meta["excluded"])}', flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='+', choices=list(DATASETS), default=[k for k in DATASETS if k not in ON_REQUEST])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {name: run_dataset(name, DATASETS[name], a.out, a.overwrite) for name in a.only}
    print('\nDone.')
    for k, m in summary.items():
        print(f"  {k:<12}{m['n_subjects']:>5} subjects  grids {list(m['distinct_grids'].values())}")


if __name__ == '__main__':
    main()
