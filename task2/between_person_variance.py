"""Between-person spread of the DTF, simulated vs measured (thesis table tab:task2-var).

Same subjects, same DTF and same bands as task2/Task2_ShapeModes.ipynb:
    SONICOM: the 164 FFM_3_2_1 development subjects (mesh2z_sonicom_latents_v1.npz eval_sids)
    SYMARE:  the 46 development subjects of split_48k.json (train + val)
    DTF:     48 kHz data, bins 1-85, directions with elevation >= -45 deg (2652), each ear's
             direction mean removed per bin; one row per ear, right ear mirrored
Spread (rms dB) = sqrt of the mean, over ears x bins in the band x directions, of the squared
deviation from the across-ear mean (ddof 0). Variance ratio = (measured / simulated)^2.
95% interval: 1000 bootstrap resamples of subjects (both ears together), seed 0.

Inputs are checked first: the data zip against the SHA-256 the Task 2 notebook used, each local
manifest.json byte-for-byte against the zip's, every array against its manifest SHA-256, and the
split file against its SHA-256.

    python task2/between_person_variance.py     # -> results/task2_between_person_variance.json
"""
import hashlib
import json
import os
import time
import zipfile

import healpy as hp
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(REPO)
DATA_ZIP = os.environ.get('DATA48K_ZIP', f'{ROOT}/data_48k_2026-10-10.zip')
DATA = os.environ.get('DATA48K_DIR', f'{ROOT}/data_48k')
SPLIT = f'{REPO}/preprocess_48k/split_48k.json'
LATENTS = f'{REPO}/mesh2z/output/mesh2z_sonicom_latents_v1.npz'
OUT = f'{REPO}/results/task2_between_person_variance.json'
ZIP_SHA256 = 'f51c6b072123e70d6d8a0691373ab90e3f6db9856ce6306f65d949c87fbf97c7'      # Task2_ShapeModes.ipynb Cell 0
SPLIT_SHA256 = 'dc89bc454fef351951283cccb3772fdd3e552bbdde5d13668f8f576f4cd83561'
N_BINS, FS, N_BOOT, SEED = 86, 48000, 1000, 0
DTF_F = (np.arange(N_BINS) * FS / 256)[1:]
BANDS = {'0.2-5 kHz': (200, 5000), '6-12 kHz': (6000, 12000), '12-16 kHz': (12000, 16000)}
PAIRS = {'SONICOM': ('SONICOM_SIM', 'SONICOM'), 'SYMARE': ('SYMARE_BEM', 'SYMARE')}
# the values printed in the 10-12 draft (local check, no script kept); this script must reproduce them
DRAFT = {'SONICOM': [(0.97, 1.72, 3.1), (3.98, 4.56, 1.3), (4.56, 4.63, 1.0)],
         'SYMARE': [(0.96, 1.79, 3.5), (4.23, 4.26, 1.0), (4.66, 4.48, 0.9)]}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs
t0 = time.time()
assert sha256(DATA_ZIP) == ZIP_SHA256, f'{DATA_ZIP}: not the zip the Task 2 notebook used'
assert sha256(SPLIT) == SPLIT_SHA256, f'{SPLIT}: SHA-256 differs'
with zipfile.ZipFile(DATA_ZIP) as z:
    for tag in ('SONICOM', 'SONICOM_SIM', 'SYMARE', 'SYMARE_BEM'):
        assert z.read(f'data_48k/{tag}_48k/manifest.json') == open(f'{DATA}/{tag}_48k/manifest.json', 'rb').read(), tag
    assert z.read('data_48k/healpix_mirror_idx.npy') == open(f'{DATA}/healpix_mirror_idx.npy', 'rb').read()
print(f'zip, manifests and split verified ({time.time() - t0:.0f} s)')

_th, _ = hp.pix2ang(16, np.arange(3072))
MEAS_DIRS = (90 - np.degrees(_th)) >= -45.0
assert MEAS_DIRS.sum() == 2652
MIRROR = np.load(f'{DATA}/healpix_mirror_idx.npy')
assert np.array_equal(MIRROR[MIRROR], np.arange(3072))
split = json.load(open(SPLIT))
DEV = {'SONICOM': [str(s) for s in np.load(LATENTS)['eval_sids']],
       'SYMARE': sorted(split['SYMARE']['train'] + split['SYMARE']['val'])}
assert len(DEV['SONICOM']) == 164 and len(DEV['SYMARE']) == 46
for db, ids in DEV.items():
    assert not set(ids) & set(split[db]['test']), f'{db}: a locked test subject in the development set'


def load_rows(tag, ids):
    """As in the notebook: two rows per subject (left, mirrored right); every array checked."""
    d = f'{DATA}/{tag}_48k'
    man = {m['id']: m for m in json.load(open(f'{d}/manifest.json'))}
    X = np.empty((2 * len(ids), N_BINS, 3072), np.float32)
    for k, s in enumerate(ids):
        for side in ('left', 'right'):
            f = f'{d}/npy_{side}/{man[s][side]}'
            assert sha256(f) == man[s][f'{side}_sha256'], f
            a = np.load(f)[:N_BINS]
            X[2 * k + (side == 'right')] = a[:, MIRROR] if side == 'right' else a
    return X


def dtf(X):
    D = X[:, 1:, :][:, :, MEAS_DIRS]
    return (D - D.mean(2, keepdims=True)).astype(np.float32)


def spread(x):
    return float(np.sqrt(((x - x.mean(0)) ** 2).mean()))


def boot_spreads(x, W):
    """rms spread for each row of subject-count weights W (n_boot, n_subjects); x (n_ears, ...)."""
    Xf = x.reshape(len(x), -1).astype(np.float64)
    We = np.repeat(W, 2, axis=1).astype(np.float64)                    # both ears of a subject together
    n = We.sum(1, keepdims=True)
    out = []
    for i in range(0, len(We), 100):
        w = We[i:i + 100]
        m1, m2 = (w @ Xf) / n[i:i + 100], (w @ Xf ** 2) / n[i:i + 100]
        out.append(np.sqrt((m2 - m1 ** 2).mean(1)))
    return np.concatenate(out)


# ---------------------------------------------------------------- compute
rng = np.random.default_rng(SEED)
res = {}
for db, (sim_tag, meas_tag) in PAIRS.items():
    t0 = time.time()
    Ds, Dm = dtf(load_rows(sim_tag, DEV[db])), dtf(load_rows(meas_tag, DEV[db]))
    ns = len(DEV[db])
    W = np.stack([np.bincount(rng.integers(0, ns, ns), minlength=ns) for _ in range(N_BOOT)])
    res[db] = dict(n_subjects=ns, n_ears=2 * ns, simulated=sim_tag, measured=meas_tag, bands={})
    for (band, (lo, hi)), draft in zip(BANDS.items(), DRAFT[db]):
        k = (DTF_F >= lo) & (DTF_F <= hi)
        s, m = spread(Ds[:, k]), spread(Dm[:, k])
        ratio = (m / s) ** 2
        bs, bm = boot_spreads(Ds[:, k], W), boot_spreads(Dm[:, k], W)
        ci = np.percentile((bm / bs) ** 2, [2.5, 97.5])
        res[db]['bands'][band] = dict(n_bins=int(k.sum()), rms_sim_db=s, rms_meas_db=m, var_ratio=ratio,
                                      var_ratio_ci95=[float(ci[0]), float(ci[1])])
        assert (round(s, 2), round(m, 2), round(ratio, 1)) == draft, (db, band, s, m, ratio, draft)
    print(f'{db}: {ns} subjects, {time.time() - t0:.0f} s')
print('reproduces the values in the 10-12 draft table')

json.dump(dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), script='task2/between_person_variance.py',
               data_zip_sha256=ZIP_SHA256, split_sha256=SPLIT_SHA256,
               definition='rms over ears x band bins x 2652 directions of (DTF - across-ear mean), ddof 0; '
                          'var_ratio = (measured / simulated)^2; CI: subject bootstrap, both ears together',
               n_boot=N_BOOT, seed=SEED, databases=res), open(OUT, 'w'), indent=1)
print(f'-> {OUT}\n')
print(f"{'':10s}" + ''.join(f'{b:>34s}' for b in BANDS))
for db in PAIRS:
    print(f'{db:10s}' + ''.join(f"{r['rms_sim_db']:>9.2f} / {r['rms_meas_db']:.2f} ({r['var_ratio']:.1f}x) "
                                f"[{r['var_ratio_ci95'][0]:.1f}, {r['var_ratio_ci95'][1]:.1f}]"
                                for r in res[db]['bands'].values()))
