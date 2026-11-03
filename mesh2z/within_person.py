"""Within-person test: does the left-minus-right PINNA difference predict the left-minus-right
LATENT difference?

Between people, a mesh feature can predict z through anything that travels with the person
(head size, torso, age, measurement session). Within one person those cancel: the left and right
ears share the head, the session and the measurement rig, so a left-right shape difference can only
predict a left-right latent difference through the pinna itself (or through whatever differs
between the two ears' measurements). This is the cleanest form of "learn how a change in the pinna
changes z".

Inputs (all already produced by the mesh2z pipeline): the frozen-FFM26 latents
(`output/mesh2z_sonicom_latents_v1.npz`), the 164 development subjects and their order
(`output/mesh2z_rff_v1/oof_ridge.npz`), and the point samples written by `sample_meshes.py`.
The right ear's points and latent are both already mirrored into the left-ear frame.
The 36 locked subjects are not read.

Features are the same ear-crop descriptors as FFM_3_2_1: the seven per-ear simple measures (the two
head measures are the same for both ears and are dropped) and RFF mean embeddings of the
canal-relative ear crop, sigma in {0.005, 0.01, 0.02, 0.05, 0.1} mm^-1, M = 256. Kernel ridge with
alpha by inner 5-fold CV, outer 10-fold CV with one row per subject, fold-local target PCA; the
significance test is a max-statistic permutation over the three feature sets.

    python within_person.py            # ~1 min with torch (RFF on CPU), several without
"""
import csv
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
SAMPLES = os.environ.get('MESH2Z_SAMPLES', os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM/mesh2z_samples'))
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1]
M_RFF = 256
ALPHAS = np.logspace(-4, 5, 37)
N_PERM = 200
FLIP = np.array([1.0, -1.0, 1.0])

L = np.load(os.path.join(OUT, 'mesh2z_sonicom_latents_v1.npz'))
O = np.load(os.path.join(OUT, 'mesh2z_rff_v1', 'oof_ridge.npz'))
assert str(O['latents_subj_hash']) == str(L['subj_hash']), 'oof_ridge.npz and the latent cache disagree'
DEV = [str(s) for s in O['dev']]
N = len(DEV)
assert not set(DEV) & {str(s) for s in L['locked_sids']}
IDX = {s: i for i, s in enumerate(L['sid'])}
with open(os.path.join(SAMPLES, 'manifest.csv'), newline='') as f:
    MAN = {f"SON_{r['subject']}": r for r in csv.DictReader(f)}

G0 = np.random.default_rng(12345).standard_normal((M_RFF, 3))      # same draw as FFM_3_2_1
try:
    import torch
    _G0T = torch.as_tensor(G0.T, dtype=torch.float32)

    def rff(P):
        base = 2 * np.pi * (torch.as_tensor(P, dtype=torch.float32) @ _G0T)
        return np.concatenate([np.concatenate([torch.cos(s * base).mean(0).numpy(), torch.sin(s * base).mean(0).numpy()])
                               for s in SIGMAS])
except ImportError:
    def rff(P):
        base = 2 * np.pi * (P @ G0.T)
        return np.concatenate([np.concatenate([np.cos(s * base).mean(0), np.sin(s * base).mean(0)]) for s in SIGMAS])


def simple(q, canal, area):
    out = q[q[:, 1] > 3.0]
    p = np.percentile
    return [canal[0], canal[1], canal[2], p(q[:, 1], 95),
            p(out[:, 2], 95) - p(out[:, 2], 5), p(out[:, 0], 95) - p(out[:, 0], 5), area]


lines = []


def say(s=''):
    print(s, flush=True); lines.append(s)


t0 = time.time()
FR, FS, Z = [[], []], [[], []], [[], []]
for s in DEV:
    d = np.load(os.path.join(SAMPLES, f'{s[4:]}.npz'))
    ears = ((d['earL_pts'], d['ear_left_mm'].astype(float), float(MAN[s]['crop_area_L_mm2'])),
            (d['earR_pts'], d['ear_right_mm'].astype(float) * FLIP, float(MAN[s]['crop_area_R_mm2'])))
    for side, (pts, canal, area) in enumerate(ears):
        assert canal[1] > 0, f'{s}: canal not on +y after mirroring'
        q = pts.astype(np.float64) - canal
        FR[side].append(rff(q)); FS[side].append(simple(q, canal, area))
        Z[side].append(L['Z_ear'][IDX[s], side].ravel())
FR, FS, Z = [np.array(x) for x in FR], [np.array(x) for x in FS], [np.array(x, dtype=np.float64) for x in Z]
say(f'Within-person test, {N} SONICOM development subjects (features in {time.time()-t0:.0f} s)')

dZ = Z[0] - Z[1]
dF = {'simple': FS[0] - FS[1], 'ear RFF multi-sigma': FR[0] - FR[1]}
dF['simple + ear RFF'] = np.hstack([dF['simple'], dF['ear RFF multi-sigma']])
res = dict(n_subjects=N)

# how large are within-person differences relative to between-person ones?
iu = np.triu_indices(N, 1)
lz_w = float(np.median(np.linalg.norm(dZ, axis=1)))
lz_b = float(np.median(np.linalg.norm(Z[0][:, None] - Z[0][None], axis=-1)[iu]))
say(f'\nSize of within-person differences (median distance; ratio to between-person)')
say(f'    latent z          left vs right {lz_w:7.0f} | person i vs j (left ears) {lz_b:7.0f} | ratio {lz_w/lz_b:.2f}')
res['ratio_latent'] = lz_w / lz_b
for k, F0 in (('simple', FS[0]), ('ear RFF multi-sigma', FR[0])):
    sd = F0.std(0) + 1e-12
    w = float(np.median(np.linalg.norm(dF[k] / sd, axis=1)))
    b = float(np.median(np.linalg.norm((F0[:, None] - F0[None]) / sd, axis=-1)[iu]))
    res[f'ratio_geometry_{k}'] = w / b
    say(f'    {k:17s} ratio {w/b:.2f}')


# ---- kernel ridge on globally z-scored features; fold-local target PCA; permutations reorder the Gram
def gram(X):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return Xz @ Xz.T / X.shape[1]


def prep(tr, va):
    my = dZ[tr].mean(0)
    U, s, Vt = np.linalg.svd(dZ[tr] - my, full_matrices=False)
    k = int((s > s[0] * 1e-9).sum()); V = Vt[:k].T
    Yv = dZ[va] - my; Yvp = Yv @ V
    return dict(tr=tr, va=va, Ytr=U[:, :k] * s[:k], Yv=Yvp,
                resid=float((Yv ** 2).sum() - (Yvp ** 2).sum()), sst=float((Yv ** 2).sum()))


rng0 = np.random.default_rng(0)
PLAN = []
for va in np.array_split(rng0.permutation(N), 10):
    tr = np.setdiff1d(np.arange(N), va)
    PLAN.append((prep(tr, va), [prep(np.setdiff1d(tr, iv), iv) for iv in np.array_split(rng0.permutation(tr), 5)]))
assert np.array_equal(np.sort(np.concatenate([o['va'] for o, _ in PLAN])), np.arange(N))


def path(G, sp, alphas):
    tr, va = sp['tr'], sp['va']
    Gtt, Gvt = G[np.ix_(tr, tr)], G[np.ix_(va, tr)]
    cm, tot = Gtt.mean(0), Gtt.mean()
    K = Gtt - cm[None] - cm[:, None] + tot
    Kv = Gvt - cm[None] - Gvt.mean(1)[:, None] + tot
    lam, Q = np.linalg.eigh(K); lam = np.clip(lam, 0, None)
    B = Q.T @ sp['Ytr']; KQ = Kv @ Q; sc = np.trace(K) / len(tr)
    return [float(((sp['Yv'] - (KQ / (lam + a * sc)) @ B) ** 2).sum()) + sp['resid'] for a in alphas]


def cv(G):
    sse = sst = 0.0; chosen = []
    for outer, inner in PLAN:
        j = int(np.argmin(np.sum([path(G, sp, ALPHAS) for sp in inner], 0))); chosen.append(j)
        sse += path(G, outer, [ALPHAS[j]])[0]; sst += outer['sst']
    return 1 - sse / sst, chosen


GR = {k: gram(X) for k, X in dF.items()}
selfcheck = cv(gram(np.random.default_rng(5).standard_normal((N, 300))))[0]
say(f'\nSelf-check: random features CV-R2 {selfcheck:+.4f} (must be <= ~0)')
assert selfcheck < 0.01, 'random features predict the target: the pipeline leaks'

say('\nPredict z_L - z_R from (left pinna) - (mirrored right pinna): one row per subject, 10-fold CV')
obs = {}
for k, G in GR.items():
    r2, ch = cv(G); obs[k] = r2
    edge = sum(c in (0, len(ALPHAS) - 1) for c in ch)
    say(f'    {k:22s} CV-R2 {r2:+.4f}' + (f'   ({edge}/10 folds chose an alpha at the grid edge)' if edge else ''))
rng = np.random.default_rng(1)
null = np.array([max(cv(G[np.ix_(p, p)])[0] for G in GR.values()) for p in (rng.permutation(N) for _ in range(N_PERM))])
best = max(obs.values()); pval = (1 + (null >= best).sum()) / (1 + N_PERM)
say(f'    max-statistic permutation ({N_PERM}): null median {np.median(null):+.4f}, 95th pct '
    f'{np.percentile(null, 95):+.4f}; observed best {best:+.4f}, p = {pval:.3f}')
say('\nFor comparison, between-person mesh -> z (FFM_3_2_1, ear-level, same features plus head): CV-R2 +0.076.')
res.update(cv_r2=obs, null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
           p_max=float(pval), selfcheck=float(selfcheck))

open(os.path.join(OUT, 'within_person.txt'), 'w').write('\n'.join(lines) + '\n')
json.dump(res, open(os.path.join(OUT, 'within_person.json'), 'w'), indent=1)
print(f'\n-> {OUT}/within_person.txt, within_person.json')
