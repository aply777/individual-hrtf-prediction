"""Why mesh -> latent is weak: the physics ceiling on SONICOM.

Asks whether an exact acoustic solve from the SAME mesh predicts the person's MEASURED HRTF
better than the population mean. SONICOM ships, for its 200 quality-controlled scans, a
Mesh2HRTF (BEM) simulation on the measured 793-direction grid; both are pushed through the
project's own STPS -> HEALPix pipeline (asserted to reproduce the stored measured arrays to
0 dB) and compared with the ridge mesh -> z -> decoder predictions of FFM_3_2_1.

Run in two steps, in two processes (torch and h5py segfault together in one local venv):

    python physics_ceiling.py decode     # needs torch; writes output/physics_ceiling/ridge_decoded.npy
    python physics_ceiling.py analyse    # needs h5py + healpy; writes report.txt and results.json

Only the 164 development subjects are read. The 36 locked subjects are not touched.

Conventions. "DTF" = each ear's HRTF minus its own direction-mean spectrum over the measured
sphere, which removes level and calibration so a simulation and a measurement can be compared.
"Measured sphere" = HEALPix pixels at elevation >= -45 deg; SONICOM measures nothing below.
Errors are pooled RMSE over ears x 93 bins (DC-15.85 kHz) x pixels. Right ears are mirrored
into the left-ear frame throughout, as in training.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output', 'physics_ceiling')
NB = 93


def args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['decode', 'analyse'])
    ap.add_argument('--oof', default=os.path.join(HERE, 'output', 'mesh2z_rff_v1', 'oof_ridge.npz'))
    ap.add_argument('--ckpt', default=os.environ.get('FFM26_CKPT', os.path.join(REPO, 'results', 'FFM26_best.pth')))
    ap.add_argument('--sonicom', default=os.environ.get(
        'SONICOM_ROOT', os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM')))
    return ap.parse_args()


# ------------------------------------------------------------------ step 1: decode (torch only)
def decode(a):
    import torch
    nb = json.load(open(os.path.join(REPO, 'landmark2z', 'FFM_2_8_Landmark2Z.ipynb')))
    src = ''.join(nb['cells'][3]['source'])
    assert 'class FFM_OSLO_Autoencoder' in src, 'FFM_2_8 cell 3 is no longer the model definition'
    ns = {}
    exec(src, ns)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = ns['FFM_OSLO_Autoencoder'](nside=16, latent_dim_per_freq=64, freq_embed_dim=32,
                                       num_datasets=4, num_bins=NB, tie_encoder_film=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    O = np.load(a.oof)
    n = len(O['dev'])
    Z = O['pred_best'].reshape(n, 2, NB, 64)
    P = np.zeros((n, 2, NB, 3072), np.float32)
    with torch.no_grad():
        for i in range(n):
            P[i] = model.decoder(torch.from_numpy(Z[i]).float(), torch.full((2,), 2)).numpy()
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, 'ridge_decoded.npy'), P)
    print(f"decoded {n} subjects' out-of-fold ridge predictions ({str(O['best'])}) -> {OUT}/ridge_decoded.npy")


# ------------------------------------------------------------------ step 2: analyse (h5py only)
def analyse(a):
    import h5py
    import healpy as hp
    sys.path.insert(0, os.path.join(REPO, 'analysis'))
    import preprocess_multidataset_stps as pp

    S = a.sonicom
    O = np.load(a.oof)
    dev = [str(s) for s in O['dev']]
    n = len(dev)
    th, ph = hp.pix2ang(16, np.arange(3072))
    MEAS = (90 - np.degrees(th)) >= -45
    ALL = np.ones(3072, bool)
    MIRROR = hp.ang2pix(16, th, (-ph) % (2 * np.pi))
    F = np.arange(NB) * 44100 / 256
    NOTCH, LOWF = (F >= 6000) & (F <= 12000), F < 500
    lines = []

    def say(s=''):
        print(s); lines.append(s)

    def sofa(p):
        with h5py.File(p, 'r') as f:
            return np.array(f['Data.IR']), float(np.array(f['Data.SamplingRate']).ravel()[0]), np.array(f['SourcePosition'])

    # the simulation must go through exactly the pipeline that made the measured arrays
    ir, fs, pos = sofa(f'{S}/dataset/P0001/HRTF/HRTF/44kHz/P0001_FreeFieldCompMinPhase_44kHz.sofa')
    W = pp.build_interpolation_matrix(pp.positions_to_xyz(pos, 'spherical'), pp.healpix_xyz())
    d = np.abs(pp.hrir_to_mag_healpix(ir[:, 0, :], fs, W)[:NB]
               - np.load(f'{S}/npy_left/SON_P0001_left_full.npy')[:NB]).max()
    assert d < 1e-3, f'pipeline does not reproduce the stored measured array ({d:.3g} dB)'
    say(f'Pipeline check: reprocessing P0001 reproduces the stored measured array, max |diff| {d:.1e} dB')

    M = np.zeros((n, 2, NB, 3072)); SIM = np.zeros_like(M)
    for i, s in enumerate(dev):
        M[i, 0] = np.load(f'{S}/npy_left/{s}_left_full.npy')[:NB]
        M[i, 1] = np.load(f'{S}/npy_right/{s}_right_full.npy')[:NB][:, MIRROR]
        ir, fs, pos = sofa(f'{S}/synthetic/{s[4:]}/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa')
        assert ir.shape[0] == 793 and fs == 44100
        with h5py.File(f'{S}/dataset/{s[4:]}/HRTF/HRTF/44kHz/{s[4:]}_FreeFieldCompMinPhase_44kHz.sofa', 'r') as f:
            mpos = np.array(f['SourcePosition'])
        assert np.allclose(pp.positions_to_xyz(pos, 'spherical'), pp.positions_to_xyz(mpos, 'spherical'), atol=1e-6), \
            f'{s}: simulated and measured direction grids differ (the shared STPS matrix would be wrong)'
        with contextlib.redirect_stdout(io.StringIO()):      # 294-sample IRs: silence the truncation note
            SIM[i, 0] = pp.hrir_to_mag_healpix(ir[:, 0, :], fs, W)[:NB]
            SIM[i, 1] = pp.hrir_to_mag_healpix(ir[:, 1, :], fs, W)[:NB][:, MIRROR]

    dec = os.path.join(OUT, 'ridge_decoded.npy')
    P = np.load(dec).astype(np.float64) if os.path.exists(dec) else None
    if P is None:
        say('(no ridge_decoded.npy: run the decode step first to include the ridge rows)')

    def dtf(X): return X - X[..., MEAS].mean(-1, keepdims=True)
    def rms(e, pix=MEAS): return float(np.sqrt((e[..., pix] ** 2).mean()))
    loo = lambda X: (X.mean(0, keepdims=True) * n - X) / (n - 1)
    res = dict(n_subjects=n, dev_hash=str(O['latents_subj_hash']))

    # D1 -- how much of between-person variation is direction-independent
    say(f'\nD1  Between-person variation of the measured HRTF (measured sphere, {n} subjects x 2 ears)')
    dv = M - M.mean(0, keepdims=True); c = M[..., MEAS].mean(-1); cv = c - c.mean(0, keepdims=True)
    res['D1'] = {}
    for name, m in (('all bins', slice(None)), ('< 0.5 kHz', LOWF), ('6-12 kHz', NOTCH)):
        tot = (dv[:, :, m][..., MEAS] ** 2).mean(); share = (cv[:, :, m] ** 2).mean() / tot
        res['D1'][name] = dict(rms=float(np.sqrt(tot)), direction_independent=float(share))
        say(f'    {name:10s} RMS {np.sqrt(tot):.2f} dB | direction-independent {share:5.1%} | direction-dependent {1-share:5.1%}')

    # D2 -- the two ears of one person against two different people
    Lm, Rm = M[:, 0], M[:, 1]; perm = np.roll(np.arange(n), 1)
    res['D2'] = dict(full_same=rms(Lm - Rm), full_other=rms(Lm - Rm[perm]),
                     dtf_same=rms(dtf(Lm) - dtf(Rm)), dtf_other=rms(dtf(Lm) - dtf(Rm[perm])),
                     dtf_notch_same=rms(dtf(Lm)[:, NOTCH] - dtf(Rm)[:, NOTCH]),
                     dtf_notch_other=rms(dtf(Lm)[:, NOTCH] - dtf(Rm[perm])[:, NOTCH]))
    r2 = res['D2']
    say(f"\nD2  Left vs mirrored right ear      same person   different person")
    say(f"    full HRTF                       {r2['full_same']:8.2f}      {r2['full_other']:8.2f} dB")
    say(f"    DTF                             {r2['dtf_same']:8.2f}      {r2['dtf_other']:8.2f} dB")
    say(f"    DTF, 6-12 kHz                   {r2['dtf_notch_same']:8.2f}      {r2['dtf_notch_other']:8.2f} dB")

    # D3 -- physics ceiling, one metric for every predictor
    Md, Sd = dtf(M), dtf(SIM); base = loo(Md)
    Sc = Sd - loo(Sd - Md)                                    # sim-vs-measurement bias, leave-one-out
    num = ((Md - base) * (Sc - base))[..., MEAS].sum((1, 2, 3)); den = ((Sc - base) ** 2)[..., MEAS].sum((1, 2, 3))
    beta = (num.sum() - num) / (den.sum() - den)              # optimal shrinkage, leave-one-out
    Ss = base + beta[:, None, None, None] * (Sc - base)
    sse0 = ((base - Md)[..., MEAS] ** 2).sum(); sse0n = ((base - Md)[:, :, NOTCH][..., MEAS] ** 2).sum()

    def ident(X):
        r = []
        for side in (0, 1):
            A = X[:, side][..., MEAS].reshape(n, -1); B = Md[:, side][..., MEAS].reshape(n, -1)
            d2 = (A * A).sum(1)[None] + (B * B).sum(1)[:, None] - 2 * B @ A.T
            r.append((d2 < d2.diagonal()[:, None]).sum(1) + 1)
        r = np.concatenate(r); return float(np.mean(r == 1)), float(np.median(r))

    rows = [('population mean (leave-one-out)', base)]
    if P is not None:
        rows.append(('ridge: mesh features -> z -> decoder', dtf(P)))
    rows += [('own Mesh2HRTF simulation, bias removed', Sc),
             (f'own simulation, shrunk to mean (beta {beta.mean():.2f})', Ss)]
    say(f'\nD3  Predicting each ear\'s measured DTF (measured sphere). Chance identification: rank-1 {1/n:.1%}, median rank {(n+1)/2:.0f}')
    say(f"    {'predictor':48s}{'RMS dB':>8s}{'6-12k':>8s}{'R2':>8s}{'R2 6-12k':>10s}{'rank-1':>8s}{'med.rank':>9s}")
    res['D3'] = {}
    for name, X in rows:
        e = X - Md
        row = dict(rms=rms(e), rms_notch=rms(e[:, :, NOTCH]),
                   r2=float(1 - (e[..., MEAS] ** 2).sum() / sse0),
                   r2_notch=float(1 - (e[:, :, NOTCH][..., MEAS] ** 2).sum() / sse0n))
        if not name.startswith('population'):
            row['rank1'], row['median_rank'] = ident(X)
        res['D3'][name] = row
        k = f"{row['rank1']:8.1%}{row['median_rank']:9.0f}" if 'rank1' in row else ''
        say(f"    {name:48s}{row['rms']:8.2f}{row['rms_notch']:8.2f}{row['r2']:+8.3f}{row['r2_notch']:+10.3f}{k}")
    res['beta_mean'] = float(beta.mean())

    # D4 -- the 2026-08-08 gain-aligned magnitude convention (full magnitude, one scalar dB gain removed
    # per ear, 0.2-16 kHz, r of deviations from the mean) -- but with right ears mirrored, i.e. per side
    B = F >= 200
    ga = lambda X: X - X[:, :, B][..., MEAS].mean((-1, -2), keepdims=True)
    m, sm = ga(M)[:, :, B][..., MEAS], ga(SIM)[:, :, B][..., MEAS]
    dm, ds = m - m.mean(0, keepdims=True), sm - sm.mean(0, keepdims=True)
    A_, B_ = sm.reshape(n, -1), m.reshape(n, -1)
    d2 = (A_ * A_).sum(1)[None] + (B_ * B_).sum(1)[:, None] - 2 * B_ @ A_.T
    res['D4'] = dict(r=float(np.corrcoef(dm.ravel(), ds.ravel())[0, 1]),
                     r_notch=float(np.corrcoef(dm[:, :, NOTCH[B]].ravel(), ds[:, :, NOTCH[B]].ravel())[0, 1]),
                     argmin_rate=float(np.mean(d2.argmin(1) == np.arange(n))))
    d4 = res['D4']
    say(f"\nD4  Gain-aligned magnitude, 0.2-16 kHz, per side: r(dev_meas, dev_sim) = {d4['r']:+.3f} "
        f"(r^2 {d4['r']**2:.3f}); 6-12 kHz {d4['r_notch']:+.3f}; own simulation is the argmin for {d4['argmin_rate']:.1%}")
    say('    Mirrored right ears, so deviations are per side. For SYMARE vs SONICOM on one footing see sim_vs_measured.py:')
    say('    the 08-08 SYMARE r = 0.756 mixed left and right ears; per side it is 0.225 (SONICOM 0.164).')
    if P is not None:
        res['corr_sim_ridge'] = float(np.corrcoef(((Sc - base)[..., MEAS]).ravel(), ((dtf(P) - base)[..., MEAS]).ravel())[0, 1])
        say(f"    correlation of (simulation - mean) with (ridge - mean): {res['corr_sim_ridge']:+.3f}")

        # bridge to the FFM_3_2_1 headline: the same ridge predictions under four conventions
        say('\nBridge  The same ridge predictions under four conventions (mean = leave-one-out over these subjects)')
        say(f"    {'convention':48s}{'mean':>8s}{'ridge':>8s}{'gain':>7s}")
        res['bridge'] = {}
        for name, f, pix in (('full HRTF, all 3072 directions (FFM_3_2_1)', lambda X: X, ALL),
                             ('full HRTF, measured sphere', lambda X: X, MEAS),
                             ('DTF, all directions', dtf, ALL),
                             ('DTF, measured sphere (D3)', dtf, MEAS)):
            b, r = rms(f(loo(M)) - f(M), pix), rms(f(P) - f(M), pix)
            res['bridge'][name] = dict(mean=b, ridge=r)
            say(f'    {name:48s}{b:8.3f}{r:8.3f}{b-r:7.3f}')

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    print(f'\n-> {OUT}/report.txt, results.json')


if __name__ == '__main__':
    a = args()
    decode(a) if a.mode == 'decode' else analyse(a)
