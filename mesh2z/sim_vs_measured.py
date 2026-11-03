"""Does a BEM simulation from a person's own mesh reproduce how their MEASURED HRTF deviates
from the population? SYMARE (MRI meshes, BEM by the database authors, N = 61) and SONICOM
(optical scans, Mesh2HRTF by the curators, 164 development subjects), with identical code on
each database's native measurement grid (no interpolation).

Deviations are taken two ways. "mixed": one mean over all ears with left and right together
and nothing mirrored -- what analysis/bem_vs_measured.py did on 2026-08-08. "per-side": each
side against its own population mean. The mixed version adds the systematic left/right
difference to every deviation, and the simulation shares it, so it inflates the correlation:
SYMARE 0.756 mixed vs 0.225 per-side. Only the per-side figures measure individual variation.

    python sim_vs_measured.py        # prints the report and writes output/sim_vs_measured.txt
"""
import contextlib
import io
import numpy as np, scipy.io as sio, h5py, os
NF = 512
def mag(h, fs):                     # h: (ndir, ntaps) -> (ndir, nbins) dB, plus freqs
    H = np.fft.rfft(h, n=NF, axis=-1); return 20 * np.log10(np.abs(H) + 1e-12), np.fft.rfftfreq(NF, 1 / fs)

# ---------------- SYMARE: Acoustic HA (measured) vs Bem/HE16kHz HB (BEM from MRI meshes), 393 directions, 48 kHz
R = os.path.expanduser('~/Documents/26S2/ELEC5021/SYMARE/HRIRs')
symM, symS = [], []
for i in range(1, 62):
    a, b = f'{R}/Acoustic/HA{i:02d}.mat', f'{R}/Bem/HE16kHz/HB{i:02d}.mat'
    if not (os.path.exists(a) and os.path.exists(b)): continue
    A, B = sio.loadmat(a), sio.loadmat(b)
    symM.append([mag(A['hL'].T, 48000)[0], mag(A['hR'].T, 48000)[0]])
    symS.append([mag(B['hL'].T, 48000)[0], mag(B['hR'].T, 48000)[0]])
symM, symS = np.array(symM), np.array(symS); fsym = np.fft.rfftfreq(NF, 1 / 48000)
print(f'SYMARE: {len(symM)} subjects with measured + HE16kHz BEM, shape {symM.shape}')

# ---------------- SONICOM: measured FreeFieldCompMinPhase vs Mesh2HRTF (optical scans), 793 directions, 44.1 kHz
S = os.path.expanduser('~/Documents/26S1/ELEC5020/SONICOM')
dev = [str(s) for s in np.load('/Users/harry/Documents/26S2/ELEC5021/hrtf-GitHub/mesh2z/output/mesh2z_rff_v1/oof_ridge.npz')['dev']]
def sofa(p):
    with h5py.File(p, 'r') as f: return np.array(f['Data.IR']), np.array(f['SourcePosition'])
sonM, sonS = [], []
for s in dev:
    im, pm = sofa(f'{S}/dataset/{s[4:]}/HRTF/HRTF/44kHz/{s[4:]}_FreeFieldCompMinPhase_44kHz.sofa')
    isim, ps = sofa(f'{S}/synthetic/{s[4:]}/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa')
    assert np.allclose(pm, ps)
    sonM.append([mag(im[:, 0], 44100)[0], mag(im[:, 1], 44100)[0]])
    sonS.append([mag(isim[:, 0], 44100)[0], mag(isim[:, 1], 44100)[0]])
sonM, sonS = np.array(sonM), np.array(sonS); fson = np.fft.rfftfreq(NF, 1 / 44100)
print(f'SONICOM: {len(sonM)} subjects, shape {sonM.shape}\n')

def analyse(Mx, Sx, f, label):
    # Mx, Sx: (n, 2 sides, ndir, nbins)
    out = {}
    for bname, lo, hi in (('0.2-16 kHz', 200, 16000), ('0.2-5 kHz', 200, 5000), ('6-12 kHz', 6000, 12000)):
        fm = (f >= lo) & (f <= hi)
        m, s = Mx[..., fm], Sx[..., fm]
        res = {}
        for how in ('mixed', 'per-side'):
            if how == 'mixed':     # 08-08 script: one mean over all ears, left and right together, no mirroring
                mm = m.reshape(-1, *m.shape[2:]).mean(0); sm = s.reshape(-1, *s.shape[2:]).mean(0)
                dm, ds = m - mm, s - sm
            else:                  # each side against its own population mean
                dm, ds = m - m.mean(0, keepdims=True), s - s.mean(0, keepdims=True)
            a = dm.reshape(-1, dm[0, 0].size); b = ds.reshape(-1, ds[0, 0].size)
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)      # one scalar per ear
            r_ear = np.array([np.corrcoef(a[j], b[j])[0, 1] for j in range(len(a))])
            r_pool = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            r2 = 1 - ((a - b) ** 2).sum() / (a ** 2).sum()                           # BEM used directly as predictor
            res[how] = (r_ear.mean(), r_pool, r2)
        out[bname] = res
    # identification, per side, own-side means: is the person's own simulation the closest to their measurement?
    fm = (f >= 200) & (f <= 16000); n = len(Mx); hits = []
    for side in (0, 1):
        A = (Sx[:, side][..., fm]); B = (Mx[:, side][..., fm])
        A = (A - A.mean((1, 2), keepdims=True)).reshape(n, -1); B = (B - B.mean((1, 2), keepdims=True)).reshape(n, -1)
        d2 = (A * A).sum(1)[None] + (B * B).sum(1)[:, None] - 2 * B @ A.T
        hits.append(d2.argmin(1) == np.arange(n))
    print(f'{label}  (N = {n}; identification: own simulation closest for {np.mean(hits):.1%}, chance {1/n:.1%})')
    print(f"    {'band':11s}{'deviation from':>16s}{'r per ear (mean)':>18s}{'r pooled':>10s}{'R2 as predictor':>17s}")
    for bname, res in out.items():
        for how, (re, rp, r2) in res.items():
            print(f'    {bname:11s}{how:>16s}{re:+18.3f}{rp:+10.3f}{r2:+17.3f}')
    print()

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    analyse(symM, symS, fsym, 'SYMARE  measured vs BEM (HE16kHz, MRI meshes, database authors)')
    analyse(sonM, sonS, fson, 'SONICOM measured vs Mesh2HRTF (optical scans, SONICOM curators)')
print(buf.getvalue())
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'), exist_ok=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'sim_vs_measured.txt'), 'w').write(buf.getvalue())
