"""Population-average LSD on HUTUBS under the protocol GISNO's authors described (personal communication, 2026-10-27; thesis Section 4.2).

Protocol as stated: reference = 256-point FFT magnitude of the measured HUTUBS HRIRs; LSD over 200 Hz - 8 kHz,
computed per ear and averaged over both ears, then over the 440 directions and the 15 test subjects (40 train / 15 test).
The authors report 6.25 dB for the training-set mean on their 15 test subjects.

Their 15 test subjects are not known, so the 40/15 split is drawn at random 500 times from the 94 HUTUBS subjects
(pp1 and pp96, the FABIAN dummy head, excluded). Four ways of forming the mean / the dB scale are compared.
Read-only, local: python3 -I gisno_baseline_check.py
"""
import glob, re
import h5py
import numpy as np

D = '/Users/harry/Documents/26S1/ELEC5020/HUTUBS/HRIRs/'
GISNO_LSD = 2.92

ids, F = [], []
for p in sorted(glob.glob(D + 'pp*_HRIRs_measured.sofa')):
    s = int(re.search(r'pp(\d+)_', p).group(1))
    if s in (1, 96):
        continue
    with h5py.File(p, 'r') as f:
        assert f['Data.SamplingRate'][()][0] == 44100 and f['Data.IR'].shape == (440, 2, 256)
        F.append(np.fft.rfft(f['Data.IR'][()], 256, axis=-1))
    ids.append(s)
F = np.stack(F)                                           # (94, 440, 2, 129), complex
f = np.fft.rfftfreq(256, 1 / 44100)
band = (f >= 200) & (f <= 8000)                           # 45 bins, 344.5 - 7924.2 Hz
Fb = F[..., band]
A = np.abs(Fb)
print(f'{len(ids)} subjects; {band.sum()} bins, {f[band][0]:.1f}-{f[band][-1]:.1f} Hz')


def lsd(Hdb, Rdb):                                        # (440, 2, bins) -> (440,), mean of the two ears
    return np.sqrt(((Hdb - Rdb) ** 2).mean(-1)).mean(-1)


VARIANTS = {
    'mean of dB spectra (20 log10 |H|)': (lambda tr: (20 * np.log10(A[tr])).mean(0), 20),
    'mean of linear magnitudes': (lambda tr: 20 * np.log10(A[tr].mean(0)), 20),
    'magnitude of the mean complex spectrum': (lambda tr: 20 * np.log10(np.abs(Fb[tr].mean(0))), 20),
    'mean of dB spectra, 20 log10 of power': (lambda tr: (40 * np.log10(A[tr])).mean(0), 40),
}
for name, (make_ref, scale) in VARIANTS.items():
    rng = np.random.default_rng(0)
    means, stds = [], []
    for _ in range(500):
        perm = rng.permutation(len(ids))
        tr, te = perm[:40], perm[40:55]
        R = make_ref(tr)
        L = np.stack([lsd(scale * np.log10(A[t]), R) for t in te])   # (15, 440)
        means.append(L.mean())
        stds.append(L.std())
    m = np.array(means)
    lo, hi = np.percentile(m, [2.5, 97.5])
    print(f'{name:40s} median {np.median(m):.2f} dB, 95% of splits [{lo:.2f}, {hi:.2f}], max {m.max():.2f}; '
          f'std over subjects x directions {np.median(stds):.2f}; 2.92 dB would be {100 * (1 - GISNO_LSD / np.median(m)):.0f}% below')
