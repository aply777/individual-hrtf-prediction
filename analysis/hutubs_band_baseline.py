"""How much the frequency range alone moves the population-mean LSD on HUTUBS (2026-10-27, for Ch2 sec:bg-metrics).

Same data, subjects and splits as gisno_baseline_check.py (raw measured HUTUBS HRIRs, 94 subjects without FABIAN,
256-point FFT, mean of the training subjects' dB spectra, LSD per ear then mean of ears, 500 random 40/15 splits);
only the upper frequency limit changes. Read-only, local: python3 -I hutubs_band_baseline.py
"""
import glob, re
import h5py
import numpy as np

D = '/Users/harry/Documents/26S1/ELEC5020/HUTUBS/HRIRs/'
A = []
for p in sorted(glob.glob(D + 'pp*_HRIRs_measured.sofa')):
    if int(re.search(r'pp(\d+)_', p).group(1)) in (1, 96):
        continue
    with h5py.File(p, 'r') as f:
        A.append(20 * np.log10(np.abs(np.fft.rfft(f['Data.IR'][()], 256, axis=-1))))
A = np.stack(A)                                            # (94, 440, 2, 129) dB
f = np.fft.rfftfreq(256, 1 / 44100)
for lo, hi in [(200, 8000), (200, 16000), (200, 20000)]:
    band = (f >= lo) & (f <= hi)
    rng = np.random.default_rng(0)
    m = []
    for _ in range(500):
        perm = rng.permutation(len(A)); tr, te = perm[:40], perm[40:55]
        R = A[tr][..., band].mean(0)
        m.append(np.mean([np.sqrt(((A[t][..., band] - R) ** 2).mean(-1)).mean(-1).mean() for t in te]))
    m = np.array(m)
    print(f'{lo/1000:.1f}-{hi/1000:.0f} kHz ({band.sum()} bins): median {np.median(m):.2f} dB, 95% of splits [{np.percentile(m,2.5):.2f}, {np.percentile(m,97.5):.2f}]')
