#!/usr/bin/env python3
"""
SYMARE: BEM vs measured HRTF discrepancy.

Purpose (ICASSP briefing 5ter.3):
  Zhang, Wang & James (ICASSP 2021) report that on HUTUBS, BEM "converges to a
  wrong distribution": BEM-vs-measured SD stays ~7.23 dB no matter how fine the
  mesh. Their meshes had NO TORSO. SYMARE ships both a head+ears (HE, valid to
  16 kHz) and a head+torso+ears (HTE, valid to 5 kHz) BEM set, plus measured
  HRIRs on the identical 393-direction grid -- so we can (a) measure our own
  gap, and (b) isolate how much of it is the missing torso.

Outputs: a printed report + figures + a .npz of per-subject numbers.

CORRECTION (2026-10-02, PROJECT_LOG §5C #21). Section 5 ("does BEM capture the right
individual variation?") takes each ear's deviation from ONE mean over all 122 ears, left and
right together, with nothing mirrored. Every deviation then carries the systematic left/right
difference, which the BEM shares, so the correlation is inflated: 0.756 here, 0.225 when each
side is compared with its own mean. Use mesh2z/sim_vs_measured.py for the per-side figures.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = os.path.expanduser("~/Documents/26S2/ELEC5021/SYMARE/HRIRs")
OUT = os.path.expanduser("~/Documents/26S2/ELEC5021/analysis")
os.makedirs(OUT, exist_ok=True)

NFFT = 256          # HRIRs are 256 samples
FS = 48000.0
EPS = 1e-12
NOTCH_BAND = (6000.0, 12000.0)   # the band this project cares about


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_set(subdir, prefix, n=61):
    """Return dict id -> (hL, hR), each (256, 393)."""
    out = {}
    for i in range(1, n + 1):
        p = os.path.join(ROOT, subdir, f"{prefix}{i:02d}.mat")
        if not os.path.exists(p):
            continue
        d = sio.loadmat(p)
        out[i] = (d["hL"].astype(float), d["hR"].astype(float))
    return out


def mag_db(h):
    """h: (256, 393) real HRIR -> (129, 393) magnitude in dB.

    Magnitude is invariant to pure delay, so the different time origins of BEM
    vs measured do not matter here.
    """
    H = np.fft.rfft(h, n=NFFT, axis=0)          # (129, 393)
    return 20.0 * np.log10(np.abs(H) + EPS)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def band_mask(freqs, lo, hi):
    return (freqs >= lo) & (freqs <= hi)


def distortions(A_db, B_db, fmask):
    """A,B: (129, 393) dB. Returns dict of scalar metrics over the band.

    raw       : no alignment (includes any calibration/level offset)
    gain      : after removing the single best scalar dB offset  (= std of diff)
    perfreq   : after removing a per-frequency dB offset -> isolates *directional*
                shape error, i.e. the part a global EQ cannot fix
    L1 (SD)   : mean |dB difference|, matching Zhang et al.'s eq. (8) definition,
                so our numbers are comparable to their 7.23 dB
    """
    D = (A_db - B_db)[fmask, :]                      # (nf, ndir)
    raw_rms = np.sqrt(np.mean(D ** 2))
    raw_l1 = np.mean(np.abs(D))

    g = D.mean()
    Dg = D - g
    gain_rms = np.sqrt(np.mean(Dg ** 2))
    gain_l1 = np.mean(np.abs(Dg))

    Dp = D - D.mean(axis=1, keepdims=True)           # per-frequency offset
    perfreq_rms = np.sqrt(np.mean(Dp ** 2))

    return dict(raw_rms=raw_rms, raw_l1=raw_l1, offset_db=g,
                gain_rms=gain_rms, gain_l1=gain_l1, perfreq_rms=perfreq_rms)


def per_freq_curve(A_db, B_db, align_mask):
    """RMS dB error vs frequency -> (129,).

    The single scalar gain offset MUST be estimated inside the band where the
    simulation is valid; using the full 0-24 kHz range lets the invalid
    high-frequency bins corrupt the offset for every other bin.
    """
    D = A_db - B_db
    D = D - D[align_mask, :].mean()
    return np.sqrt(np.mean(D ** 2, axis=1))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)          # (129,) 0..24000, 187.5 Hz step

    meas = load_set("Acoustic", "HA")
    he = load_set("Bem/HE16kHz", "HB")               # head + ears, valid <= 16 kHz
    hte = load_set("Bem/HTE05kHz", "HB")             # head + torso + ears, <= 5 kHz
    ids = sorted(set(meas) & set(he) & set(hte))
    print(f"subjects: measured={len(meas)}  HE={len(he)}  HTE={len(hte)}  common={len(ids)}")

    # Precompute dB magnitudes, both ears stacked as separate "samples"
    M, E, T = {}, {}, {}
    for i in ids:
        M[i] = [mag_db(meas[i][0]), mag_db(meas[i][1])]
        E[i] = [mag_db(he[i][0]), mag_db(he[i][1])]
        T[i] = [mag_db(hte[i][0]), mag_db(hte[i][1])]

    # ---------------------------------------------------------------- ordering
    # Does HA{n} correspond to HB{n}?  Correct pairing must beat mismatched.
    print("\n" + "=" * 74)
    print("1. SUBJECT CORRESPONDENCE CHECK  (is HA_n the same person as HB_n?)")
    print("=" * 74)
    fm16 = band_mask(freqs, 200.0, 16000.0)
    n = len(ids)
    Xm = np.stack([np.concatenate(M[i], axis=1) for i in ids])   # (n,129,786)
    Xe = np.stack([np.concatenate(E[i], axis=1) for i in ids])
    C = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            D = (Xm[a] - Xe[b])[fm16, :]
            C[a, b] = np.sqrt(np.mean((D - D.mean()) ** 2))
    diag = np.diag(C)
    off = C[~np.eye(n, dtype=bool)]
    rank = (C < diag[:, None]).sum(axis=1)          # how many others beat the true match
    print(f"  gain-aligned LSD, correct pairing : {diag.mean():.3f} dB (median {np.median(diag):.3f})")
    print(f"  gain-aligned LSD, mismatched      : {off.mean():.3f} dB")
    print(f"  separation                        : {(off.mean()-diag.mean()):.3f} dB "
          f"({(off.mean()-diag.mean())/off.std():.2f} sigma)")
    print(f"  true match is argmin for          : {(rank==0).sum()}/{n} subjects "
          f"(mean rank {rank.mean():.2f})")
    # chance level for "true match is argmin" is 1/n
    print(f"  chance level for argmin            : {100.0/n:.1f}%  "
          f"(observed {100.0*(rank==0).sum()/n:.1f}%)")
    if (rank == 0).sum() > 5 * (n / n):   # >5x chance count
        print("  => ordering HA_n <-> HB_n CONFIRMED (far above chance).")
        print("     BUT the margin is small (~1 sigma): BEM only weakly preserves")
        print("     individual identity relative to the measured HRTF. See section 5.")
    else:
        print("  => WARNING: ordering NOT confirmed.")

    # ------------------------------------------------------------ main numbers
    print("\n" + "=" * 74)
    print("2. BEM vs MEASURED  (head+ears mesh, HE16kHz)")
    print("=" * 74)
    bands = [("full 0.2-16 kHz", 200.0, 16000.0),
             ("low  0.2-5  kHz", 200.0, 5000.0),
             ("notch 6-12  kHz", NOTCH_BAND[0], NOTCH_BAND[1]),
             ("high 12-16  kHz", 12000.0, 16000.0)]
    res_he = {}
    for name, lo, hi in bands:
        fm = band_mask(freqs, lo, hi)
        rows = [distortions(M[i][e], E[i][e], fm) for i in ids for e in (0, 1)]
        agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        res_he[name] = agg
        print(f"  {name:16s}  LSD_raw={agg['raw_rms']:5.2f}  "
              f"LSD_gain={agg['gain_rms']:5.2f}  LSD_perfreq={agg['perfreq_rms']:5.2f}  "
              f"| SD_L1_gain={agg['gain_l1']:5.2f} dB   (offset {agg['offset_db']:+.1f} dB)")

    print("\n" + "=" * 74)
    print("3. TORSO ABLATION  (only valid <= 5 kHz, where HTE is trusted)")
    print("=" * 74)
    fm5 = band_mask(freqs, 200.0, 5000.0)
    rows_e = [distortions(M[i][e], E[i][e], fm5) for i in ids for e in (0, 1)]
    rows_t = [distortions(M[i][e], T[i][e], fm5) for i in ids for e in (0, 1)]
    le = np.mean([r["gain_rms"] for r in rows_e])
    lt = np.mean([r["gain_rms"] for r in rows_t])
    print(f"  head+ears        (no torso) vs measured : {le:5.2f} dB")
    print(f"  head+torso+ears             vs measured : {lt:5.2f} dB")
    print(f"  improvement from adding the torso       : {le-lt:+5.2f} dB "
          f"({100*(le-lt)/le:+.1f}%)")

    # -------------------------------------------------------------- references
    print("\n" + "=" * 74)
    print("4. REFERENCE POINTS")
    print("=" * 74)
    fm = band_mask(freqs, 200.0, 16000.0)
    # mean-HRTF baseline: how far is the population mean from each individual?
    allm = np.stack([M[i][e] for i in ids for e in (0, 1)])       # (2n,129,393)
    mean_hrtf = allm.mean(axis=0)
    rows = [distortions(allm[j], mean_hrtf, fm) for j in range(allm.shape[0])]
    print(f"  mean-HRTF baseline (measured)           : "
          f"{np.mean([r['gain_rms'] for r in rows]):5.2f} dB")
    print(f"  BEM (HE16kHz)                           : {res_he['full 0.2-16 kHz']['gain_rms']:5.2f} dB")
    print(f"  Zhang et al. 2021 on HUTUBS (SD, L1)    :  7.23 dB   <- their metric is L1")
    print(f"  ours in the same L1 metric              : {res_he['full 0.2-16 kHz']['gain_l1']:5.2f} dB")

    # ------------------------------------------------- does BEM carry the individual?
    # THE decisive question for a BEM-targeted surrogate: are BEM's *individual
    # deviations from the population mean* the same deviations the measurement
    # shows?  If not, no surrogate trained on BEM can personalize, however
    # perfectly it reproduces BEM.
    print("\n" + "=" * 74)
    print("5. DOES BEM CAPTURE THE RIGHT INDIVIDUAL VARIATION?")
    print("=" * 74)
    alle = np.stack([E[i][e] for i in ids for e in (0, 1)])
    dm = allm - allm.mean(axis=0)                    # measured deviation from mean
    de = alle - alle.mean(axis=0)                    # BEM deviation from mean
    for name, lo, hi in [("0.2-16 kHz", 200.0, 16000.0),
                         ("0.2-5  kHz", 200.0, 5000.0),
                         ("6-12   kHz", NOTCH_BAND[0], NOTCH_BAND[1])]:
        fmb = band_mask(freqs, lo, hi)
        a = dm[:, fmb, :].reshape(dm.shape[0], -1)
        b = de[:, fmb, :].reshape(de.shape[0], -1)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        r = np.array([np.corrcoef(a[j], b[j])[0, 1] for j in range(a.shape[0])])
        # variance of the measured individual deviation that BEM explains
        expl = 1.0 - np.sum((a - b) ** 2) / np.sum(a ** 2)
        print(f"  {name}:  corr(dev_measured, dev_BEM) = {r.mean():+.3f} "
              f"+/- {r.std():.3f}   |  R^2 of BEM as a predictor = {expl:+.3f}")
    print("  (corr ~ 0 means BEM's between-subject differences are unrelated to the")
    print("   real ones -> a perfect BEM surrogate could not personalize.)")

    # ------------------------------- how much of the gap is a subject-independent bias?
    print("\n" + "=" * 74)
    print("5b. DECOMPOSING THE GAP: systematic bias vs per-subject error")
    print("=" * 74)
    for name, lo, hi in [("0.2-16 kHz", 200.0, 16000.0),
                         ("0.2-5  kHz", 200.0, 5000.0),
                         ("6-12   kHz", NOTCH_BAND[0], NOTCH_BAND[1])]:
        fmb = band_mask(freqs, lo, hi)
        D = (allm - alle)[:, fmb, :]                 # (2n, nf, ndir) dB difference
        D = D - D.mean(axis=(1, 2), keepdims=True)   # scalar gain aligned per ear
        tot = np.sqrt(np.mean(D ** 2))
        bias = D.mean(axis=0)                        # systematic map B(f, dir)
        resid = D - bias
        print(f"  {name}:  total {tot:5.2f} dB  =  systematic bias {np.sqrt(np.mean(bias**2)):5.2f} dB"
              f"  (+)  per-subject residual {np.sqrt(np.mean(resid**2)):5.2f} dB")
    print("  The systematic part is one fixed (freq x direction) map shared by all")
    print("  subjects -> removable by a calibration/residual step. The per-subject")
    print("  residual is the part no fixed correction can fix.")

    # ------------------------------------------------- ipsi vs contra breakdown
    print("\n" + "=" * 74)
    print("6. DIRECTIONAL BREAKDOWN (left ear, 0.2-16 kHz, gain-aligned)")
    print("=" * 74)
    az = sio.loadmat(os.path.join(ROOT, "Parameters/azim.mat"))["azim"].ravel()
    el = sio.loadmat(os.path.join(ROOT, "Parameters/elev.mat"))["elev"].ravel()
    # Matlab spherical convention; left ear is ipsilateral for azim in (0, pi)
    ipsi = (np.sin(az) > 0.35)
    contra = (np.sin(az) < -0.35)
    for lbl, sel in [("ipsilateral", ipsi), ("contralateral", contra),
                     ("median plane", np.abs(np.sin(az)) < 0.2)]:
        vals = []
        for i in ids:
            D = (M[i][0] - E[i][0])[fm][:, sel]
            vals.append(np.sqrt(np.mean((D - D.mean()) ** 2)))
        print(f"  {lbl:15s} ({sel.sum():3d} dirs): {np.mean(vals):5.2f} dB")

    # ------------------------------------------------------------------ figures
    fm16 = band_mask(freqs, 200.0, 16000.0)
    fm5 = band_mask(freqs, 200.0, 5000.0)
    fq_he = np.sqrt(np.mean([per_freq_curve(M[i][e], E[i][e], fm16) ** 2
                             for i in ids for e in (0, 1)], axis=0))
    # For the torso comparison both curves must use the SAME alignment band,
    # otherwise the two lines are not comparable.
    fq_te = np.sqrt(np.mean([per_freq_curve(M[i][e], T[i][e], fm5) ** 2
                             for i in ids for e in (0, 1)], axis=0))
    fq_he5 = np.sqrt(np.mean([per_freq_curve(M[i][e], E[i][e], fm5) ** 2
                              for i in ids for e in (0, 1)], axis=0))
    fq_mean = np.sqrt(np.mean([per_freq_curve(allm[j], mean_hrtf, fm16) ** 2
                               for j in range(allm.shape[0])], axis=0))
    # per-subject residual after removing the shared systematic bias map
    Dall = allm - alle
    Dall = Dall - Dall[:, fm16, :].mean(axis=(1, 2), keepdims=True)
    fq_resid = np.sqrt(np.mean((Dall - Dall.mean(axis=0)) ** 2, axis=(0, 2)))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    v = freqs <= 16000
    ax[0].plot(freqs[v] / 1000, fq_he[v], lw=2, label="BEM head+ears (HE16kHz)")
    v5 = freqs <= 5000
    ax[0].plot(freqs[v5] / 1000, fq_he5[v5], lw=1.2, ls=":", color="tab:blue",
               label="  HE, re-aligned in 0.2-5 kHz (for torso comparison)")
    ax[0].plot(freqs[v5] / 1000, fq_te[v5], lw=2, ls="--", label="BEM head+torso+ears (HTE05kHz)")
    ax[0].plot(freqs[v] / 1000, fq_resid[v], lw=2, color="seagreen",
               label="BEM after removing shared bias map")
    ax[0].plot(freqs[v] / 1000, fq_mean[v], lw=1.5, color="gray", label="mean-HRTF baseline")
    ax[0].axvspan(6, 12, color="orange", alpha=0.12)
    ax[0].text(9, ax[0].get_ylim()[1] * 0.05, "notch band", ha="center", fontsize=8, color="darkorange")
    ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("RMS dB error (gain-aligned)")
    ax[0].set_title("SYMARE: BEM vs measured, per frequency")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    d_he = np.array([distortions(M[i][e], E[i][e], fm)["gain_rms"] for i in ids for e in (0, 1)])
    ax[1].hist(d_he, bins=28, color="steelblue", alpha=0.85)
    ax[1].axvline(d_he.mean(), color="k", ls="--", label=f"mean {d_he.mean():.2f} dB")
    ax[1].set_xlabel("gain-aligned LSD (dB), 0.2-16 kHz")
    ax[1].set_ylabel("# ears"); ax[1].set_title("Per-ear spread (61 subjects x 2 ears)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "bem_vs_measured.png"), dpi=150)
    print(f"\nfigure -> {os.path.join(OUT, 'bem_vs_measured.png')}")

    np.savez(os.path.join(OUT, "bem_vs_measured.npz"),
             freqs=freqs, fq_he=fq_he, fq_te=fq_te, fq_mean=fq_mean,
             per_ear_lsd_he=d_he, confusion=C, ids=np.array(ids))
    print(f"data   -> {os.path.join(OUT, 'bem_vs_measured.npz')}")


if __name__ == "__main__":
    main()
