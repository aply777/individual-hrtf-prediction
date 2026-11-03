#!/usr/bin/env python3
"""
HRTF 的本征维度：直接对数据做 SVD，不训练任何网络。

回答的问题：
  "32 维够不够描述一个人的 HRTF？notch 带需要更多吗？"

方法：
  行 = 一只耳朵（61 subject x 2 = 122），列 = 该耳朵摊平的 log-magnitude HRTF。
  去掉人群均值（= PCA），SVD，截断到 rank R，看重建误差。

两个关键设计：
  1) 报 **LSD vs rank**，不只报"方差解释率"。notch 是低方差、高感知重要性的
     结构，方差论会系统性高估"够用的维度"。
  2) 同时给 **in-sample** 和 **留出耳朵** 两条曲线。122 个样本、5 万个特征，
     in-sample 必然过于乐观；留出曲线才是"一个模型真正需要多少维"。
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import symare_root, here                          # noqa: E402

ROOT = os.path.join(symare_root(), "HRIRs")
OUT = here()
os.makedirs(OUT, exist_ok=True)

NFFT, FS, EPS = 256, 48000.0, 1e-12
BANDS = [("full 0.2-16k", 200., 16000.),
         ("low  0.2-5k", 200., 5000.),
         ("notch 6-12k", 6000., 12000.),
         ("high 12-16k", 12000., 16000.)]


def load_ears():
    """-> X list of (129,393) dB arrays, one per ear."""
    ears = []
    for i in range(1, 62):
        p = os.path.join(ROOT, "Acoustic", f"HA{i:02d}.mat")
        if not os.path.exists(p):
            continue
        d = sio.loadmat(p)
        for key in ("hL", "hR"):
            H = np.fft.rfft(d[key].astype(float), n=NFFT, axis=0)
            ears.append(20 * np.log10(np.abs(H) + EPS))
    return np.stack(ears)            # (n_ears, 129, 393)


def lsd(A, B):
    """A,B in dB, same shape -> RMS dB difference."""
    return float(np.sqrt(np.mean((A - B) ** 2)))


def rank_curves(X, ranks, n_folds=5, seed=0):
    """X: (n, p) already in dB, rows = ears.

    Returns (var_explained, lsd_in, lsd_out) each indexed like `ranks`.
    lsd_out uses k-fold: basis built on train ears only, evaluated on held-out
    ears by projecting them onto that basis.
    """
    n = X.shape[0]
    mu = X.mean(0)
    Xc = X - mu

    # ---- in-sample ----
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = np.cumsum(S ** 2) / np.sum(S ** 2)
    var_explained = np.array([var[min(r, len(S)) - 1] for r in ranks])
    lsd_in = []
    for r in ranks:
        Xr = (U[:, :r] * S[:r]) @ Vt[:r]
        lsd_in.append(lsd(Xr + mu, X))

    # ---- held-out ----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    err = np.zeros((n_folds, len(ranks)))
    for f, te in enumerate(folds):
        tr = np.setdiff1d(idx, te)
        mu_tr = X[tr].mean(0)
        _, _, Vt_tr = np.linalg.svd(X[tr] - mu_tr, full_matrices=False)
        Xte_c = X[te] - mu_tr
        for j, r in enumerate(ranks):
            r_eff = min(r, Vt_tr.shape[0])
            B = Vt_tr[:r_eff]                       # (r, p) orthonormal rows
            Xte_r = (Xte_c @ B.T) @ B               # project & reconstruct
            err[f, j] = lsd(Xte_r + mu_tr, X[te])
    return var_explained, np.array(lsd_in), err.mean(0)


def main():
    freqs = np.fft.rfftfreq(NFFT, 1 / FS)
    E = load_ears()
    n = E.shape[0]
    print(f"ears: {n}  (max achievable rank after centering = {n-1})\n")

    ranks = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 110, 121])
    ranks = ranks[ranks <= n - 1]

    results = {}
    print(f"{'band':<14}{'rank':>6}{'var%':>8}{'LSD_in':>9}{'LSD_heldout':>13}")
    print("-" * 52)
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs <= hi)
        X = E[:, m, :].reshape(n, -1)
        v, li, lo_ = rank_curves(X, ranks)
        results[name] = (v, li, lo_)
        base = lsd(X, np.tile(X.mean(0), (n, 1)))     # mean-HRTF baseline
        print(f"{name:<14}{'(mean)':>6}{0:>8.1f}{base:>9.2f}{base:>13.2f}")
        for j, r in enumerate(ranks):
            if r in (8, 16, 32, 64, 121):
                print(f"{'':<14}{r:>6}{100*v[j]:>8.1f}{li[j]:>9.2f}{lo_[j]:>13.2f}")
        print()

    # ---------- 每个频点单独看需要多少维 ----------
    print("per-frequency effective rank (held-out LSD within 1 dB of its own floor)")
    eff = []
    for fi in range(len(freqs)):
        if freqs[fi] > 16000 or freqs[fi] < 200:
            eff.append(np.nan); continue
        X = E[:, fi, :]
        _, _, lo_ = rank_curves(X, ranks, n_folds=5)
        floor = lo_.min()
        ok = np.where(lo_ <= floor + 1.0)[0]
        eff.append(ranks[ok[0]] if len(ok) else np.nan)
    eff = np.array(eff, dtype=float)
    for f0, f1 in [(200, 2000), (2000, 5000), (6000, 12000), (12000, 16000)]:
        m = (freqs >= f0) & (freqs <= f1)
        print(f"  {f0/1000:>5.1f}-{f1/1000:<5.1f} kHz : median {np.nanmedian(eff[m]):.0f} dims")

    # ---------- 决定性实验：维度不够，还是人不够？ ----------
    # 固定 rank，改变用来建基的耳朵数量。若曲线在 N=100 处仍在下降，
    # 说明瓶颈是样本量而不是维度 —— 那么 CARLab 那 248 人才是解药。
    print("\nlearning curve: held-out LSD vs #training ears (notch band 6-12 kHz)")
    m = (freqs >= 6000) & (freqs <= 12000)
    X = E[:, m, :].reshape(n, -1)
    rng = np.random.default_rng(1)
    lc = {}
    for R in (8, 16, 32):
        row = []
        for n_tr in (20, 40, 60, 80, 100):
            errs = []
            for rep in range(8):
                perm = rng.permutation(n)
                tr, te = perm[:n_tr], perm[n_tr:n_tr + 20]
                mu_tr = X[tr].mean(0)
                _, _, Vt = np.linalg.svd(X[tr] - mu_tr, full_matrices=False)
                B = Vt[:min(R, Vt.shape[0])]
                Xr = ((X[te] - mu_tr) @ B.T) @ B + mu_tr
                errs.append(lsd(Xr, X[te]))
            row.append(np.mean(errs))
        lc[R] = row
        print(f"  rank {R:>3}: " + "  ".join(f"N={a}:{b:.2f}" for a, b in
                                            zip((20, 40, 60, 80, 100), row)))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    for name, (v, li, lo_) in results.items():
        ax[0].plot(ranks, 100 * v, marker="o", ms=3, label=name)
        ax[1].plot(ranks, lo_, marker="o", ms=3, label=name)
    ax[0].axhline(90, color="gray", ls=":"); ax[0].axhline(95, color="gray", ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlabel("rank R"); ax[0].set_ylabel("variance explained (%)")
    ax[0].set_title("(a) variance explained (misleading)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].set_xscale("log"); ax[1].set_xlabel("rank R"); ax[1].set_ylabel("held-out LSD (dB)")
    ax[1].set_title("(b) held-out reconstruction LSD (what matters)")
    ax[1].axvline(32, color="crimson", ls="--", lw=1)
    ax[1].text(33, ax[1].get_ylim()[1]*0.9, "z=32", color="crimson", fontsize=8)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    v = (freqs >= 200) & (freqs <= 16000)
    ax[2].plot(freqs[v]/1000, eff[v], lw=2, color="darkgreen")
    ax[2].axvspan(6, 12, color="orange", alpha=.12)
    ax[2].set_xlabel("frequency (kHz)"); ax[2].set_ylabel("dims needed")
    ax[2].set_title("(c) dims needed, per frequency")
    ax[2].grid(alpha=.3)
    for R, row in lc.items():
        ax[3].plot((20, 40, 60, 80, 100), row, marker="o", label=f"rank {R}")
    ax[3].set_xlabel("# training ears"); ax[3].set_ylabel("held-out LSD (dB)")
    ax[3].set_title("(d) more dims, or more people?  [6-12 kHz]")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "hrtf_svd_rank.png"), dpi=150)
    np.savez(os.path.join(OUT, "hrtf_svd_rank.npz"), ranks=ranks, eff=eff,
             freqs=freqs, **{k.replace(" ", "_"): np.array(v) for k, v in results.items()})
    print(f"\nfigure -> {OUT}/hrtf_svd_rank.png")


if __name__ == "__main__":
    main()
