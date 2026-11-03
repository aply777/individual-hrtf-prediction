#!/usr/bin/env python3
"""
raw hrirfilt 与 processed HA*.mat 的差异，是"一个固定滤波器"还是"逐方向的几何校正"？

意义：56 个人同时存在于两边。若差异主要是【方向无关】的频响（= 扬声器/麦克风校准），
      就能从这 56 人估出一个校正滤波器，直接套到另外 163 人 —— 不必重跑
      ProcessCARLab_Hrirs.m 的全套管线（那里还有头部朝向校正、近场移除等）。

分解： D(ear, f, dir) = 全局增益 + 方向无关频响 C(ear,f) + 逐方向残差
"""
import os
import re
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import symare_root, here                          # noqa: E402

ROOT = symare_root()
DB = os.path.join(ROOT, "New_hrtfs/CARLabHrirDataBaseFixedByPierre")
REC, FS = 2052, 48000


def read_raw(idx):
    b = np.fromfile(os.path.join(DB, f"hrirfilt{idx}"), dtype=np.uint8)
    body = b[2:].reshape(-1, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(-1, 2, 256)
    return -pos[:, 0], pos[:, 1], sig          # az 取反


def mag(h):
    return 20 * np.log10(np.abs(np.fft.rfft(h, n=256, axis=0)) + 1e-12)


def main():
    az_t = np.degrees(sio.loadmat(f"{ROOT}/HRIRs/Parameters/azim.mat")["azim"].ravel())
    el_t = np.degrees(sio.loadmat(f"{ROOT}/HRIRs/Parameters/elev.mat")["elev"].ravel())
    f = np.fft.rfftfreq(256, 1 / FS)
    band = (f >= 500) & (f <= 16000)

    # 姓名 -> HA 编号
    ha = {i: sio.loadmat(f"{ROOT}/HRIRs/Acoustic/HA{i:02d}.mat")
          for i in range(1, 62)
          if os.path.exists(f"{ROOT}/HRIRs/Acoustic/HA{i:02d}.mat")}
    name2ha = {}
    for fn in os.listdir(f"{ROOT}/New_hrtfs/SymareHrirs"):
        if not fn.startswith("HRIR_"):
            continue
        h = sio.loadmat(f"{ROOT}/New_hrtfs/SymareHrirs/{fn}")["hL"]
        for i, a in ha.items():
            if a["hL"].shape == h.shape and np.abs(a["hL"] - h).max() == 0:
                name2ha[fn[5:-4].lower()] = i
                break

    # CARLab 索引
    d = sio.loadmat(f"{ROOT}/New_hrtfs/carlabdb.mat")["carlab_hrir"][0]
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    cl = {}
    for i, e in enumerate(d, 1):
        nm = np.array(e["username"]).ravel()
        nm = str(nm[0]).strip() if nm.size else ""
        p = os.path.join(DB, f"hrirfilt{i}")
        if nm and os.path.exists(p) and (os.path.getsize(p) - 2) // REC == 393:
            cl.setdefault(norm(nm), []).append(i)

    # 配对
    # 正确的姓氏匹配：SYMARE 用姓氏(可能带一个尾字母作区分)，CARLab 用全名
    def candidates(sur):
        s = norm(sur)
        forms = [s]
        if len(s) > 5:
            forms.append(s[:-1])          # 去掉可能的尾部区分字母
        return forms
    pairs, ambiguous = [], []
    for sur, hn in name2ha.items():
        hits = set()
        for form in candidates(sur):
            for k in cl:
                if form and k.endswith(form):     # 姓在全名末尾
                    hits.add(k)
            if hits:
                break
        if len(hits) == 1:
            pairs.append((sur, hn, max(cl[list(hits)[0]])))
        elif len(hits) > 1:
            ambiguous.append((sur, sorted(hits)))
    print(f"唯一配对: {len(pairs)}   歧义: {len(ambiguous)}   未匹配: {len(name2ha)-len(pairs)-len(ambiguous)}")
    for a in ambiguous[:5]:
        print("   歧义:", a)

    Ds = []
    for sur, hn, ci in pairs:
        try:
            azc, elc, sig = read_raw(ci)
            m = np.array([np.argmin(np.abs(((azc - a + 180) % 360) - 180) + np.abs(elc - e))
                          for a, e in zip(az_t, el_t)])
            P = np.stack([mag(ha[hn]["hL"]), mag(ha[hn]["hR"])])
            C = np.stack([mag(sig[m, 0, :].T), mag(sig[m, 1, :].T)])
            D = P - C
            D -= D[:, band, :].mean()                    # 去全局增益
            Ds.append(D)
        except Exception as ex:
            print("  skip", sur, ex)
    D = np.stack(Ds)                                     # (S,2,129,393)
    S = D.shape[0]
    print(f"成功处理: {S}\n")

    tot = np.sqrt((D[:, :, band, :] ** 2).mean())
    # 方向无关部分：对方向取平均（每人每耳一条曲线）
    C_sub = D.mean(axis=3, keepdims=True)                # (S,2,129,1)
    resid_dir = D - C_sub
    # 跨受试者共享的部分
    C_shared = C_sub.mean(axis=0, keepdims=True)         # (1,2,129,1)
    resid_sub = C_sub - C_shared

    def rms(x):
        return float(np.sqrt((x[:, :, band, :] ** 2).mean()))

    print("=" * 66)
    print("差异分解 (500 Hz - 16 kHz, RMS dB)")
    print("=" * 66)
    print(f"  总差异                                : {tot:6.2f} dB")
    print(f"  ├ 所有人共享的方向无关频响 C(ear,f)   : {rms(np.broadcast_to(C_shared, D.shape)):6.2f} dB"
          f"   <- 可用一个固定滤波器修掉")
    print(f"  ├ 逐受试者的频响偏差                  : {rms(np.broadcast_to(resid_sub, D.shape)):6.2f} dB")
    print(f"  └ 逐方向残差 (几何/近场校正)          : {rms(resid_dir):6.2f} dB"
          f"   <- 修不掉, 需跑完整管线")
    frac = rms(np.broadcast_to(C_shared, D.shape)) ** 2 / tot ** 2
    print(f"\n  共享频响占总方差: {100*frac:.0f}%")
    after = np.sqrt(rms(np.broadcast_to(resid_sub, D.shape)) ** 2 + rms(resid_dir) ** 2)
    print(f"  套用共享滤波器后剩余差异: {after:.2f} dB  (原 {tot:.2f} dB)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    v = (f >= 200) & (f <= 20000)
    for e, lb in [(0, "left"), (1, "right")]:
        ax[0].plot(f[v] / 1000, C_shared[0, e, v, 0], lw=2, label=lb)
        for s in range(min(S, 20)):
            ax[0].plot(f[v] / 1000, C_sub[s, e, v, 0], lw=.5, alpha=.3, color="gray")
    ax[0].axvspan(0.5, 16, color="green", alpha=.06)
    ax[0].set_xscale("log"); ax[0].set_xlabel("frequency (kHz)")
    ax[0].set_ylabel("processed - raw (dB)")
    ax[0].set_title("(a) direction-independent difference\n(bold = shared, grey = per subject)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    perf = np.sqrt((resid_dir ** 2).mean(axis=(0, 1, 3)))
    ax[1].plot(f[v] / 1000, perf[v], lw=2)
    ax[1].axvspan(0.5, 16, color="green", alpha=.06)
    ax[1].set_xscale("log"); ax[1].set_xlabel("frequency (kHz)")
    ax[1].set_ylabel("RMS dB"); ax[1].set_title("(b) direction-dependent residual")
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    out = here("calib_transfer.png")
    fig.savefig(out, dpi=150)
    np.savez(here("calib_filter.npz"),
             freqs=f, C_shared=C_shared[0, :, :, 0])
    print(f"\n图 -> {out}\n校正滤波器 -> calib_filter.npz  (C_shared: 2 x 129 dB)")


if __name__ == "__main__":
    main()
