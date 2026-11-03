#!/usr/bin/env python3
"""
头部位置/朝向校正 —— ProcessCARLab_Hrirs.m 的 Python 移植 + 闭环验证

为什么要做：受试者在测量装置里没坐正 / 没对准，会给每个人的 HRTF 烙上一个
            与解剖无关的随机旋转平移。这直接污染 geometry -> HRTF 的映射。
            42 个人同时有 raw 和 processed，可以量化"我们做到多干净"。

与 MATLAB 原版的两处必要偏离：
  1) hrirfilt 的【绝对】飞行时间被裁掉了约 45 样本（峰值中位 92, 理论 137）。
     原版的三边定位用平方距离差, 对常数偏移不是不变的。
     -> 这里把偏移当第 4 个未知量一起解（GPS 钟差问题）, 393 方程解 4 未知数。
  2) 原版用 resample(...,10,1) 做 10x 上采样再互相关求峰。
     -> 这里在频域对互相关做零填充上采样（等价, 快得多）。
"""
import os
import sys
import re
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # spherical_spline.py sits beside this file
from spherical_spline import build_interpolation_matrix          # noqa: E402
from _paths import symare_root                                # noqa: E402

ROOT = symare_root()
DB = f"{ROOT}/New_hrtfs/CARLabHrirDataBaseFixedByPierre"
REC, FS, C = 2052, 48000.0, 343.0
SPK_DIST = 0.98
UP = 10                                    # 互相关上采样倍数


# ---------------------------------------------------------------- I/O
def read_raw(idx):
    b = np.fromfile(f"{DB}/hrirfilt{idx}", dtype=np.uint8)
    body = b[2:].reshape(-1, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(-1, 2, 256)
    return np.radians(-pos[:, 0]), np.radians(pos[:, 1]), sig      # az 取反


def sph2cart(az, el, r=1.0):
    return np.column_stack((r * np.cos(el) * np.cos(az),
                            r * np.cos(el) * np.sin(az),
                            r * np.sin(el)))


# ------------------------------------------------------- 步骤 2: 纯延迟
def min_phase(h, nfft=1024):
    """实倒谱法求最小相位版本。h: (N, L) 沿 axis=0"""
    H = np.fft.rfft(h, n=nfft, axis=0)
    logmag = np.log(np.abs(H) + 1e-12)
    full = np.concatenate([logmag, logmag[-2:0:-1]], axis=0)
    c = np.fft.ifft(full, axis=0).real
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:nfft // 2] = 2.0
    w[nfft // 2] = 1.0
    cmin = c * w[:, None]
    return np.fft.irfft(np.exp(np.fft.rfft(cmin, axis=0)), n=nfft, axis=0)


def excess_delay(h, bp):
    """h: (L, Ndir) -> 每个方向相对其最小相位版本的纯延迟 (秒)"""
    L, N = h.shape
    hm = min_phase(h)[:L * 2]
    a = fftconvolve(np.pad(h, ((0, len(bp)), (0, 0))), bp[:, None], axes=0)
    b = fftconvolve(np.pad(hm, ((0, len(bp)), (0, 0))), bp[:, None], axes=0)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    R = np.fft.rfft(a, n, axis=0) * np.conj(np.fft.rfft(b, n, axis=0))
    # 频域零填充 = 时域 UP 倍上采样
    Rp = np.zeros((n // 2 * UP + 1, N), complex)
    Rp[: R.shape[0]] = R
    xc = np.fft.irfft(Rp, n=n * UP, axis=0)
    k = np.argmax(np.abs(xc), axis=0)
    k = np.where(k > n * UP // 2, k - n * UP, k)
    return k / (UP * FS)


# --------------------------------------------- 步骤 3: 带钟差的三边定位
def locate(spk, tau):
    """spk:(N,3) 扬声器位置; tau:(N,) 测得延迟(秒, 含未知常数偏移)
    解 |p - s_i| = C*(tau_i - b), 未知 (p, b)。

    ★ 不能用线性化解法: u_i = C*tau_i ~ |s_i| - p·s_hat_i, 所以偏移那一列
      几乎是位置三列的线性组合 -> 严重共线, 解会跑飞(实测半头距只有 11 mm、
      yaw 翻到 -180 deg)。这里用非线性最小二乘(标准 GPS 做法)。
    """
    from scipy.optimize import least_squares

    def resid(x):
        return np.linalg.norm(spk - x[:3], axis=1) - C * (tau - x[3])

    b0 = np.median(tau) - SPK_DIST / C          # 初值: 由平均飞行时间反推
    sol = least_squares(resid, np.array([0.0, 0.0, 0.0, b0]), method="lm")
    return sol.x[:3], sol.x[3]


# ------------------------------------------------------------ 主流程
def correct(idx, verbose=False):
    az, el, sig = read_raw(idx)
    spk = sph2cart(az, el, SPK_DIST)
    bp = firwin(1001, [100 / (FS / 2), 5000 / (FS / 2)], pass_zero=False)

    tau = np.stack([excess_delay(sig[:, e, :].T, bp) for e in (0, 1)])   # (2,N)
    posL, bL = locate(spk, tau[0])
    posR, bR = locate(spk, tau[1])

    trans = (posL + posR) / 2
    v = posL - trans                                   # 指向左耳的向量
    yaw = np.arctan2(v[1], v[0]) - np.pi / 2
    roll = np.arcsin(np.clip(v[2] / (np.linalg.norm(v) + 1e-12), -1, 1))
    fwd = np.array([SPK_DIST * np.cos(yaw), SPK_DIST * np.sin(yaw), 0.0]) - trans
    pitch = np.arcsin(np.clip(-fwd[2] / (np.linalg.norm(fwd) + 1e-12), -1, 1))

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R = Rz @ Ry @ Rx

    act = (spk @ R) - trans                            # 相对头部的真实位置
    r = np.linalg.norm(act, axis=1)
    azA = np.arctan2(act[:, 1], act[:, 0])
    elA = np.arcsin(np.clip(act[:, 2] / r, -1, 1))

    if verbose:
        print(f"    头心偏移 {1000*np.linalg.norm(trans):5.1f} mm | "
              f"半头距 {1000*np.linalg.norm(v):5.1f} mm | "
              f"yaw {np.degrees(yaw):+5.1f} pitch {np.degrees(pitch):+5.1f} "
              f"roll {np.degrees(roll):+5.1f} deg | 偏移 {1e3*bL:+.2f} ms")
        print(f"    方向修正量: 中位 {np.degrees(np.median(np.abs(azA-az))):.2f} deg (az), "
              f"{np.degrees(np.median(np.abs(elA-el))):.2f} deg (el)")

    # 步骤 6: 从真实方向重采样回名义栅格
    W = build_interpolation_matrix(sph2cart(azA, elA), sph2cart(az, el),
                                   spline_order=2, smoothing=1e-5)
    H = np.abs(np.fft.rfft(sig.transpose(2, 1, 0), n=256, axis=0))       # (129,2,N)
    Hdb = 20 * np.log10(H + 1e-12)
    out = np.einsum("fen,mn->fem", Hdb, W)
    return out, dict(trans=trans, yaw=yaw, pitch=pitch, roll=roll)


# ------------------------------------------------------------ 验证
def main():
    az_t = sio.loadmat(f"{ROOT}/HRIRs/Parameters/azim.mat")["azim"].ravel()
    el_t = sio.loadmat(f"{ROOT}/HRIRs/Parameters/elev.mat")["elev"].ravel()
    f = np.fft.rfftfreq(256, 1 / FS)
    band = (f >= 500) & (f <= 16000)
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())

    ha = {i: sio.loadmat(f"{ROOT}/HRIRs/Acoustic/HA{i:02d}.mat") for i in range(1, 62)
          if os.path.exists(f"{ROOT}/HRIRs/Acoustic/HA{i:02d}.mat")}
    name2ha = {}
    for fn in os.listdir(f"{ROOT}/New_hrtfs/SymareHrirs"):
        if fn.startswith("HRIR_"):
            h = sio.loadmat(f"{ROOT}/New_hrtfs/SymareHrirs/{fn}")["hL"]
            for i, a in ha.items():
                if a["hL"].shape == h.shape and np.abs(a["hL"] - h).max() == 0:
                    name2ha[fn[5:-4].lower()] = i
                    break
    d = sio.loadmat(f"{ROOT}/New_hrtfs/carlabdb.mat")["carlab_hrir"][0]
    cl = {}
    for i, e in enumerate(d, 1):
        nm = np.array(e["username"]).ravel()
        nm = str(nm[0]).strip() if nm.size else ""
        p = f"{DB}/hrirfilt{i}"
        if nm and os.path.exists(p) and (os.path.getsize(p) - 2) // REC == 393:
            cl.setdefault(norm(nm), []).append(i)
    pairs = []
    for sur, hn in name2ha.items():
        s = norm(sur)
        for form in ([s, s[:-1]] if len(s) > 5 else [s]):
            hits = [k for k in cl if form and k.endswith(form)]
            if hits:
                break
        if len(hits) == 1:
            pairs.append((sur, hn, max(cl[hits[0]])))

    print(f"验证受试者: {len(pairs)}\n")
    before, after = [], []
    for n, (sur, hn, ci) in enumerate(pairs):
        try:
            az, el, sig = read_raw(ci)
            m = np.array([np.argmin(np.abs(((np.degrees(az) - np.degrees(a) + 180) % 360) - 180)
                                    + np.abs(np.degrees(el) - np.degrees(e)))
                          for a, e in zip(az_t, el_t)])
            P = np.stack([20 * np.log10(np.abs(np.fft.rfft(ha[hn][k], n=256, axis=0)) + 1e-12)
                          for k in ("hL", "hR")]).transpose(1, 0, 2)      # (129,2,393)
            raw = np.stack([20 * np.log10(np.abs(np.fft.rfft(sig[m, e, :].T, n=256, axis=0)) + 1e-12)
                            for e in (0, 1)]).transpose(1, 0, 2)
            cor, info = correct(ci, verbose=(n < 3))
            cor = cor[:, :, m]
            for tag, X, box in (("before", raw, before), ("after", cor, after)):
                D = P - X
                D -= D[band].mean()
                box.append(np.sqrt((D[band] ** 2).mean()))
            if n < 3:
                print(f"  {sur:<16} before {before[-1]:.2f} -> after {after[-1]:.2f} dB")
        except Exception as ex:
            print("  skip", sur, type(ex).__name__, ex)
    before, after = np.array(before), np.array(after)
    print("\n" + "=" * 62)
    print(f"n={len(before)}   (500 Hz - 16 kHz, 去全局增益)")
    print(f"  校正前 vs HA*.mat : {before.mean():.2f} dB  (中位 {np.median(before):.2f})")
    print(f"  校正后 vs HA*.mat : {after.mean():.2f} dB  (中位 {np.median(after):.2f})")
    print(f"  改善              : {before.mean()-after.mean():+.2f} dB "
          f"({100*(1-after.mean()/before.mean()):+.0f}%)")
    print(f"  变好的受试者      : {(after<before).sum()}/{len(before)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
