#!/usr/bin/env python3
"""
CARLab hrirfilt (48 kHz) -> 44.1 kHz -> HEALPix npy

为什么要做：HUTUBS / SONICOM 已经按 44.1 kHz 处理好了。同一个 bin 索引在
两个采样率下对应的物理频率不同：

    48   kHz / 256 点 : df = 187.50 Hz,  bin 64 = 12000 Hz
    44.1 kHz / 256 点 : df = 172.27 Hz,  bin 64 = 11025 Hz

notch 带附近错位接近 1 kHz —— 和一个 notch 本身的宽度同量级。不对齐就混着训，
等于把最关心的特征抹平。

做法：在【时域】用多相重采样 (resample_poly 147/160)，而不是在幅度谱上插值。
      前者自带抗混叠且是精确的；后者会在 notch 处产生平滑损失。

输出与 48 kHz 版格式完全一致：(129, 3072) float32, dB, HEALPix nside=16。
每个目录附 meta.json 记录采样率等参数，供 notebook 的检查 cell 读取。
"""
import os
import sys
import json
import collections
import numpy as np
import scipy.io as sio
import healpy as hp
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # spherical_spline.py sits beside this file
from spherical_spline import build_interpolation_matrix          # noqa: E402
from _paths import symare_root                                # noqa: E402

ROOT = symare_root()
DB = f"{ROOT}/New_hrtfs/CARLabHrirDataBaseFixedByPierre"
OUT_L = f"{ROOT}/CARLab_44k1_left"
OUT_R = f"{ROOT}/CARLab_44k1_right"

REC = 2052
FS_SRC, FS_DST = 48000, 44100          # 44100 = 48000 * 147/160
UP, DOWN = 147, 160
NFFT, NBINS, NSIDE = 256, 129, 16


def read_hrirfilt(path):
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), "<i2")[0])
    body = b[2:].reshape(-1, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(-1, 2, L)
    return np.radians(-pos[:, 0]), np.radians(pos[:, 1]), sig      # az 取反


def to_healpix_44k1(h48, W):
    """h48: (Ndir, L) @48 kHz -> (129, npix) dB @44.1 kHz 的频率栅格"""
    h44 = resample_poly(h48, UP, DOWN, axis=1)        # 256 -> 236 样本
    H = np.fft.rfft(h44, n=NFFT, axis=1)[:, :NBINS]   # 零填充到 256 再取 129 bin
    return (20 * np.log10(np.abs(H).T + 1e-9) @ W.T).astype(np.float32)


def main():
    os.makedirs(OUT_L, exist_ok=True)
    os.makedirs(OUT_R, exist_ok=True)

    # 复用 48 kHz 版建立的编号（CARLab_subjects.json）
    man = json.load(open(f"{ROOT}/CARLab_subjects.json"))
    print(f"受试者 {len(man)}   {FS_SRC} Hz -> {FS_DST} Hz  (resample_poly {UP}/{DOWN})")
    print(f"频率步进 {FS_SRC/NFFT:.2f} Hz -> {FS_DST/NFFT:.2f} Hz\n")

    xyz_hp = np.column_stack(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    Wcache = {}

    for n, r in enumerate(man, 1):
        az, el, sig = read_hrirfilt(f"{DB}/hrirfilt{r['carlab_idx']}")
        gk = (len(az), round(float(az[0]), 6), round(float(el[0]), 6))
        if gk not in Wcache:
            xyz = np.column_stack((np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)))
            print(f'  [建 STPS 矩阵] {len(az)} 方向 ...', flush=True)
            Wcache[gk] = build_interpolation_matrix(xyz, xyz_hp,
                                                    spline_order=2, smoothing=1e-5)
        W = Wcache[gk]
        np.save(f"{OUT_L}/{r['id']}_left_full.npy",  to_healpix_44k1(sig[:, 0, :], W))
        np.save(f"{OUT_R}/{r['id']}_right_full.npy", to_healpix_44k1(sig[:, 1, :], W))
        if n % 50 == 0:
            print(f'  {n}/{len(man)}', flush=True)

    # ---------------- meta.json ----------------
    meta = {
        "dataset": "CARLab",
        "sample_rate": FS_DST,
        "nfft": NFFT,
        "n_bins": NBINS,
        "freq_step_hz": FS_DST / NFFT,
        "freq_max_hz": FS_DST / 2,
        "nside": NSIDE,
        "npix": hp.nside2npix(NSIDE),
        "units": "dB magnitude",
        "n_subjects": len(man),
        "provenance": (f"hrirfilt (raw, {FS_SRC} Hz) -> resample_poly({UP}/{DOWN}) "
                       f"-> rfft(n={NFFT}) -> 20log10 -> STPS to HEALPix"),
        "recommended_band_hz": [500, 16000],
        "notes": ("Raw CARLab pipeline (no ProcessHRIRs.m post-processing). "
                  "Head alignment already applied in hrirfilt ('FixedByPierre'). "
                  "Do not mix with HA*.mat-derived data: they differ by ~3.6 dB."),
    }
    for d, side in ((OUT_L, "left"), (OUT_R, "right")):
        json.dump({**meta, "ear": side}, open(f"{d}/meta.json", "w"), indent=1)

    # 给 48 kHz 版也补一份 meta
    meta48 = {**meta, "sample_rate": FS_SRC, "freq_step_hz": FS_SRC / NFFT,
              "freq_max_hz": FS_SRC / 2,
              "provenance": f"hrirfilt (raw, {FS_SRC} Hz) -> rfft(n={NFFT}) -> 20log10 -> STPS"}
    for d, side in ((f"{ROOT}/CARLab_left", "left"), (f"{ROOT}/CARLab_right", "right")):
        if os.path.isdir(d):
            json.dump({**meta48, "ear": side}, open(f"{d}/meta.json", "w"), indent=1)

    print(f"\n完成 {len(man)} 人")
    print(f"  {OUT_L}\n  {OUT_R}\n  两目录均含 meta.json")

    # ---------------- 自检 ----------------
    # 换采样率必然带来一个【常数】增益差 20log10(fs_src/fs_dst) = +0.736 dB
    # （DFT 与连续谱的换算 X(f) ~ DFT/fs）。谱形不受影响，所以自检要先去掉它。
    # 不修正这个常数：HUTUBS/SONICOM 本来就是 44.1 kHz 走同一套 20log10|rfft|，
    # 与本批采用同一约定；且常数 dB 偏移会被 FiLM 的 beta / 电平归一化吸收。
    a48 = np.load(f"{ROOT}/CARLab_left/{man[0]['id']}_left_full.npy")
    a44 = np.load(f"{OUT_L}/{man[0]['id']}_left_full.npy")
    f48 = np.arange(NBINS) * FS_SRC / NFFT
    f44 = np.arange(NBINS) * FS_DST / NFFT
    v = (f44 >= 500) & (f44 <= 16000)
    a48_on44 = np.stack([np.interp(f44, f48, a48[:, p]) for p in range(0, 3072, 64)], 1)
    d = a48_on44[v] - a44[v][:, ::64]
    print(f"\n自检 ({man[0]['id']}, 500-16 kHz):")
    print(f"  常数增益差      : {d.mean():+.3f} dB  (理论 "
          f"{20*np.log10(FS_SRC/FS_DST):+.3f} dB)")
    print(f"  去掉常数后 RMS  : {np.sqrt(((d - d.mean()) ** 2).mean()):.3f} dB"
          f"   <- 谱形误差, 应 < 0.4 dB (含检查用线性插值自身的 ~0.33 dB)")
    print(f"  按 bin 索引直接比(错误做法): "
          f"{np.sqrt(((a48[v] - a44[v]) ** 2).mean()):.3f} dB  <- 不对齐的代价")
