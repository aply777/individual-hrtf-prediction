#!/usr/bin/env python3
"""
CARLab hrirfilt -> HEALPix npy，与 spherical_spline.py 的输出格式完全一致。

为什么可以直接这么做（今天查证的结论）：
  * hrirfilt 已经是 48 kHz / 256 点 / 393 方向，栅格与 SYMARE 逐一相同（角距 0.00 deg）
  * hrirfilt 【已做过头部位置与朝向对齐】(重解后头心偏移 0.7-1.2 mm, 旋转 ±0.1 deg)
    —— 这就是 "FixedByPierre" 的含义。所以 raw 里没有姿态 nuisance。
  * 耳序已验证: ear0 = 左, ear1 = 右 (幅度谱 corr 0.969 / 0.971)
  * 方位角需取反 (与 ProcessHRIRs.m 的 sphCoo(:,1) = -sphCoo(:,1) 一致)

为什么不去复现 HA*.mat 的处理：
  造 HA01-61 的脚本不在仓库里（ProcessHRIRs.m 产出的是 HRIR_ANL*，
  ProcessHRIRsSymare.m 是【读】HA 往下转格式）。无法精确复现。
  两者差 3.6 dB，且实测 ILD 不符合近场移除的特征，成因无法追溯。
  -> 正确做法是全部走同一条 raw 管线，包括与 SYMARE 重合的那 56 人。

输出与旧管线一致: (129, 3072) float32, dB 幅度, HEALPix nside=16
"""
import os
import re
import sys
import json
import collections
import numpy as np
import scipy.io as sio
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # spherical_spline.py sits beside this file
from spherical_spline import build_interpolation_matrix          # noqa: E402
from _paths import symare_root                                # noqa: E402

ROOT = symare_root()
DB = f"{ROOT}/New_hrtfs/CARLabHrirDataBaseFixedByPierre"
OUT_L = f"{ROOT}/carlab_npy_left"
OUT_R = f"{ROOT}/carlab_npy_right"
REC, FS, NSIDE = 2052, 48000, 16
MIN_DIR = 300
EXCLUDE = {"hats", "kemar"}          # 假头, 非人


def read_hrirfilt(path):
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), "<i2")[0])
    body = b[2:].reshape(-1, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(-1, 2, L)
    az = np.radians(-pos[:, 0])                       # ★ 取反
    el = np.radians(pos[:, 1])
    return az, el, sig


def to_healpix(h, W):
    """h: (Ndir, L) -> (129, npix) dB。与 spherical_spline.py 的算法一致"""
    H = np.fft.fft(h.T, axis=0)[:129, :]              # (129, Ndir)
    return (20 * np.log10(np.abs(H) + 1e-9) @ W.T).astype(np.float32)


def main():
    os.makedirs(OUT_L, exist_ok=True)
    os.makedirs(OUT_R, exist_ok=True)
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())

    # ---- 每个人挑方向最多的那次测量（并列取最新，同 ProcessHRIRs.m 的 max(idx)）----
    d = sio.loadmat(f"{ROOT}/New_hrtfs/carlabdb.mat")["carlab_hrir"][0]
    cand = collections.defaultdict(list)
    for i, e in enumerate(d, 1):
        nm = np.array(e["username"]).ravel()
        nm = str(nm[0]).strip() if nm.size else ""
        key = norm(nm)
        p = f"{DB}/hrirfilt{i}"
        if not key or key in EXCLUDE or not os.path.exists(p):
            continue
        n = (os.path.getsize(p) - 2) // REC
        if n >= MIN_DIR:
            cand[key].append((n, i, nm))
    subs = {k: max(v) for k, v in cand.items()}
    print(f"受试者: {len(subs)}   (来自 {sum(len(v) for v in cand.values())} 个文件)")
    print(f"方向栅格: {dict(collections.Counter(v[0] for v in subs.values()))}\n")

    xyz_hp = np.column_stack(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    Wcache, manifest = {}, []

    for n_done, (key, (ndir, idx, nm)) in enumerate(sorted(subs.items()), 1):
        az, el, sig = read_hrirfilt(f"{DB}/hrirfilt{idx}")
        gk = (ndir, round(float(az[0]), 6), round(float(el[0]), 6))
        if gk not in Wcache:
            xyz = np.column_stack((np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)))
            print(f"  [建 STPS 矩阵] {ndir} 方向 ...", flush=True)
            Wcache[gk] = build_interpolation_matrix(xyz, xyz_hp,
                                                    spline_order=2, smoothing=1e-5)
        W = Wcache[gk]
        tag = f"CL_{key[:24]}"
        np.save(f"{OUT_L}/{tag}_left_full.npy", to_healpix(sig[:, 0, :], W))
        np.save(f"{OUT_R}/{tag}_right_full.npy", to_healpix(sig[:, 1, :], W))
        manifest.append(dict(tag=tag, name=nm, carlab_idx=int(idx), ndir=int(ndir)))
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(subs)}", flush=True)

    with open(f"{ROOT}/carlab_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=1, ensure_ascii=False)

    print(f"\n完成 {len(manifest)} 人")
    print(f"  {OUT_L}")
    print(f"  {OUT_R}")
    print(f"  {ROOT}/carlab_manifest.json   (tag -> 真实姓名 / carlabdb 索引)")
    x = np.load(f"{OUT_L}/{manifest[0]['tag']}_left_full.npy")
    print(f"\n样例 {manifest[0]['tag']}: shape={x.shape} dtype={x.dtype} "
          f"range=[{x.min():.1f}, {x.max():.1f}] dB")
    print("\n★ 与旧数据混用前请注意：本批为 raw 管线，与 HA*.mat 相差约 3.6 dB")
    print("  (500 Hz-16 kHz, 去全局增益)。建议 SYMARE 那 56 人也从本批取，只用一条管线。")
    print("★ 建议训练频段限制在 500 Hz - 16 kHz：")
    print("  <500 Hz 实测低频不可靠；>16 kHz 原始数据滚降到 -42 dB (测量系统上限)。")


if __name__ == "__main__":
    main()
