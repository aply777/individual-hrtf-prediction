"""SUPERSEDED — early draft of step 2 of the CAR Lab export chain.

This was the first pass at reverse-engineering the `hrirfilt` binary format. It is kept
for provenance only. The chain that actually produced the training data is, in order:

    1. analysis/audit_carlab.py          which subjects are usable  -> 218
    2. analysis/export_carlab_npy.py     hrirfilt -> HEALPix npy at 48 kHz
    3. analysis/finalize_carlab.py       numbering CL001-CL218, mirror index
    4. analysis/export_carlab_44k1.py    48 -> 44.1 kHz; THIS is what training uses

See docs/DATA.md. Do not run this file.
"""

#!/usr/bin/env python3
"""
CAR Lab HRIR 档案 -> HEALPix npy，接到现有的 spherical_spline.py 管线。

二进制格式（反推自文件大小规律 + 字节检视，已用 SYMARE 交叉验证）：
    [int16 LE] L = 256                       滤波器长度
    N 条记录, 每条 4 + 2*L*4 = 2052 字节:
        [int16 az][int16 el]                 度
        [float32 x L] 耳0 = 左                （已验证）
        [float32 x L] 耳1 = 右                （已验证）
    N = (filesize - 2) // 2052

验证（hrirfilt82 与 SYMARE 中同一受试者的 HRIR 对比）：
    * 方位角取反后 393 个方向【逐一精确对应】(角距 0.00 deg)   <- 与 ProcessHRIRs.m 的
      `sphCoo(:,1) = -sphCoo(:,1)` 一致
    * 幅度谱 corr: hL<->ear0 = +0.969, hR<->ear1 = +0.971
                   交叉配对 = -0.17 / -0.20  -> 耳序确认
    * 系统偏移 -4 dB, 去偏后残差 2.5 dB  <- ProcessHRIRs.m 的后处理造成, 见下方警告
"""
import os
import re
import sys
import collections
import numpy as np
import scipy.io as sio
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # spherical_spline.py sits beside this file
from spherical_spline import build_interpolation_matrix   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "SYMARE", "New_hrtfs")
DB = os.path.join(ROOT, "CARLabHrirDataBaseFixedByPierre")
OUT_L = os.path.join(os.path.dirname(__file__), "..", "SYMARE", "carlab_left_stps")
OUT_R = os.path.join(os.path.dirname(__file__), "..", "SYMARE", "carlab_right_stps")
REC, NSIDE, MIN_DIR = 2052, 16, 300
EXCLUDE = {"hats", "kemar"}          # 非人（假头）


def read_hrirfilt(path):
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), "<i2")[0])
    body = b[2:]
    n = len(body) // REC
    body = body[: n * REC].reshape(n, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(n, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(n, 2, L)
    az = -pos[:, 0]                  # ★ 取反，对齐 ProcessHRIRs.m
    return np.radians(az), np.radians(pos[:, 1]), sig


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def pick_subjects():
    """每个人挑方向最多的那次测量（并列时取索引最大 = 最近一次，同 ProcessHRIRs.m）"""
    d = sio.loadmat(os.path.join(ROOT, "carlabdb.mat"))["carlab_hrir"][0]
    cand = collections.defaultdict(list)
    for i, e in enumerate(d, 1):
        nm = np.array(e["username"]).ravel()
        nm = str(nm[0]).strip() if nm.size else ""
        key = norm(nm)
        if not key or key in EXCLUDE:
            continue
        p = os.path.join(DB, f"hrirfilt{i}")
        if not os.path.exists(p):
            continue
        n = (os.path.getsize(p) - 2) // REC
        if n >= MIN_DIR:
            cand[key].append((n, i, nm, p))
    return {k: max(v) for k, v in cand.items()}     # (ndir, idx) 最大


def to_healpix(sig_ear, W):
    """sig_ear: (Ndir, L) -> (129, npix) dB"""
    H = np.fft.fft(sig_ear.T, axis=0)[:129, :]      # (129, Ndir)
    return (20 * np.log10(np.abs(H) + 1e-9) @ W.T).astype(np.float32)


def main():
    os.makedirs(OUT_L, exist_ok=True)
    os.makedirs(OUT_R, exist_ok=True)
    subs = pick_subjects()
    print(f"去重后受试者: {len(subs)}")
    bygrid = collections.Counter(v[0] for v in subs.values())
    print(f"方向栅格分布: {dict(bygrid)}  -> 每种栅格建一次 STPS 矩阵\n")

    xyz_hp = np.column_stack(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    Wcache = {}
    done = 0
    for key, (ndir, idx, nm, path) in sorted(subs.items()):
        az, el, sig = read_hrirfilt(path)
        gkey = (ndir, round(float(az[0]), 6), round(float(el[0]), 6))
        if gkey not in Wcache:
            xyz = np.column_stack((np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)))
            print(f"  [建矩阵] {ndir} 方向 ...", flush=True)
            Wcache[gkey] = build_interpolation_matrix(xyz, xyz_hp,
                                                      spline_order=2, smoothing=1e-5)
        W = Wcache[gkey]
        tag = f"CL_{key[:20]}"
        np.save(os.path.join(OUT_L, f"{tag}_left_full.npy"), to_healpix(sig[:, 0, :], W))
        np.save(os.path.join(OUT_R, f"{tag}_right_full.npy"), to_healpix(sig[:, 1, :], W))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(subs)}", flush=True)
    print(f"\n完成: {done} 名受试者 -> {OUT_L} / {OUT_R}")

    print("""
================================ ★ 使用前必读 ================================
1) 校准差异。CARLab 是【原始】测量；SYMARE 的 HRIRs/Acoustic 是经
   ProcessHRIRs.m 处理过的（min-phase + delay 重构、低频增益修正、低仰角
   样条补点）。两者在同一个人身上对比：固定偏移 -4 dB，去偏后仍有 2.5 dB 残差。
   -> 直接混合训练会引入一个【真正的人为库指纹】（这与 Craig 反对的"人群差异"
      不同，这个是货真价实的处理管线差异，该消除）。
   两个选项：
     (a) 全部用原始 CARLab（把 SYMARE 那 48 个重合的人也从 CARLab 重取）；
     (b) 对 CARLab 施加同一套 ProcessHRIRs.m 后处理。
   (a) 更简单也更干净，推荐先做 (a)。

2) 与 SYMARE 的重合。CARLab 的 219 人中，至少 48 人与 SYMARE 的 61 人同名
   （按姓氏匹配，实际可能更多）。合并时必须按人去重，否则同一个人会以两种
   处理方式各出现一次 —— 那正是最坏的情况。

3) 栅格异构。393 / 475 / 412 / 392 四种方向栅格，本脚本已按栅格分别建 STPS
   矩阵。插值误差随栅格不同而不同，属于已知成本。
==============================================================================""")


if __name__ == "__main__":
    main()
