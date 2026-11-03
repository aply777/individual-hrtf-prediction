#!/usr/bin/env python3
"""
CAR Lab HRIR 档案清点：到底有多少个可用受试者？

背景：carlabdb.mat 索引表里有 267 条记录、248 个不重名的 username。但
      "表里有名字" != "有可用数据"。本脚本直接解码二进制、逐个核对。

二进制格式（由文件大小规律 + 字节检视反推，无需 MATLAB carlabtoolbox）：
    [int16 LE] 滤波器长度 L (= 256)
    随后 N 条记录，每条 4 + 2*L*4 = 2052 字节：
        [int16 az][int16 el]   方向，单位 度
        [float32 x L]          耳 1
        [float32 x L]          耳 2
    文件大小 = 2 + N*2052   ->   N = (size-2)/2052

注意 ProcessHRIRs.m 里有 `sphCoo(:,1) = -sphCoo(:,1)`，即方位角要取反
才符合他们的约定。这里保留原始值并注明。
"""
import os
import re
import sys
import collections
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import symare_root                                # noqa: E402

ROOT = os.path.join(symare_root(), "New_hrtfs")
DB = os.path.join(ROOT, "CARLabHrirDataBaseFixedByPierre")
REC = 2052


def read_hrirfilt(path):
    """-> (az_deg, el_deg, hrir[N,2,L])"""
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), "<i2")[0])
    body = b[2:]
    n = len(body) // REC
    body = body[: n * REC].reshape(n, REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(n, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(n, 2, L)
    return pos[:, 0], pos[:, 1], sig, L


def norm_name(s):
    """粗归一化，用于找疑似同一个人的不同拼写"""
    return re.sub(r"[^a-z]", "", s.lower())


def main():
    d = sio.loadmat(os.path.join(ROOT, "carlabdb.mat"))["carlab_hrir"][0]
    entries = []
    for i, e in enumerate(d, 1):
        nm = np.array(e["username"]).ravel()
        nm = str(nm[0]).strip() if nm.size else ""
        dt = np.array(e["date"]).ravel()
        dt = str(dt[0]).strip() if dt.size else ""
        entries.append({"idx": i, "name": nm, "date": dt})

    print("=" * 78)
    print("1. 索引表 vs 实际文件")
    print("=" * 78)
    ok = 0
    for e in entries:
        p = os.path.join(DB, f"hrirfilt{e['idx']}")
        e["file"] = p if os.path.exists(p) else None
        if e["file"]:
            e["ndir"] = (os.path.getsize(p) - 2) // REC
            ok += 1
        else:
            e["ndir"] = 0
    print(f"   索引条目           : {len(entries)}")
    print(f"   实际存在 hrirfilt  : {ok}")
    print(f"   缺失               : {[e['idx'] for e in entries if not e['file']]}")

    print("\n" + "=" * 78)
    print("2. 方向数分布（决定能不能用）")
    print("=" * 78)
    c = collections.Counter(e["ndir"] for e in entries if e["file"])
    for n, k in sorted(c.items(), reverse=True):
        tag = "  <- 与 SYMARE 同栅格" if n == 393 else ""
        flag = "" if n >= 300 else "   ★ 太稀疏"
        print(f"   {n:>5} 方向 : {k:>3} 个文件{tag}{flag}")

    print("\n" + "=" * 78)
    print("3. 数据健全性抽检（读实际波形）")
    print("=" * 78)
    bad = []
    for e in entries:
        if not e["file"]:
            continue
        try:
            az, el, sig, L = read_hrirfilt(e["file"])
            e["L"] = L
            e["peak"] = float(np.abs(sig).max())
            e["silent"] = int((np.abs(sig).max(axis=(1, 2)) < 1e-8).sum())
            e["nan"] = bool(np.isnan(sig).any())
            e["az_span"] = (float(az.min()), float(az.max()))
            e["el_span"] = (float(el.min()), float(el.max()))
            if e["nan"] or e["peak"] < 1e-6 or e["silent"] > 0.1 * len(az):
                bad.append(e["idx"])
        except Exception as ex:
            e["peak"] = None
            bad.append(e["idx"])
    good = [e for e in entries if e["file"] and e.get("peak")]
    Ls = collections.Counter(e["L"] for e in good)
    print(f"   滤波器长度 L       : {dict(Ls)}")
    print(f"   峰值幅度 中位      : {np.median([e['peak'] for e in good]):.3f}")
    print(f"   含 NaN / 全静音    : {len(bad)} 个 -> {bad[:10]}")
    e0 = good[0]
    print(f"   方位角范围         : {e0['az_span']}  (ProcessHRIRs.m 会取反)")
    print(f"   仰角范围           : {e0['el_span']}")

    print("\n" + "=" * 78)
    print("4. ★ 去重后到底有多少个人")
    print("=" * 78)
    usable = [e for e in good if e["ndir"] >= 300 and e["idx"] not in bad]
    byname = collections.defaultdict(list)
    for e in usable:
        byname[norm_name(e["name"])].append(e)
    print(f"   可用文件 (>=300 方向, 数据健全) : {len(usable)}")
    print(f"   去重后的不同人                  : {len(byname)}")
    multi = {k: v for k, v in byname.items() if len(v) > 1}
    print(f"   被测过多次的人                  : {len(multi)}"
          f"  (共占 {sum(len(v) for v in multi.values())} 个文件)")
    for k, v in list(multi.items())[:5]:
        print(f"      {v[0]['name']:<22} x{len(v)}  idx={[x['idx'] for x in v]}")

    print("\n" + "=" * 78)
    print("5. 与已有数据的重叠")
    print("=" * 78)
    sym = os.path.join(ROOT, "SymareHrirs")
    if os.path.isdir(sym):
        symn = {norm_name(f[5:-4]) for f in os.listdir(sym) if f.startswith("HRIR_")}
        inter = symn & set(byname)
        print(f"   SymareHrirs 文件数              : {len(symn)}")
        print(f"   与 CARLab 姓氏重合              : {len(inter)}")
        print(f"   -> CARLab 中的【新增】人数      : {len(byname) - len(inter)}")
        print(f"      (SYMARE 用姓氏命名, CARLab 用全名, 此处按姓氏粗匹配, 可能低估重合)")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"   可直接使用的独立受试者 ≈ {len(byname)}")
    print(f"   其中方向栅格 = 393 (同 SYMARE) 的文件: "
          f"{sum(1 for e in usable if e['ndir']==393)}")
    print("   注意: 不同文件的方向栅格不完全一致 -> 必须各自建 STPS 权重矩阵")
    print("         (参考 HRIRs/Code/spherical_spline.py 的 build_interpolation_matrix)")


if __name__ == "__main__":
    main()
