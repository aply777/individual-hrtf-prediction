#!/usr/bin/env python3
"""
把 carlab_npy_* 整理成最终格式：
  * 编号 CL001 ~ CL218（不用 HA 前缀，避免与 SYMARE 冲突）
  * 左右耳分两个文件夹
  * 一份 txt 对照表：编号 -> 姓名 / carlabdb 索引 / 方向数
  * 顺便验证并保存 HEALPix 左右镜像索引（见文末说明）
"""
import os
import sys
import json
import shutil
import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import symare_root                                # noqa: E402

ROOT = symare_root()
SRC_L, SRC_R = f"{ROOT}/carlab_npy_left", f"{ROOT}/carlab_npy_right"
DST_L, DST_R = f"{ROOT}/CARLab_left", f"{ROOT}/CARLab_right"
NSIDE = 16


def main():
    man = json.load(open(f"{ROOT}/carlab_manifest.json"))
    man.sort(key=lambda r: r["name"].lower())          # 按姓名排序后编号
    for d in (DST_L, DST_R):
        os.makedirs(d, exist_ok=True)

    lines = ["ID       Name                          carlabdb_idx  n_dirs",
             "-" * 62]
    for k, r in enumerate(man, 1):
        sid = f"CL{k:03d}"
        shutil.move(f"{SRC_L}/{r['tag']}_left_full.npy",  f"{DST_L}/{sid}_left_full.npy")
        shutil.move(f"{SRC_R}/{r['tag']}_right_full.npy", f"{DST_R}/{sid}_right_full.npy")
        lines.append(f"{sid:<9}{r['name']:<30}{r['carlab_idx']:>12}{r['ndir']:>8}")
        r["id"] = sid
    open(f"{ROOT}/CARLab_subjects.txt", "w").write("\n".join(lines) + "\n")
    json.dump(man, open(f"{ROOT}/CARLab_subjects.json", "w"), indent=1, ensure_ascii=False)
    for d in (SRC_L, SRC_R):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

    # ---------------- HEALPix 左右镜像索引 ----------------
    npix = hp.nside2npix(NSIDE)
    th, ph = hp.pix2ang(NSIDE, np.arange(npix))
    idx = hp.ang2pix(NSIDE, th, (-ph) % (2 * np.pi))
    exact = len(np.unique(idx)) == npix and np.array_equal(idx[idx], np.arange(npix))
    np.save(f"{ROOT}/healpix_mirror_idx.npy", idx)

    print(f"完成 {len(man)} 名受试者")
    print(f"  {DST_L}/CL001_left_full.npy ...")
    print(f"  {DST_R}/CL001_right_full.npy ...")
    print(f"  {ROOT}/CARLab_subjects.txt   (编号 -> 姓名)")
    print(f"  {ROOT}/healpix_mirror_idx.npy  (左右镜像置换, 精确={exact})")
    print()
    print("\n".join(lines[:6]))
    print("   ...")


if __name__ == "__main__":
    main()
