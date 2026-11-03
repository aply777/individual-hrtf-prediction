# ==========================================================================
# Cell 8 — 用已知的姓名配对反过来标定比较口径
#
# 背景：目视核对时"一个都对不上"。但重合肯定存在 ——
#   · CL053 与 SYMARE 中的一名受试者是同一人（按记录核对，身份不公开）
#   · PROJECT_LOG 记过 SymareHrirs/HRIR_<name>.mat 与 HRIRs/Acoustic/HA<nn>.mat
#     逐字节相同，由此得到完整的 名字 <-> HA 映射
#   · 姓名唯一匹配 51 对，其中 35 对的数据 argmax 也一致（随机命中 0.46%）
#
# 所以问题更可能在【比较口径】。两个最可能的错：
#   H1  SYMARE61 的 fs 填错（48000 vs 44100）-> 频率轴整体拉伸 8.8%，
#       12 kHz 处 notch 错位 ~1 kHz，目视必然对不上
#   H2  方位角约定相反（CARLab 导出时做了 az 取反）-> 左右耳/前后颠倒
#
# 检验方法：把那 51 对姓名配对当 ground truth，对 2x2 四种假设各算一次
# rank-1 命中率。命中率最高的那一组就是正确口径 —— 这是用独立标签标定管线。
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

A_TAG, B_TAG = 'SYMARE61', 'CARLab'
NSIDE = 16

# ---- 姓名唯一匹配（来自 analysis/ha_to_carlab_map.csv，与数据完全独立）----
NAME_MAP = {
    'SYM_HA01': 'CL161', 'SYM_HA02': 'CL048', 'SYM_HA03': 'CL014', 'SYM_HA04': 'CL176',
    'SYM_HA05': 'CL182', 'SYM_HA06': 'CL009', 'SYM_HA07': 'CL216', 'SYM_HA08': 'CL053',
    'SYM_HA09': 'CL042', 'SYM_HA10': 'CL106', 'SYM_HA11': 'CL160', 'SYM_HA12': 'CL110',
    'SYM_HA13': 'CL005', 'SYM_HA14': 'CL063', 'SYM_HA15': 'CL022', 'SYM_HA16': 'CL065',
    'SYM_HA18': 'CL003', 'SYM_HA19': 'CL178', 'SYM_HA20': 'CL132', 'SYM_HA21': 'CL121',
    'SYM_HA23': 'CL026', 'SYM_HA24': 'CL147', 'SYM_HA25': 'CL031', 'SYM_HA26': 'CL079',
    'SYM_HA27': 'CL213', 'SYM_HA28': 'CL070', 'SYM_HA30': 'CL040', 'SYM_HA31': 'CL114',
    'SYM_HA32': 'CL134', 'SYM_HA33': 'CL044', 'SYM_HA34': 'CL208', 'SYM_HA35': 'CL112',
    'SYM_HA36': 'CL030', 'SYM_HA37': 'CL153', 'SYM_HA38': 'CL111', 'SYM_HA40': 'CL007',
    'SYM_HA41': 'CL018', 'SYM_HA42': 'CL150', 'SYM_HA43': 'CL033', 'SYM_HA44': 'CL165',
    'SYM_HA45': 'CL061', 'SYM_HA46': 'CL130', 'SYM_HA49': 'CL174', 'SYM_HA50': 'CL205',
    'SYM_HA51': 'CL113', 'SYM_HA52': 'CL075', 'SYM_HA53': 'CL038', 'SYM_HA54': 'CL008',
    'SYM_HA55': 'CL041', 'SYM_HA56': 'CL210', 'SYM_HA57': 'CL163',
}

ids_a, ids_b = DATA[A_TAG]['ids'], DATA[B_TAG]['ids']
gt = [(ids_a.index(a), ids_b.index(b)) for a, b in NAME_MAP.items()
      if a in ids_a and b in ids_b]
print(f'ground truth（姓名匹配）{len(gt)} 对，随机命中率 1/{len(ids_b)} = '
      f'{1/len(ids_b):.2%}\n')

# ---- 中矢面镜像置换（用于 H2：方位角约定相反）----
th_, ph_ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
MIR_FULL = hp.ang2pix(NSIDE, th_, (-ph_) % (2 * np.pi))
MIR = MIR_FULL[::PIX_STEP] // PIX_STEP          # 映射到 MAT 的下采样列
MIR = np.clip(MIR, 0, MAT[A_TAG].shape[2] - 1)


def load_raw(tag):
    """重新读原始 129 bin（不插值），供改 fs 假设用"""
    import os
    D = DATA[tag]
    return np.stack([np.load(os.path.join(D['dir'], f)).astype(np.float32)
                     for f in D['files']])[:, :, ::PIX_STEP]


RAW = {t: load_raw(t) for t in (A_TAG, B_TAG)}
print('原始张量:', {t: RAW[t].shape for t in RAW})


def score(fs_a, mirror):
    """按给定假设重算相似度，返回 (rank1 命中率, 命中的平均排名, R)"""
    Wa = interp_matrix(np.arange(129) * fs_a / 256.0, TARGET_F)
    Wb = interp_matrix(np.arange(129) * DATA[B_TAG]['fs'] / 256.0, TARGET_F)
    Ma = np.einsum('fk,nkp->nfp', Wa, RAW[A_TAG])
    Mb = np.einsum('fk,nkp->nfp', Wb, RAW[B_TAG])
    if mirror:
        Mb = Mb[:, :, MIR]
    Xa = Ma.reshape(len(ids_a), -1).astype(np.float64)
    Xb = Mb.reshape(len(ids_b), -1).astype(np.float64)
    Xa -= Xa.mean(1, keepdims=True); Xb -= Xb.mean(1, keepdims=True)
    Xa -= Xa.mean(0, keepdims=True); Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
    Xb -= Xb.mean(0, keepdims=True); Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    R = Xa @ Xb.T
    ranks = [int(np.where(np.argsort(-R[i]) == j)[0][0]) + 1 for i, j in gt]
    hit1 = sum(r == 1 for r in ranks)
    return hit1 / len(gt), float(np.median(ranks)), R, ranks


print(f"\n{'SYMARE61 fs':>12}{'镜像':>7}{'rank-1 命中率':>15}{'命中排名中位':>14}")
best, results = None, {}
for fs_a in (48000, 44100):
    for mir in (False, True):
        h, med, R, ranks = score(fs_a, mir)
        results[(fs_a, mir)] = (h, med, R, ranks)
        star = ''
        if best is None or h > best[0]:
            best = (h, fs_a, mir); star = ''
        print(f'{fs_a:>12}{"是" if mir else "否":>7}{h:>14.1%}{med:>14.0f}')
print(f'\n★ 最优假设: SYMARE61 fs = {best[1]}, 镜像 = {"是" if best[2] else "否"}'
      f'   (rank-1 {best[0]:.1%})')

h0 = results[(48000, False)][0]
if best[0] > h0 + 0.10:
    print(f'  当前 Cell 0 用的是 fs=48000 / 不镜像，命中率 {h0:.1%} —— '
          f'比最优低 {best[0]-h0:.1%}，说明口径确实错了。')
    print(f'  ★ 去 Cell 0 把 SYMARE61 的 fs 改成 {best[1]}，然后从 Cell 2 重跑。')
elif best[0] < 0.25:
    print('  四种假设命中率都很低 —— 问题不在 fs / 方位角，可能是：')
    print('    · SYMARE61 的 npy 其实来自 BEM 仿真而非实测')
    print('    · 两边的 HEALPix 插值/网格不一致')
    print('    · 姓名映射本身有问题（去核对 carlabdb 的测量日期）')
else:
    print(f'  当前口径已经是最优（{h0:.1%}）—— fs 和方位角都没错，'
          '目视对不上是因为两个管线的方向相关差异本来就有 3.52 dB。')

# ---- 高频滚降边缘：独立佐证 fs ----
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), dpi=120)
for k, fs_try in enumerate((48000, 44100)):
    f = np.arange(129) * fs_try / 256.0 / 1000
    for tag, c in ((A_TAG, '#111111'), (B_TAG, '#c0392b')):
        m = RAW[tag].mean(axis=(0, 2))
        fx = np.arange(129) * DATA[tag]['fs'] / 256.0 / 1000 if tag == B_TAG else f
        m = m - m[(fx >= 1) & (fx <= 8)].mean()
        ax[k].plot(fx, m, color=c, lw=1.8, label=tag)
    ax[k].axvline(16.3125, color='#2471a3', ls='--', lw=1.1)
    ax[k].set_title(f'假设 SYMARE61 fs = {fs_try}')
    ax[k].set_xlabel('kHz'); ax[k].set_ylabel('dB'); ax[k].set_xlim(0, 24)
    ax[k].legend(fontsize=8); ax[k].grid(alpha=.3)
plt.tight_layout(); plt.show()
print('蓝虚线是 CARLab 已知的 16.31 kHz 断崖。哪个假设下 SYMARE61 的滚降边缘')
print('落在物理上合理的位置（接近它自己的 Nyquist，而不是被拉伸/压缩），哪个就对。')
