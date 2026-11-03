# ==========================================================================
# Cell 6 — 逐个人工核对 SYMARE61 ↔ CARLab 的配号
#
# 每页 8 个人，3 个方向。改 PAGE 翻页（0..7），重跑本 cell。
#
# 每张子图三条线：
#   黑实线   SYM_HAxx        （旧 SYMARE，61 人）
#   红虚线   第 1 名 CLxxx    （数据匹配的最优候选）
#   灰点线   第 2 名 CLxxx    （次优候选，用来判断"第 1 名是不是真的更像"）
#
# ★ 判据在 6–12 kHz（橙色带）的 notch：
#   同一个人 -> 黑线和红线的 notch 位置/深度对得上，而灰线明显对不上
#   不同的人 -> 红线和灰线都对不上，或者两条都"差不多像"（说明区分度不够）
#
# 两个库管线不同（raw vs HA 处理，整体差 3.81 dB），所以只看【谱形】，
# 每条线都减掉了自己的整体电平。绝对高低没有意义。
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

PAGE = 0            # ← 每检查完一页就 +1，重跑本 cell（0..7）
PER_PAGE = 8
NSIDE = 16
A_TAG, B_TAG = 'SYMARE61', 'CARLab'

DIRS = {'Front  (az 0, el 0)':      (np.pi / 2, 0.0),
        'Ipsi   (az 90, el 0)':     (np.pi / 2, np.pi / 2),
        'Up-front (az 0, el 45)':   (np.pi / 4, 0.0)}

# ---------- 相似度矩阵（和 Cell 5 同一套口径，这里自带一份以便独立运行）----------
Xa = MAT[A_TAG].reshape(MAT[A_TAG].shape[0], -1).astype(np.float64)
Xb = MAT[B_TAG].reshape(MAT[B_TAG].shape[0], -1).astype(np.float64)
Xa = Xa - Xa.mean(1, keepdims=True)          # ① 每人减自己的整体电平
Xb = Xb - Xb.mean(1, keepdims=True)
Xa = Xa - Xa.mean(0, keepdims=True); Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
Xb = Xb - Xb.mean(0, keepdims=True); Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
R = Xa @ Xb.T                                 # (61, 218)
ids_a, ids_b = DATA[A_TAG]['ids'], DATA[B_TAG]['ids']

# ---------- 方向 -> MAT 里的列号（MAT 做过 PIX_STEP 下采样）----------
PIXCOL, PIXLBL = {}, {}
for name, (th, ph) in DIRS.items():
    p_full = hp.ang2pix(NSIDE, th, ph)
    col = min(p_full // PIX_STEP, MAT[A_TAG].shape[2] - 1)
    t2, p2 = hp.pix2ang(NSIDE, col * PIX_STEP)          # 实际用到的方向
    PIXCOL[name] = col
    PIXLBL[name] = f'实际 az {np.degrees(p2):.0f}°, el {90-np.degrees(t2):.0f}°'

n_pages = int(np.ceil(len(ids_a) / PER_PAGE))
lo, hi = PAGE * PER_PAGE, min((PAGE + 1) * PER_PAGE, len(ids_a))
assert lo < len(ids_a), f'PAGE 最大 {n_pages-1}'
print(f'第 {PAGE+1}/{n_pages} 页 —— {ids_a[lo]} .. {ids_a[hi-1]}  '
      f'（共 {len(ids_a)} 人）')
for k, v in PIXLBL.items():
    print(f'    {k:<26}{v}')

# ---------- 先打印这一页的数字，看图时对照 ----------
print(f"\n  {'HA':<10}{'第1名':<9}{'r1':>7}{'第2名':<9}{'r2':>7}{'领先':>8}")
for i in range(lo, hi):
    o = np.argsort(-R[i])
    print(f'  {ids_a[i]:<10}{ids_b[o[0]]:<9}{R[i,o[0]]:>7.3f}'
          f'{ids_b[o[1]]:<9}{R[i,o[1]]:>7.3f}{R[i,o[0]]-R[i,o[1]]:>8.3f}')

# ---------- 画图 ----------
fkhz = TARGET_F / 1000
nrow = hi - lo
fig, axes = plt.subplots(nrow, len(DIRS), figsize=(5.0 * len(DIRS), 2.7 * nrow),
                         dpi=110, squeeze=False)
for r, i in enumerate(range(lo, hi)):
    o = np.argsort(-R[i])
    j1, j2 = o[0], o[1]
    for c, (name, col) in enumerate(PIXCOL.items()):
        ax = axes[r][c]
        ya = MAT[A_TAG][i, :, col]; ya = ya - ya.mean()
        y1 = MAT[B_TAG][j1, :, col]; y1 = y1 - y1.mean()
        y2 = MAT[B_TAG][j2, :, col]; y2 = y2 - y2.mean()
        ax.axvspan(6, 12, color='orange', alpha=.10)
        ax.plot(fkhz, y2, color='#999999', lw=1.0, ls=':',
                label=f'#2 {ids_b[j2]} ({R[i,j2]:.2f})')
        ax.plot(fkhz, ya, color='#111111', lw=2.0, label=ids_a[i])
        ax.plot(fkhz, y1, color='#c0392b', lw=1.8, ls='--',
                label=f'#1 {ids_b[j1]} ({R[i,j1]:.2f})')
        ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc='lower left', ncol=1)
        if r == 0:
            ax.set_title(name, fontsize=10)
        if c == 0:
            ax.set_ylabel(f'{ids_a[i]}\ndB (去电平)', fontsize=9)
        if r == nrow - 1:
            ax.set_xlabel('frequency (kHz)')
    axes[r][0].text(0.02, 0.97,
                    f'领先 {R[i,j1]-R[i,j2]:.3f}',
                    transform=axes[r][0].transAxes, va='top', fontsize=8.5,
                    bbox=dict(fc='white', ec='none', alpha=.7))
plt.tight_layout()
plt.show()

print(f'\n检查完把 PAGE 改成 {PAGE+1} 重跑本 cell'
      f'{"（这是最后一页）" if PAGE+1 >= n_pages else ""}')
print('把结论记在下一个 cell 的 VERDICT 里。')
