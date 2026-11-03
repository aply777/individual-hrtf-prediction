# ==========================================================================
# Cell 9 — 用 51 对姓名配对标定 r 的可信阈值
#
# 目的：把"我觉得不可信"换成一个数 —— r 大于多少时，配对的准确率是多少。
#
# 做法：姓名匹配和数据匹配是两条【完全独立】的证据链
#   （一条来自 carlabdb 的真名，一条来自 HRTF 谱形）。
# 把姓名匹配当标签，就能算出数据方法在各个阈值下的准确率，
# 进而给没有姓名的那 10 个人一个有依据的置信度。
#
# ⚠️ 标签本身不完美：姓名匹配在常见姓上翻过车（PROJECT_LOG §5 记录 5）。
#    所以"姓名说 A、数据说 B 且 r 很高"的情况，可能是姓名错而不是数据错。
#    下面把这类单独列出来。
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt

# NAME_MAP 沿用 Cell 8 里那份（51 对）
gt = {a: b for a, b in NAME_MAP.items() if a in ids_a and b in ids_b}
print(f'标签 {len(gt)} 对 | 候选池 {len(ids_b)} 人 | 随机命中 {1/len(ids_b):.2%}\n')

top1, top1_r, margin, is_hit, rank_of_gt, r_at_gt = [], [], [], [], [], []
for a, b in gt.items():
    i, j = ids_a.index(a), ids_b.index(b)
    o = np.argsort(-R[i])
    top1.append(ids_b[o[0]]); top1_r.append(R[i, o[0]])
    margin.append(R[i, o[0]] - R[i, o[1]])
    is_hit.append(ids_b[o[0]] == b)
    rank_of_gt.append(int(np.where(o == j)[0][0]) + 1)
    r_at_gt.append(R[i, j])
top1_r = np.array(top1_r); margin = np.array(margin)
is_hit = np.array(is_hit); rank_of_gt = np.array(rank_of_gt)

print(f'rank-1 命中 {is_hit.sum()}/{len(gt)} = {is_hit.mean():.1%}   '
      f'(rank<=3: {(rank_of_gt<=3).mean():.1%}, rank<=5: {(rank_of_gt<=5).mean():.1%})')
print(f'命中者的 r 中位 {np.median(top1_r[is_hit]):.3f} | '
      f'未命中者的 r 中位 {np.median(top1_r[~is_hit]):.3f}')

# ---------------- 按阈值统计 ----------------
print(f"\n{'阈值 r>':>8}{'保留':>7}{'其中与姓名一致':>16}{'准确率':>9}{'覆盖率':>9}")
best_t = None
for t in np.arange(0.40, 0.91, 0.05):
    sel = top1_r >= t
    if sel.sum() == 0:
        continue
    prec = is_hit[sel].mean()
    rec = is_hit[sel].sum() / len(gt)
    print(f'{t:>8.2f}{sel.sum():>7}{is_hit[sel].sum():>16}{prec:>9.1%}{rec:>9.1%}')
    if prec >= 0.95 and best_t is None:
        best_t = t
if best_t is not None:
    print(f'\n★ 准确率 >=95% 的最低阈值: r >= {best_t:.2f}')
else:
    print('\n没有阈值能达到 95% 准确率 —— 用下表里准确率最高的那档')

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4), dpi=120)
bins = np.linspace(0.2, 1.0, 30)
ax[0].hist(top1_r[is_hit], bins=bins, alpha=.75, color='#2471a3', label='与姓名一致')
ax[0].hist(top1_r[~is_hit], bins=bins, alpha=.75, color='#c0392b', label='与姓名不一致')
ax[0].set_xlabel('top-1 相关 r'); ax[0].set_ylabel('人数')
ax[0].set_title('两组的 r 分布（分得开就说明 r 是好判据）')
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

ax[1].scatter(top1_r[is_hit], margin[is_hit], s=42, c='#2471a3', label='与姓名一致')
ax[1].scatter(top1_r[~is_hit], margin[~is_hit], s=42, c='#c0392b', marker='x',
              label='与姓名不一致')
if best_t:
    ax[1].axvline(best_t, color='gray', ls='--', lw=1.2, label=f'r={best_t:.2f}')
ax[1].set_xlabel('top-1 相关 r'); ax[1].set_ylabel('领先幅度 r1-r2')
ax[1].set_title('r 与领先幅度'); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

# ---------------- 三类需要人看的 ----------------
T = best_t if best_t else 0.70

print(f'\n【1】姓名说 A、数据说 B，但数据 r 很高（>= {T:.2f}）—— 可能是姓名错')
n = 0
for k, (a, b) in enumerate(gt.items()):
    if not is_hit[k] and top1_r[k] >= T:
        print(f'    {a}  姓名→{b}  数据→{top1[k]} (r={top1_r[k]:.3f}, '
              f'领先 {margin[k]:.3f}, 姓名那位排第 {rank_of_gt[k]})')
        n += 1
print('    （无）' if n == 0 else '')

print(f'\n【2】姓名与数据一致但 r 低（< {T:.2f}）—— 大概率仍然对，只是谱形差异大')
n = 0
for k, (a, b) in enumerate(gt.items()):
    if is_hit[k] and top1_r[k] < T:
        print(f'    {a} = {b}  (r={top1_r[k]:.3f}, 领先 {margin[k]:.3f})'); n += 1
print('    （无）' if n == 0 else '')

print(f'\n【3】没有姓名匹配的 {len(ids_a)-len(gt)} 人 —— 按标定后的阈值给结论')
for i, a in enumerate(ids_a):
    if a in gt:
        continue
    o = np.argsort(-R[i])
    r1, m = R[i, o[0]], R[i, o[0]] - R[i, o[1]]
    tag = '★ 可信' if r1 >= T else ('? 存疑' if r1 >= T - 0.15 else '  不可信')
    print(f'    {a}  →  {ids_b[o[0]]}  r={r1:.3f}  领先={m:.3f}   {tag}')

print(f'\n注：两个管线之间有 3.52 dB 的【方向相关】差异（PROJECT_LOG 已量化），')
print(f'    所以曲线永远不会完全重合。r≈0.85 就是这条链路上"同一个人"的样子，')
print(f'    r≈0.30 才是"不同的人"。别拿"完全重合"当判据。')
