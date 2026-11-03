# ==========================================================================
# PC1 到底装的是什么？—— 方差分解
#
# 上一版的 L/R 测试因为标签顺序搞错而无效（提取顺序是 [L*8, R*8, ...]，
# 不是 L,R 交替）。这里重新提取并显式记录 side / subject 标签，
# 然后直接做方差分解，而不是靠猜 PC1 的含义。
# ==========================================================================
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = next(model.parameters()).device
model.eval()

# ---------------- 提取 latent + 显式标签 ----------------
Z, side, subj, level = [], [], [], []
with torch.no_grad():
    for L, R, names in DataLoader(train_dataset, batch_size=8, shuffle=False):
        for tag, x in (("L", L), ("R", R)):
            _, z = model(x.to(device))
            Z.append(z.cpu())
            side += [tag] * len(x)
            subj += list(names)                       # 文件名 -> 受试者
            level += x.mean(dim=(1, 2)).tolist()      # 该耳的整体 dB 电平
Z = torch.cat(Z, 0).double().reshape(len(side), -1).numpy()   # (N, 8256)
side = np.array([s == "L" for s in side])
subj = np.array(subj)
level = np.array(level)
N = len(side)
print(f"N={N}  左耳 {side.sum()}  右耳 {(~side).sum()}  唯一受试者 {len(set(subj))}")

Zc = Z - Z.mean(0)
tot = (Zc ** 2).sum()

# ---------------- ① 方差分解：side vs subject ----------------
def var_by_group(Zc, g):
    """组均值所解释的方差占比"""
    out = np.zeros_like(Zc)
    for u in np.unique(g):
        m = g == u
        out[m] = Zc[m].mean(0)
    return (out ** 2).sum() / tot

v_side = var_by_group(Zc, side.astype(int))
# 去掉 side 效应后，再看 subject 能解释多少
Z_nos = Zc.copy()
for u in (0, 1):
    m = side.astype(int) == u
    Z_nos[m] -= Zc[m].mean(0)
v_subj = (np.array([Z_nos[subj == u].mean(0) for u in subj]) ** 2).sum() / tot

print("\n" + "=" * 62)
print("① 方差分解")
print(f"   '哪只耳' (side)          : {100*v_side:5.1f}%")
print(f"   '哪个人' (subject|side)  : {100*v_subj:5.1f}%")
print(f"   残余 (测量噪声/耳内差异)  : {100*(1-v_side-v_subj):5.1f}%")

# ---------------- ② PC1 到底是什么 ----------------
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
pcs = U * S                                            # (N, 95) 主成分得分
print("\n" + "=" * 62)
print("② 前几个主成分各自与什么相关")
print(f"{'PC':>4}{'var%':>8}{'|corr| side':>13}{'|corr| level':>14}")
for i in range(6):
    c_side = abs(np.corrcoef(pcs[:, i], side.astype(float))[0, 1])
    c_lvl = abs(np.corrcoef(pcs[:, i], level)[0, 1])
    print(f"{i+1:>4}{100*S[i]**2/ (S**2).sum():>8.1f}{c_side:>13.3f}{c_lvl:>14.3f}")

# 用 PC1 单独分左右耳的最佳准确率（正确版）
thr = np.median(pcs[:, 0])
acc = max(((pcs[:, 0] > thr) == side).mean(), ((pcs[:, 0] > thr) != side).mean())
print(f"\n   PC1 二分左右耳准确率 : {acc:.3f}  (随机 = 0.500)")

# ---------------- ③ 去掉 side 之后，有效维度变多少 ----------------
def eff(M, p=0.9):
    s = np.linalg.svd(M - M.mean(0), compute_uv=False)
    c = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int((c < p).sum()) + 1

print("\n" + "=" * 62)
print("③ 有效维度 (90% 方差)")
print(f"   原始 (含 side)     : {eff(Zc)}")
print(f"   去掉 side 效应后   : {eff(Z_nos)}")
print(f"   只用左耳           : {eff(Zc[side])}")

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(pcs[side, 0], pcs[side, 1], s=18, label="left", alpha=.8)
ax[0].scatter(pcs[~side, 0], pcs[~side, 1], s=18, label="right", alpha=.8, marker="^")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("(a) PC1-PC2, colored by ear side")

ax[1].scatter(level, pcs[:, 0], s=18)
r = np.corrcoef(level, pcs[:, 0])[0, 1]
ax[1].set_xlabel("mean dB level of that ear"); ax[1].set_ylabel("PC1 score")
ax[1].set_title(f"(b) PC1 vs overall level  (r={r:.3f})"); ax[1].grid(alpha=.3)

ax[2].bar(["side", "subject", "residual"],
          [100*v_side, 100*v_subj, 100*(1-v_side-v_subj)])
ax[2].set_ylabel("% of latent variance"); ax[2].set_title("(c) variance decomposition")
ax[2].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()
