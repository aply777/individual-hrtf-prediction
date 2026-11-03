# ==========================================================================
# Latent 诊断 v2 —— 修正 v1 的两个错误
#
# v1 错在哪：
#   * 对【各维的边际方差】排序累加，当成了"方差集中度"。那不是 PCA，
#     完全没度量维度间的相关性。8256 个坐标可以每个都有方差，
#     而整体只活在一个低维子空间里 —— 这正是本例的情况。
#   * N=96 只耳，去中心化后【秩最多 95】，所以 v1 报的 "4799 维" 不可能有意义。
#
# v2 做真正的 PCA。并且逐频点 PCA 是良定义的（96 样本 > 64 维），
# 因此能给出"每个频点真正用了多少维"的可靠答案。
# ==========================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = next(model.parameters()).device
model.eval()

# ---------------- 提取 latent ----------------
Zs = []
with torch.no_grad():
    for L, R, _ in DataLoader(train_dataset, batch_size=8, shuffle=False):
        for x in (L, R):
            _, z = model(x.to(device))
            Zs.append(z.cpu())
Z = torch.cat(Zs, 0).double()               # (N,129,64)
N, F, D = Z.shape
print(f"latent {tuple(Z.shape)}  |  N={N} 耳 = {N//2} 受试者")
print(f"★ 去中心化后秩上限 = {N-1}  —— 有效维度不可能超过它，与 latent_dim 无关\n")

fs = 48000
freqs = np.arange(F) * fs / 256.0

def eff_rank(S, p):
    c = np.cumsum(S**2) / np.sum(S**2)
    return int((c < p).sum()) + 1

# ---------------- ① 全局 PCA（正确版）----------------
Zc = (Z - Z.mean(0)).reshape(N, -1).numpy()
S = np.linalg.svd(Zc, compute_uv=False)
cum = np.cumsum(S**2) / np.sum(S**2)
print("=" * 66)
print("① 全局 PCA（对 96 x 8256 去中心化矩阵做 SVD）")
for p in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(f"   捕获 {p*100:>4.0f}% 方差需要 {eff_rank(S,p):>3} 个主成分 / 上限 {N-1}")
print(f"   participation ratio = {(S**2).sum()**2/ (S**4).sum():.1f}  (另一种有效维度度量)")

# ---------------- ② 逐频点 PCA（这次是良定义的）----------------
er90, er95 = [], []
for f in range(F):
    Xf = (Z[:, f, :] - Z[:, f, :].mean(0)).numpy()      # (96,64), 96>64 -> 良定义
    Sf = np.linalg.svd(Xf, compute_uv=False)
    er90.append(eff_rank(Sf, 0.90)); er95.append(eff_rank(Sf, 0.95))
er90, er95 = np.array(er90), np.array(er95)
print("\n" + "=" * 66)
print("② 每个频点真正用了 64 维中的几维（逐频点 PCA）")
for lo, hi in [(200, 2000), (2000, 5000), (6000, 12000), (12000, 16000), (16000, 20000)]:
    m = (freqs >= lo) & (freqs <= hi)
    print(f"   {lo/1000:>5.1f}-{hi/1000:<5.1f} kHz : 90% 方差需 {np.median(er90[m]):>4.0f} 维,"
          f"  95% 需 {np.median(er95[m]):>4.0f} 维")

# ---------------- ③ 边际尺度（v1 唯一站得住的部分）----------------
sd = Z.std(0)
print("\n" + "=" * 66)
print("③ 边际尺度差异（这条 v1 是对的）")
print(f"   std: min {sd.min():.3f}, max {sd.max():.1f}, 动态范围 {sd.max()/sd.min():.0f}x")
print("   -> 下游回归/算 R^2 之前必须 per-dim 标准化，否则损失被高方差维主导")

torch.save({"mu": Z.mean(0).float(), "sd": sd.clamp_min(1e-6).float()},
           "/content/drive/MyDrive/HRTF_Checkpoints/latent_standardizer.pt")

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(np.arange(1, len(cum)+1), cum, marker="o", ms=3)
for p in (0.9, 0.95):
    ax[0].axhline(p, ls=":", color="gray")
ax[0].set_xlabel("PCA component"); ax[0].set_ylabel("cumulative variance")
ax[0].set_title(f"(a) global PCA of latent (rank cap = {N-1})"); ax[0].grid(alpha=.3)

ax[1].plot(freqs/1000, er90, lw=2, label="90% var")
ax[1].plot(freqs/1000, er95, lw=1.5, ls="--", label="95% var")
ax[1].axhline(64, color="crimson", ls=":", label="latent_dim = 64")
ax[1].axvspan(6, 12, color="orange", alpha=.12)
ax[1].set_xlim(0, 20); ax[1].set_xlabel("frequency (kHz)")
ax[1].set_ylabel("effective dims / 64")
ax[1].set_title("(b) dims actually used, per frequency"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

ax[2].semilogy(np.sort(sd.flatten().numpy())[::-1])
ax[2].set_xlabel("dim (sorted)"); ax[2].set_ylabel("std across ears (log)")
ax[2].set_title("(c) marginal scale spread"); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.show()

# ---------------- 解读 ----------------
k90 = eff_rank(S, 0.90)
print("\n" + "=" * 66)
print(f"结论：latent 的全局有效维度 ≈ {k90}（受 N={N} 限制，上限 {N-1}）。")
print(f"      下游不该回归 8256 维，应回归【前 ~{k90} 个主成分】，且先标准化。")
band_lo = np.median(er90[(freqs >= 200) & (freqs <= 5000)])
band_no = np.median(er90[(freqs >= 6000) & (freqs <= 12000)])
print(f"      逐频点：低频 {band_lo:.0f} 维 vs notch 带 {band_no:.0f} 维 "
      f"({band_no/max(band_lo,1):.1f}x)")
if band_no > 1.5 * band_lo:
    print("      -> 支持'维度随频率分配'；均匀 64 维在低频浪费、在 notch 带可能不足。")
else:
    print("      -> 各频段用量接近，'均匀分配'没有明显浪费。")
print("=" * 66)
