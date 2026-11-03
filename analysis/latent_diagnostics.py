# ==========================================================================
# Latent 诊断：8256 个数里，到底有多少在承载"个体"？
#
# 术语澄清：
#   config["latent_dim"] = 64   -> 张量最后一维的大小
#   Z                    -> 实际张量, shape (N_ears, 129, 64) = 每耳 8256 个数
#
# 输出三样东西：
#   1) 有多少维跨受试者真的在变（其余是"平均人"，不含个体信息）
#   2) 方差集中度（前 k% 的维占多少方差）
#   3) 一个存好的标准化器 —— 下游回归前必须套上
# ==========================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = next(model.parameters()).device
model.eval()

# ---------------- 1. 提取全部训练耳的 latent（全 129 频点）----------------
Zs = []
loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
with torch.no_grad():
    for L, R, _ in loader:
        for x in (L, R):
            _, z = model(x.to(device))
            Zs.append(z.cpu())
Z = torch.cat(Zs, 0)                       # (N_ears, 129, 64)
N, F, D = Z.shape
print(f"训练 latent: {tuple(Z.shape)}  = 每耳 {F*D} 个数, 共 {N} 只耳")

mu  = Z.mean(0)                            # (129,64)  "平均人"
sd  = Z.std(0)                             # (129,64)  跨受试者的变化幅度
s   = sd.flatten()

# ---------------- 2. 有多少维真的在变 ----------------
print("\n" + "="*66)
print("① 有多少维承载个体信息？")
for thr in (0.5, 0.1, 0.01):
    k = (s > thr * s.mean()).sum().item()
    print(f"   std > {thr:>4} x 平均std : {k:>5} / {s.numel()}  ({100*k/s.numel():.1f}%)")
# 变异系数：std 相对于该维自身的量级
cv = (sd / mu.abs().clamp_min(1e-6)).flatten()
print(f"   变异系数 std/|mean| 中位数 : {cv.median():.4f}")
print(f"   (很小 => 该维几乎是常数，装的是'平均人'而不是个体)")

# ---------------- 3. 方差集中度 ----------------
v = (s**2).sort(descending=True).values
cum = torch.cumsum(v, 0) / v.sum()
print("\n" + "="*66)
print("② 方差集中在多少维里？")
for p in (0.5, 0.8, 0.9, 0.95, 0.99):
    k = int((cum < p).sum()) + 1
    print(f"   捕获 {p*100:>4.0f}% 方差需要 {k:>5} 维  ({100*k/len(v):.2f}%)")
print(f"   最大/最小 std = {s.max()/s.clamp_min(1e-12).min():.0f}x")

# ---------------- 4. 每个频点有多少个活跃维 ----------------
fs = 48000
freqs = np.arange(F) * fs / 256.0
act = (sd > 0.1 * sd.mean()).sum(1).numpy()        # 每个频点活跃的维数 (<=64)
print("\n" + "="*66)
print("③ 每个频点的活跃维数 (共 64)")
for lo, hi in [(200, 2000), (2000, 5000), (6000, 12000), (12000, 16000)]:
    m = (freqs >= lo) & (freqs <= hi)
    print(f"   {lo/1000:>5.1f}-{hi/1000:<5.1f} kHz : 中位数 {np.median(act[m]):.0f} / 64")

# ---------------- 5. 存标准化器（下游回归前必须用）----------------
torch.save({"mu": mu, "sd": sd.clamp_min(1e-6)},
           "/content/drive/MyDrive/HRTF_Checkpoints/latent_standardizer.pt")
print("\n标准化器已存 -> latent_standardizer.pt")
print("下游用法:  z_std = (z - mu) / sd     再喂给回归器 / 算 R^2")

# ---------------- 6. 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].semilogy(np.sort(s.numpy())[::-1])
ax[0].set_xlabel("dim (sorted)"); ax[0].set_ylabel("std across ears (log)")
ax[0].set_title("① 各维跨受试者的变化幅度"); ax[0].grid(alpha=.3)

ax[1].plot(np.arange(1, len(cum)+1), cum.numpy())
for p in (0.9, 0.95):
    ax[1].axhline(p, ls=":", color="gray")
ax[1].set_xscale("log"); ax[1].set_xlabel("dims kept"); ax[1].set_ylabel("cum. variance")
ax[1].set_title("② 方差集中度"); ax[1].grid(alpha=.3)

ax[2].plot(freqs/1000, act, lw=2, color="darkgreen")
ax[2].axvspan(6, 12, color="orange", alpha=.12)
ax[2].set_xlim(0, 20); ax[2].set_ylim(0, 64)
ax[2].set_xlabel("frequency (kHz)"); ax[2].set_ylabel("active dims / 64")
ax[2].set_title("③ 每频点用了多少维"); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.show()

# ---------------- 7. 解读 ----------------
k90 = int((cum < 0.9).sum()) + 1
print("\n" + "="*66)
if k90 < 0.05 * len(v):
    print(f"结论: 8256 维里只有 ~{k90} 维承载 90% 的个体变化 ({100*k90/len(v):.1f}%)。")
    print("      -> latent 严重冗余。下游应当回归到【前 k 个主成分】而不是全部 8256 维，")
    print("         而且必须先标准化，否则损失被少数高方差维主导。")
else:
    print(f"结论: 需要 {k90} 维才能覆盖 90% 变化 -> latent 没有明显冗余。")
if np.median(act[(freqs>=6000)&(freqs<=12000)]) > 1.5*np.median(act[freqs<5000]):
    print("      notch 带用的维数明显多于低频 -> 支持'维度应随频率分配'。")
print("="*66)
