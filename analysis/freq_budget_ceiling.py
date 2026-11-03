# ==========================================================================
# 频率维度分配的天花板测试 —— 不重训，几分钟出结果
#
# 思路：把 latent 约束在"每个频点只用前 d_f 个主成分"的子空间里，
#       再跑 test-time optimization。得到的重建误差就是
#       "如果按这个方案分配容量，最好能做到多少"。
#
#       同样的总预算 sum(d_f) 下比较几种分配曲线，直接看 notch 带能不能变好。
#       如果重分配在天花板上就没收益，那重训也不会有 —— 省下一次训练。
# ==========================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = next(model.parameters()).device
decoder = model.decoder
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

STRIDE, STEPS, LR, N_EARS = 4, 200, 0.1, 8
NBINS, FS = 129, 48000
sel = np.arange(0, NBINS, STRIDE)                 # 频率嵌入精确对齐
freqs = sel * FS / 256.0
F = len(sel)
band = {"low<5k": freqs < 5000, "notch6-12k": (freqs >= 6000) & (freqs <= 12000),
        "high>12k": freqs > 12000, "overall": np.ones(F, bool)}

# ---------------- 训练码的逐频点 PCA 基 ----------------
Ztr = []
with torch.no_grad():
    for L, R, _ in DataLoader(train_dataset, batch_size=8, shuffle=False):
        for x in (L, R):
            _, z = model(x.to(device))
            Ztr.append(z[:, sel, :].cpu())
Ztr = torch.cat(Ztr, 0).double()                  # (N,F,64)
mu = Ztr.mean(0)                                  # (F,64)
V = torch.zeros(F, 64, 64, dtype=torch.float64)
for f in range(F):
    _, _, Vt = torch.linalg.svd(Ztr[:, f, :] - mu[f], full_matrices=False)
    V[f] = Vt                                     # 行 = 主方向, 已按重要性排序
mu, V = mu.float().to(device), V.float().to(device)

# ---------------- 验证数据 ----------------
xs = []
for L, R, _ in val_dataset:
    xs += [L, R]
    if len(xs) >= N_EARS:
        break
X = torch.stack(xs[:N_EARS]).to(device)[:, sel, :].contiguous()

def rmse(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        d = d[:, torch.as_tensor(m, device=d.device), :]
    return torch.sqrt(d.mean()).item()

# ---------------- 在 d_f 维子空间里优化 ----------------
def ceiling(dprof, steps=STEPS, lr=LR):
    """dprof: (F,) 每个频点保留多少个主成分。返回各频段 RMSE。"""
    mask = torch.zeros(F, 64, device=device)
    for f, d in enumerate(dprof):
        mask[f, :int(d)] = 1.0
    c = torch.zeros(X.shape[0], F, 64, device=device, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        z = mu.unsqueeze(0) + torch.einsum("bfd,fde->bfe", c * mask, V)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((decoder(z).float() - X) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        z = mu.unsqueeze(0) + torch.einsum("bfd,fde->bfe", c * mask, V)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec = decoder(z).float()
    return {k: rmse(rec, X, m) for k, m in band.items()}

# ---------------- 分配方案（同一总预算）----------------
BUDGET = 16 * F                     # 均匀 16 维时的总预算
# 数据侧的需求曲线（来自原生网格 SVD：低频 4 / notch 16 / 高频 24）
need = np.interp(freqs, [0, 5000, 6000, 12000, 16000, 24000],
                 [4, 4, 10, 16, 24, 24])
prop = np.maximum(2, np.round(need / need.sum() * BUDGET)).astype(int)
prop = np.minimum(prop, 64)

profiles = {
    "uniform 8":        np.full(F, 8),
    "uniform 16":       np.full(F, 16),
    "uniform 64 (现状)": np.full(F, 64),
    f"proportional (预算同 uniform16)": prop,
}
print(f"总预算对照:  uniform16 = {16*F},  proportional = {prop.sum()}")
print(f"proportional profile: 低频 {prop[freqs<5000].mean():.0f} 维, "
      f"notch {prop[(freqs>=6000)&(freqs<=12000)].mean():.0f} 维, "
      f"高频 {prop[freqs>12000].mean():.0f} 维\n")

res = {}
print(f"{'方案':<32}{'overall':>9}{'low':>8}{'notch':>8}{'high':>8}")
print("-" * 66)
for name, prof in profiles.items():
    r = ceiling(prof)
    res[name] = r
    print(f"{name:<32}{r['overall']:>9.3f}{r['low<5k']:>8.3f}"
          f"{r['notch6-12k']:>8.3f}{r['high>12k']:>8.3f}")

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(freqs/1000, profiles["uniform 16"], label="uniform 16", lw=2)
ax[0].plot(freqs/1000, prop, label="proportional", lw=2, ls="--")
ax[0].axvspan(6, 12, color="orange", alpha=.12)
ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("dims allocated")
ax[0].set_title("allocation profiles (same total budget)"); ax[0].legend(); ax[0].grid(alpha=.3)

names = list(res); bands = ["low<5k", "notch6-12k", "high>12k"]
xp = np.arange(len(bands)); w = 0.8/len(names)
for i, n in enumerate(names):
    ax[1].bar(xp + i*w, [res[n][b] for b in bands], w, label=n)
ax[1].set_xticks(xp + 0.4); ax[1].set_xticklabels(bands)
ax[1].set_ylabel("reconstruction ceiling RMSE (dB)")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

# ---------------- 判据 ----------------
u16, pr = res["uniform 16"], res[list(profiles)[-1]]
print("\n" + "="*66)
print(f"同预算下重分配对 notch 带的收益: {u16['notch6-12k'] - pr['notch6-12k']:+.3f} dB")
print(f"                低频的代价      : {u16['low<5k'] - pr['low<5k']:+.3f} dB")
if u16['notch6-12k'] - pr['notch6-12k'] > 0.1:
    print("-> 重分配在天花板上就有收益，值得改损失/架构去实现它。")
else:
    print("-> 天花板上就没收益，重训也不会有。别改分配，问题在别处。")
print("="*66)
