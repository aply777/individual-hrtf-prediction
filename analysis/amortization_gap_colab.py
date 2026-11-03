# ==========================================================================
# Test-time optimization: 测量 amortization gap
#
# 回答三个问题：
#   Q1  encoder 给的 z 离"最优 z"有多远？        -> 决定要不要 auto-decoder
#   Q2  64 维/频点 到底装不装得下？               -> 决定 Craig 那场"维度够不够"的争论
#   Q3  冷启动（不用 encoder）能不能到同样的解？   -> 决定纯 auto-decoder 可不可行
#
# 不重训任何权重。decoder 全程冻结，只对 z 做梯度下降。
# ==========================================================================
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 0. 找到 encoder / decoder 子模块 ----------
print("model children:", [n for n, _ in model.named_children()])

def _find(model, keys):
    for k in keys:
        m = getattr(model, k, None)
        if isinstance(m, nn.Module):
            return m
    for n, m in model.named_children():
        if any(k in n.lower() for k in keys):
            return m
    return None

decoder = _find(model, ["decoder", "dec"])
assert decoder is not None, "没找到 decoder 子模块，请把上面打印的名字告诉我"
print("decoder ->", type(decoder).__name__)

for p in model.parameters():
    p.requires_grad_(False)          # 全部冻结，只有 z 可训
model.eval()

# ---------- 1. 频段掩码 ----------
NBINS, FS = 129, 48000
freqs = np.arange(NBINS) * FS / 256.0
band = {
    "overall":  np.ones(NBINS, bool),
    "low<5k":   freqs < 5000,
    "notch6-12k": (freqs >= 6000) & (freqs <= 12000),
    "high>12k": freqs > 12000,
}

def rmse(a, b, fmask=None):
    """a,b: (B,129,3072) tensors in dB"""
    d = (a - b) ** 2
    if fmask is not None:
        d = d[:, torch.as_tensor(fmask, device=d.device), :]
    return torch.sqrt(d.mean()).item()

# ---------- 2. 核心：对 z 做 test-time 优化 ----------
def optimize_z(x, z_init, steps=400, lr=3e-3, wd=0.0, log_every=20):
    """x: (B,129,3072) 目标；z_init: (B,129,64) 起点。返回 (z_opt, 曲线)"""
    z = z_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    curve = []
    for s in range(steps + 1):
        rec = decoder(z)
        loss = ((rec - x) ** 2).mean()
        if wd:
            loss = loss + wd * (z ** 2).mean()
        if s % log_every == 0:
            curve.append((s, torch.sqrt(((rec - x) ** 2).mean()).item()))
        if s == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach(), curve

# ---------- 3. 在验证集上跑 ----------
N_EARS = 16          # 用多少只耳朵（够统计即可，想快就调小）
STEPS = 400

xs = []
for L, R, _ in val_dataset:
    xs.append(L); xs.append(R)
    if len(xs) >= N_EARS:
        break
X = torch.stack(xs[:N_EARS]).to(device)        # (N,129,3072)
print(f"\n评估 {X.shape[0]} 只耳朵, shape={tuple(X.shape)}\n")

# --- 3a. encoder 的 z 和它的重建 ---
with torch.no_grad():
    rec_enc, z_enc = model(X)
print("encoder latent shape:", tuple(z_enc.shape))

# --- 3b. 从 encoder 出发优化（= hybrid / refinement）---
z_warm, curve_warm = optimize_z(X, z_enc, steps=STEPS)
with torch.no_grad():
    rec_warm = decoder(z_warm)

# --- 3c. 冷启动：从零开始（= 纯 auto-decoder，完全不用 encoder）---
z_cold_init = torch.zeros_like(z_enc)
z_cold, curve_cold = optimize_z(X, z_cold_init, steps=STEPS)
with torch.no_grad():
    rec_cold = decoder(z_cold)

# --- 3d. mean-HRTF baseline ---
mean_hrtf = X.mean(0, keepdim=True).expand_as(X)

# ---------- 4. 结果表 ----------
print("=" * 78)
print(f"{'频段':<14}{'mean baseline':>14}{'encoder':>10}{'warm-opt':>10}{'cold-opt':>10}{'gap':>8}")
print("-" * 78)
rows = {}
for name, m in band.items():
    b0 = rmse(mean_hrtf, X, m)
    e  = rmse(rec_enc,  X, m)
    w  = rmse(rec_warm, X, m)
    c  = rmse(rec_cold, X, m)
    rows[name] = (b0, e, w, c)
    print(f"{name:<14}{b0:>14.3f}{e:>10.3f}{w:>10.3f}{c:>10.3f}{e-w:>8.3f}")
print("=" * 78)

# ---------- 5. z 移动了多远 ----------
dz = (z_warm - z_enc).flatten(1).norm(dim=1)
z0 = z_enc.flatten(1).norm(dim=1)
cos_wc = torch.nn.functional.cosine_similarity(
    z_warm.flatten(1), z_cold.flatten(1), dim=1)
print(f"\n||z_opt - z_enc|| / ||z_enc||  : {(dz/z0).mean():.3f}  "
      f"(min {(dz/z0).min():.3f}, max {(dz/z0).max():.3f})")
print(f"cos(z_warm, z_cold)            : {cos_wc.mean():.3f}   "
      "<- 接近 1 说明两条路收敛到同一个解")
print(f"||z_enc|| 平均                  : {z0.mean():.2f}")
print(f"||z_cold|| 平均                 : {z_cold.flatten(1).norm(dim=1).mean():.2f}")

# ---------- 6. 曲线 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(*zip(*curve_warm), label="warm start (from encoder)", lw=2)
ax[0].plot(*zip(*curve_cold), label="cold start (pure auto-decoder)", lw=2, ls="--")
ax[0].axhline(rows["overall"][1], color="crimson", ls=":", label="encoder (no optim)")
ax[0].set_xlabel("optimization step"); ax[0].set_ylabel("RMSE (dB)")
ax[0].set_title("test-time optimization of z"); ax[0].legend(); ax[0].grid(alpha=.3)

names = list(rows)
xpos = np.arange(len(names)); w_ = 0.2
for i, (lbl, j) in enumerate([("baseline", 0), ("encoder", 1),
                              ("warm-opt", 2), ("cold-opt", 3)]):
    ax[1].bar(xpos + i * w_, [rows[n][j] for n in names], w_, label=lbl)
ax[1].set_xticks(xpos + 1.5 * w_); ax[1].set_xticklabels(names, fontsize=9)
ax[1].set_ylabel("RMSE (dB)"); ax[1].set_title("per band"); ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

# ---------- 7. 自动解读 ----------
gap = rows["overall"][1] - rows["overall"][2]
gap_notch = rows["notch6-12k"][1] - rows["notch6-12k"][2]
cold_vs_warm = rows["overall"][3] - rows["overall"][2]
print("\n" + "=" * 78)
print("解读：")
if gap < 0.15:
    print(f"  Q1 amortization gap = {gap:.3f} dB —— 很小。encoder 已经接近最优，")
    print("     换 auto-decoder 在重建上换不来什么。保留 encoder。")
else:
    print(f"  Q1 amortization gap = {gap:.3f} dB —— 显著。encoder 在白白丢精度，")
    print("     auto-decoder / hybrid 值得做。")
print(f"     notch 带的 gap = {gap_notch:.3f} dB")

if rows['overall'][2] < rows['overall'][1] - 0.5:
    print(f"  Q2 优化后 RMSE 降到 {rows['overall'][2]:.3f}，说明 64 维/频点 本来就装得下更多，")
    print("     瓶颈是 encoder 的提取能力，不是维度。→ 加大 latent 不是解药。")
else:
    print(f"  Q2 优化到收敛也只到 {rows['overall'][2]:.3f} dB —— 这是 decoder 的容量上限，")
    print("     不是 encoder 的锅。→ 要改的是 decoder / 维度分配。")

if abs(cold_vs_warm) < 0.15:
    print(f"  Q3 冷启动几乎追平热启动（差 {cold_vs_warm:+.3f} dB）→ 纯 auto-decoder 可行，")
    print("     encoder 不是必需品。")
else:
    print(f"  Q3 冷启动比热启动差 {cold_vs_warm:+.3f} dB → 优化面有坏局部极小，")
    print("     encoder 作为初始化器有真实价值（hybrid 是对的路）。")
print("=" * 78)
