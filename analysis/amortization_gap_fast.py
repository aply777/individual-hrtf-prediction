# ==========================================================================
# Test-time optimization —— 快速版（~30-60s on A100，原版 ~30min）
#
# 三处加速，前两处是精确的、不损失有效性：
#   1) 频率抽样 stride=4 (129 -> 33 bins)。模型是频率隔离的，频点之间无耦合，
#      所以这对被选中的频点是【精确】的。且 4j/128 == j/32，与
#      get_freq_embedding(33) 给出的归一化频率完全一致，频率条件化不被破坏。
#   2) bf16 autocast。
#   3) 步数 400 -> 250 + 学习率衰减（这一处是近似，会检查是否收敛）。
# ==========================================================================
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------- 配置 -----------------------------
N_EARS   = 8      # 想更快就 4
STEPS    = 250
LR       = 5e-3
STRIDE   = 4      # 必须能整除 128，才能保证频率嵌入对齐 (4/8/16 都行)
# ---------------------------------------------------------------

print("model children:", [n for n, _ in model.named_children()])

def _find(m, keys):
    for k in keys:
        s = getattr(m, k, None)
        if isinstance(s, nn.Module):
            return s
    for n, s in m.named_children():
        if any(k in n.lower() for k in keys):
            return s
    return None

decoder = _find(model, ["decoder", "dec"])
assert decoder is not None, "没找到 decoder，请把上面打印的名字发我"
print("decoder ->", type(decoder).__name__)

for p in model.parameters():
    p.requires_grad_(False)
model.eval()

NBINS, FS = 129, 48000
sel = np.arange(0, NBINS, STRIDE)                 # 0,4,...,128
assert sel[-1] == NBINS - 1, "STRIDE 必须能整除 128"
freqs = sel * FS / 256.0
print(f"频点: {NBINS} -> {len(sel)} (stride={STRIDE})，频率嵌入精确对齐")

band = {"overall": np.ones(len(sel), bool),
        "low<5k": freqs < 5000,
        "notch6-12k": (freqs >= 6000) & (freqs <= 12000),
        "high>12k": freqs > 12000}

def rmse(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        d = d[:, torch.as_tensor(m, device=d.device), :]
    return torch.sqrt(d.mean()).item()

# ----------------------------- 数据 -----------------------------
xs = []
for L, R, _ in val_dataset:
    xs += [L, R]
    if len(xs) >= N_EARS:
        break
X_full = torch.stack(xs[:N_EARS]).to(device)          # (N,129,3072)
X = X_full[:, sel, :].contiguous()                    # (N,33,3072)
print(f"评估 {X.shape[0]} 只耳朵, shape={tuple(X.shape)}")

with torch.no_grad():
    _, z_full = model(X_full)
z_enc = z_full[:, sel, :].contiguous()                # (N,33,64)
print("latent:", tuple(z_enc.shape))

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    rec_enc = decoder(z_enc).float()

# --------------------- test-time optimization ---------------------
def optimize_z(x, z0, steps=STEPS, lr=LR, tag=""):
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve, t0 = [], time.time()
    for s in range(steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec = decoder(z)
            loss = ((rec.float() - x) ** 2).mean()
        if s % 10 == 0:
            curve.append((s, float(loss.detach().sqrt())))
        if s == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sch.step()
        if s == 4:
            print(f"  [{tag}] 每步 {(time.time()-t0)/5:.2f}s "
                  f"-> 预计 {(time.time()-t0)/5*steps/60:.1f} min")
    print(f"  [{tag}] 用时 {time.time()-t0:.0f}s")
    return z.detach(), curve

print("\nwarm start (from encoder):")
z_warm, cw = optimize_z(X, z_enc, tag="warm")
print("cold start (pure auto-decoder):")
z_cold, cc = optimize_z(X, torch.zeros_like(z_enc), tag="cold")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    rec_warm = decoder(z_warm).float()
    rec_cold = decoder(z_cold).float()
mean_hrtf = X.mean(0, keepdim=True).expand_as(X)

# 收敛性检查：最后 20% 的步数还在降多少
tail = cw[-max(2, len(cw)//5):]
drift = tail[0][1] - tail[-1][1]
print(f"\n收敛检查: warm 曲线最后 20% 仍下降 {drift:.4f} dB "
      f"({'已收敛' if drift < 0.02 else '未完全收敛，可加大 STEPS'})")

# ----------------------------- 结果 -----------------------------
print("=" * 78)
print(f"{'频段':<14}{'mean base':>12}{'encoder':>10}{'warm-opt':>10}{'cold-opt':>10}{'gap':>9}")
print("-" * 78)
rows = {}
for n, m in band.items():
    r = (rmse(mean_hrtf, X, m), rmse(rec_enc, X, m),
         rmse(rec_warm, X, m), rmse(rec_cold, X, m))
    rows[n] = r
    print(f"{n:<14}{r[0]:>12.3f}{r[1]:>10.3f}{r[2]:>10.3f}{r[3]:>10.3f}{r[1]-r[2]:>9.3f}")
print("=" * 78)

dz = (z_warm - z_enc).flatten(1).norm(dim=1) / z_enc.flatten(1).norm(dim=1)
cos = torch.nn.functional.cosine_similarity(z_warm.flatten(1), z_cold.flatten(1), dim=1)
print(f"\n||Δz||/||z_enc||     : {dz.mean():.3f}  (min {dz.min():.3f}, max {dz.max():.3f})")
print(f"cos(z_warm, z_cold)  : {cos.mean():.3f}   <- 接近 1 = 两条路收敛到同一解")

# ----------------------------- 图 -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(*zip(*cw), lw=2, label="warm (from encoder)")
ax[0].plot(*zip(*cc), lw=2, ls="--", label="cold (pure auto-decoder)")
ax[0].axhline(rows["overall"][1], color="crimson", ls=":", label="encoder, no optim")
ax[0].set_xlabel("step"); ax[0].set_ylabel("RMSE (dB)")
ax[0].set_title("test-time optimization of z"); ax[0].legend(); ax[0].grid(alpha=.3)

names = list(rows); xp = np.arange(len(names)); w = 0.2
for i, (lbl, j) in enumerate([("baseline", 0), ("encoder", 1), ("warm", 2), ("cold", 3)]):
    ax[1].bar(xp + i*w, [rows[n][j] for n in names], w, label=lbl)
ax[1].set_xticks(xp + 1.5*w); ax[1].set_xticklabels(names, fontsize=9)
ax[1].set_ylabel("RMSE (dB)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

# ----------------------------- 解读 -----------------------------
gap, gap_n = rows["overall"][1]-rows["overall"][2], rows["notch6-12k"][1]-rows["notch6-12k"][2]
cvw = rows["overall"][3] - rows["overall"][2]
print("\n" + "="*78)
print(f"Q1 amortization gap = {gap:.3f} dB (notch 带 {gap_n:.3f} dB)")
print("   " + ("很小 -> encoder 已接近最优，保留它。" if gap < 0.15 else
               "显著 -> encoder 在丢精度，auto-decoder / hybrid 值得做。"))
print(f"Q2 优化后 RMSE = {rows['overall'][2]:.3f} (encoder {rows['overall'][1]:.3f})")
print("   " + ("64 维本来就装得下更多，瓶颈是 encoder 的提取能力 -> 加大 latent 不是解药。"
               if gap > 0.5 else "已接近 decoder 的容量上限 -> 要改的是 decoder / 维度分配。"))
print(f"Q3 cold - warm = {cvw:+.3f} dB")
print("   " + ("冷启动追平 -> 纯 auto-decoder 可行，encoder 非必需。" if abs(cvw) < 0.15 else
               "冷启动更差 -> 优化面有坏局部极小，encoder 作初始化有真实价值 (hybrid)。"))
print("="*78)
