# ==========================================================================
# Test-time optimization  v2  —— 修正 v1 的学习率 bug
#
# v1 的错误：latent 每分量量级 ~81，却用 Adam lr=3e-3 优化 400 步，
#            最多移动 1.2/分量 = 1.5%，等于没优化。三个结论全部无效。
#
# v2 的修正：
#   1) 先诊断 latent 尺度，在【标准化空间】里优化：z = z_init + sigma * delta，
#      sigma = 各维实测 std。这样 Adam 的自然步长与数据尺度匹配。
#   2) 自动 lr 扫描（短跑 40 步选最优），不再靠猜。
#   3) cold start 从【训练集平均码】出发（DeepSDF 的公平类比），
#      而不是从 0 —— 从 0 出发在这个尺度下根本走不到。
#   4) 频率 stride 抽样 + bf16（精确加速，见注释）。
# ==========================================================================
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

N_EARS, STRIDE, STEPS = 8, 4, 300
device = next(model.parameters()).device
encoder, decoder = model.encoder, model.decoder
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

NBINS, FS = 129, 48000
sel = np.arange(0, NBINS, STRIDE)              # 4j/128 == j/32，与 get_freq_embedding 对齐
assert sel[-1] == NBINS - 1
freqs = sel * FS / 256.0
band = {"overall": np.ones(len(sel), bool), "low<5k": freqs < 5000,
        "notch6-12k": (freqs >= 6000) & (freqs <= 12000), "high>12k": freqs > 12000}

def rmse(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        d = d[:, torch.as_tensor(m, device=d.device), :]
    return torch.sqrt(d.mean()).item()

# ---------------- 数据 + 训练集平均码 ----------------
xs = []
for L, R, _ in val_dataset:
    xs += [L, R]
    if len(xs) >= N_EARS:
        break
X = torch.stack(xs[:N_EARS]).to(device)[:, sel, :].contiguous()

with torch.no_grad():
    _, z_enc = model(torch.stack(xs[:N_EARS]).to(device))
z_enc = z_enc[:, sel, :].contiguous()

# 训练集平均码（cold start 的公平起点）
tr = []
for i, (L, R, _) in enumerate(train_dataset):
    tr += [L, R]
    if len(tr) >= 32:
        break
with torch.no_grad():
    _, z_tr = model(torch.stack(tr).to(device))
z_tr = z_tr[:, sel, :]
z_mean = z_tr.mean(0, keepdim=True).expand_as(z_enc).contiguous()

# ---------------- ★ 诊断 latent 尺度 ----------------
sigma = z_tr.std(0, keepdim=True).clamp_min(1e-6)        # (1,F,64) 各维 std
print("=" * 70)
print("LATENT 尺度诊断")
print(f"  per-component  mean |z| : {z_enc.abs().mean():.2f}")
print(f"  per-component  std      : {sigma.mean():.2f}  "
      f"(min {sigma.min():.3f}, max {sigma.max():.2f})")
print(f"  ||z|| per ear           : {z_enc.flatten(1).norm(dim=1).mean():.1f}")
print(f"  std 的动态范围 (max/min): {(sigma.max()/sigma.min()):.0f}x  "
      "<- 很大的话下游回归前必须标准化")
print("=" * 70)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    rec_enc = decoder(z_enc).float()

# ---------------- 在标准化空间里优化 ----------------
def optimize(z_init, steps, lr, quiet=False):
    """z = z_init + sigma * delta，优化 delta。lr 现在是"以 sigma 为单位"的步长。"""
    delta = torch.zeros_like(z_init, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for s in range(steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((decoder(z_init + sigma * delta).float() - X) ** 2).mean()
        if s % 10 == 0:
            curve.append((s, float(loss.detach().sqrt())))
        if s == steps:
            break
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return (z_init + sigma * delta).detach(), curve

# ---- lr 扫描：短跑 40 步，挑最好的 ----
print("\nlr 扫描 (40 步):")
best_lr, best_v = None, 1e9
for lr in (0.03, 0.1, 0.3, 1.0, 3.0):
    _, c = optimize(z_enc, 40, lr)
    print(f"  lr={lr:<5} -> RMSE {c[-1][1]:.4f}")
    if c[-1][1] < best_v:
        best_lr, best_v = lr, c[-1][1]
print(f"  选用 lr={best_lr}")

t0 = time.time()
z_warm, cw = optimize(z_enc,  STEPS, best_lr)
z_cold, cc = optimize(z_mean, STEPS, best_lr)
print(f"正式优化用时 {time.time()-t0:.0f}s")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    rec_warm, rec_cold = decoder(z_warm).float(), decoder(z_cold).float()
mean_hrtf = X.mean(0, keepdim=True).expand_as(X)

# 收敛检查
tail = cw[-max(2, len(cw)//5):]
drift = tail[0][1] - tail[-1][1]
print(f"收敛检查: warm 末段仍下降 {drift:.4f} dB "
      f"({'已收敛' if drift < 0.02 else '★未收敛，加大 STEPS'})")

# ---------------- 结果 ----------------
print("=" * 78)
print(f"{'频段':<14}{'mean base':>12}{'encoder':>10}{'warm-opt':>10}{'cold-opt':>10}{'gap':>9}")
print("-" * 78)
rows = {}
for n, m in band.items():
    r = (rmse(mean_hrtf, X, m), rmse(rec_enc, X, m), rmse(rec_warm, X, m), rmse(rec_cold, X, m))
    rows[n] = r
    print(f"{n:<14}{r[0]:>12.3f}{r[1]:>10.3f}{r[2]:>10.3f}{r[3]:>10.3f}{r[1]-r[2]:>9.3f}")
print("=" * 78)

dz = ((z_warm - z_enc) / sigma).flatten(1).norm(dim=1) / np.sqrt(z_enc[0].numel())
print(f"\n每维移动了 {dz.mean():.2f} 个 sigma   <- 远大于 0 才说明优化真的在动")
print(f"||Δz||/||z_enc||     : {((z_warm-z_enc).flatten(1).norm(dim=1)/z_enc.flatten(1).norm(dim=1)).mean():.4f}")
cos = torch.nn.functional.cosine_similarity(z_warm.flatten(1), z_cold.flatten(1), dim=1)
print(f"cos(z_warm, z_cold)  : {cos.mean():.3f}")

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(*zip(*cw), lw=2, label="warm (from encoder)")
ax[0].plot(*zip(*cc), lw=2, ls="--", label="cold (from mean code)")
ax[0].axhline(rows["overall"][1], color="crimson", ls=":", label="encoder, no optim")
ax[0].set_xlabel("step"); ax[0].set_ylabel("RMSE (dB)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title(f"test-time optimization (lr={best_lr}, normalized space)")
names = list(rows); xp = np.arange(len(names)); w = 0.2
for i, (lbl, j) in enumerate([("baseline", 0), ("encoder", 1), ("warm", 2), ("cold", 3)]):
    ax[1].bar(xp + i*w, [rows[n][j] for n in names], w, label=lbl)
ax[1].set_xticks(xp + 1.5*w); ax[1].set_xticklabels(names, fontsize=9)
ax[1].set_ylabel("RMSE (dB)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

gap = rows["overall"][1] - rows["overall"][2]
print(f"\namortization gap = {gap:.3f} dB   (notch 带 "
      f"{rows['notch6-12k'][1]-rows['notch6-12k'][2]:.3f} dB)")
print(f"cold - warm      = {rows['overall'][3]-rows['overall'][2]:+.3f} dB")
