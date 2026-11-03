# ==========================================================================
# 频率维度分配的天花板测试 v2 —— 修正 v1 的学习率尺度 bug
#
# v1 错在哪：z = mu + c @ V，c 是【未归一化】的 PCA 系数（自然尺度 = 奇异值，
#            几十到上百），却用 lr=0.1 跑 200 步，最多走到 20 —— 没走到。
#            uniform-64 应当复现 v2 的 warm-opt (2.158)，实测 8.912 = 没动。
#
# v2 的修正：
#   1) 系数按训练集的 per-component std 归一化 -> c 是 O(1)，lr 尺度正确。
#   2) ★ 内置自检：uniform-64 必须复现无约束天花板，否则直接报警。
#   3) 报告"每个系数移动了几个 sigma"，让"没优化动"不可能静默通过。
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

STRIDE, STEPS, N_EARS = 4, 300, 8
NBINS, FS = 129, 48000
sel = np.arange(0, NBINS, STRIDE)
freqs = sel * FS / 256.0
F = len(sel)
band = {"low<5k": freqs < 5000, "notch6-12k": (freqs >= 6000) & (freqs <= 12000),
        "high>12k": freqs > 12000, "overall": np.ones(F, bool)}

# ---------------- 逐频点 PCA 基 + 系数尺度 ----------------
Ztr = []
with torch.no_grad():
    for L, R, _ in DataLoader(train_dataset, batch_size=8, shuffle=False):
        for x in (L, R):
            _, z = model(x.to(device))
            Ztr.append(z[:, sel, :].cpu())
Ztr = torch.cat(Ztr, 0).double()
mu = Ztr.mean(0)
V = torch.zeros(F, 64, 64, dtype=torch.float64)
Csig = torch.zeros(F, 64, dtype=torch.float64)
for f in range(F):
    Xf = Ztr[:, f, :] - mu[f]
    _, _, Vt = torch.linalg.svd(Xf, full_matrices=False)
    V[f] = Vt
    Csig[f] = (Xf @ Vt.T).std(0)                  # ★ 每个主成分系数的实际尺度
Csig = Csig.clamp_min(1e-6)
print(f"PCA 系数尺度: 中位 {Csig.median():.1f}, max {Csig.max():.1f} "
      f"<- v1 用 lr=0.1 跑 300 步只能走到 ~30，这就是没动的原因")
mu, V, Csig = mu.float().to(device), V.float().to(device), Csig.float().to(device)

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

# ---------------- 在归一化系数空间里优化 ----------------
def ceiling(dprof, steps=STEPS, lr=0.1, report=""):
    mask = torch.zeros(F, 64, device=device)
    for f, d in enumerate(dprof):
        mask[f, :int(d)] = 1.0
    c = torch.zeros(X.shape[0], F, 64, device=device, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        z = mu + torch.einsum("bfd,fde->bfe", c * mask * Csig, V)   # ★ 乘回尺度
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((decoder(z).float() - X) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        z = mu + torch.einsum("bfd,fde->bfe", c * mask * Csig, V)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec = decoder(z).float()
        moved = (c * mask).abs().sum() / mask.sum() / X.shape[0]
    out = {k: rmse(rec, X, m) for k, m in band.items()}
    out["moved_sigma"] = float(moved)
    return out

# ---------------- ★ 自检：uniform-64 必须复现无约束天花板 ----------------
print("\n自检: uniform-64 (无约束) ...")
chk = ceiling(np.full(F, 64))
print(f"  overall = {chk['overall']:.3f} dB, 系数平均移动 {chk['moved_sigma']:.2f} sigma")
print(f"  参考: v2 的 warm-opt = 2.158 dB")
if chk["overall"] > 3.0 or chk["moved_sigma"] < 0.05:
    print("  ★★ 自检失败：优化没走到。加大 lr 或 STEPS 后重跑，别信下面的数。")
else:
    print("  自检通过 ✓")

# ---------------- 分配方案 ----------------
BUDGET = 16 * F
need = np.interp(freqs, [0, 5000, 6000, 12000, 16000, 24000],
                 [4, 4, 10, 16, 24, 24])
prop = np.minimum(np.maximum(2, np.round(need / need.sum() * BUDGET)), 64).astype(int)

profiles = {"uniform 8": np.full(F, 8), "uniform 16": np.full(F, 16),
            "uniform 32": np.full(F, 32), "uniform 64 (unconstrained)": np.full(F, 64),
            "proportional (= uniform16 budget)": prop}
print(f"\n预算: uniform16={16*F}, proportional={prop.sum()}")
print(f"proportional: 低频 {prop[freqs<5000].mean():.0f}, "
      f"notch {prop[(freqs>=6000)&(freqs<=12000)].mean():.0f}, "
      f"高频 {prop[freqs>12000].mean():.0f} 维\n")

res = {}
print(f"{'方案':<36}{'overall':>9}{'low':>8}{'notch':>8}{'high':>8}{'moved':>8}")
print("-" * 78)
for name, prof in profiles.items():
    r = res[name] = ceiling(prof)
    print(f"{name:<36}{r['overall']:>9.3f}{r['low<5k']:>8.3f}"
          f"{r['notch6-12k']:>8.3f}{r['high>12k']:>8.3f}{r['moved_sigma']:>8.2f}")

# ---------------- 图 + 判据 ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(freqs/1000, np.full(F, 16), label="uniform 16", lw=2)
ax[0].plot(freqs/1000, prop, label="proportional", lw=2, ls="--")
ax[0].axvspan(6, 12, color="orange", alpha=.12)
ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("dims allocated")
ax[0].set_title("allocation (same budget)"); ax[0].legend(); ax[0].grid(alpha=.3)
bands = ["low<5k", "notch6-12k", "high>12k"]
xp = np.arange(3); w = 0.8/len(res)
for i, n in enumerate(res):
    ax[1].bar(xp + i*w, [res[n][b] for b in bands], w, label=n)
ax[1].set_xticks(xp + 0.4); ax[1].set_xticklabels(bands)
ax[1].set_ylabel("ceiling RMSE (dB)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

u16, pr = res["uniform 16"], res["proportional (= uniform16 budget)"]
print("\n" + "="*78)
print(f"同预算重分配: notch {u16['notch6-12k']-pr['notch6-12k']:+.3f} dB, "
      f"低频代价 {u16['low<5k']-pr['low<5k']:+.3f} dB")
print(f"维度本身的收益: uniform16 -> uniform64 = "
      f"{u16['overall']-res['uniform 64 (unconstrained)']['overall']:+.3f} dB")
print("="*78)
