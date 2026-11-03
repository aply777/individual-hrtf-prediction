# ==========================================================================
# 诊断 — 分频段误差 + amortization gap
#
# 问的是"天花板在哪一层"：encoder 没编好？decoder 解不出来？还是 latent 不够？
# 在已有 checkpoint 上跑，不重训。
#
# ⚠️ latent 的优化【必须】在 per-dim std 的单位里做。PROJECT_LOG §5 记录 1：
#    上一次用 lr=3e-3 而 latent 每维幅度 ≈81，优化器只挪了 1.5%，三条结论全是
#    假象。这里把参数化成 z = z0 + delta * std，lr 的单位就是"多少个 std"，
#    并且打印 ‖Δz‖/‖z‖ 让"到底动了没有"直接可见。
#    §5 记录 2：任何新参数化都要有一个【已知答案的自检】。见 [2] 的第一行。
# ==========================================================================
import numpy as np
import torch

BANDS = {'DC+low <500': (0, 500), '0.5-2k': (500, 2000), '2-6k': (2000, 6000),
         'notch 6-12k': (6000, 12000), '12-16k': (12000, 16000)}
MASKS = {k: torch.tensor((freqs >= lo) & (freqs < hi)).to(device)
         for k, (lo, hi) in BANDS.items()}
OPT_STEPS, LR_Z, BS = 300, 0.05, 8        # LR_Z 的单位是"每维 std"

model.eval()
DID0 = DATASETS[0]['id']


def band_rmse(pred, tgt):
    d = (pred - tgt) ** 2
    out = {k: d[:, m, :].mean().item() for k, m in MASKS.items()}
    out['overall'] = d[:, band_dev, :].mean().item()
    return out


def acc(store, new, n):
    for k, v in new.items():
        store[k] = store.get(k, 0.0) + v * n


# ---------- 收集 val 张量 + train 的 latent 统计 ----------
Xva = torch.stack([va_parts[0][i][0] for i in range(len(va_parts[0]))])
Dva = torch.full((len(Xva),), DID0, dtype=torch.long)
print(f'val {len(Xva)} 只耳 | train {len(tr_parts[0])} 只耳')

with torch.no_grad():
    mu_tr = torch.stack([tr_parts[0][i][0] for i in range(len(tr_parts[0]))]).mean(0)
    zs = []
    for i in range(0, len(tr_parts[0]), BS):
        xb = torch.stack([tr_parts[0][j][0]
                          for j in range(i, min(i + BS, len(tr_parts[0])))]).to(device)
        db = torch.full((len(xb),), DID0, dtype=torch.long, device=device)
        zs.append(model(xb, db)[1].cpu())
    Ztr = torch.cat(zs)                       # (N, F, 64)
z_mean = Ztr.mean(0).to(device)               # (F, 64)
z_std = Ztr.std(0).clamp_min(1e-6).to(device)
print(f'train latent: 形状 {tuple(Ztr.shape)} | 每维 std 中位数 {Ztr.std(0).median():.3f} '
      f'| 跨维跨度 {(Ztr.std(0).max()/Ztr.std(0).min()):.0f}x')
del zs, Ztr

# ==========================================================================
# [1] 分频段：模型 vs mean-HRTF 基线
# ==========================================================================
se_enc, se_base, n_tot = {}, {}, 0
with torch.no_grad():
    for i in range(0, len(Xva), BS):
        xb = Xva[i:i + BS].to(device); db = Dva[i:i + BS].to(device)
        rec, _ = model(xb, db)
        acc(se_enc, band_rmse(rec, xb), len(xb))
        acc(se_base, band_rmse(mu_tr.to(device).expand_as(xb), xb), len(xb))
        n_tot += len(xb)

print('\n[1] 分频段 RMSE (dB)')
print(f"    {'band':<14}{'model':>9}{'baseline':>10}{'gain':>9}{'残差/基线':>11}")
for k in list(BANDS) + ['overall']:
    m, b = (se_enc[k] / n_tot) ** .5, (se_base[k] / n_tot) ** .5
    print(f'    {k:<14}{m:>9.3f}{b:>10.3f}{b-m:>+9.3f}{m/b:>11.1%}')

# ==========================================================================
# [2] Amortization gap：冻结 decoder，直接优化 latent
# ==========================================================================
for p in model.parameters():
    p.requires_grad_(False)


def optimise_latent(xb, db, z_init, steps=OPT_STEPS):
    """z = z_init + delta * z_std，delta 从 0 开始。返回 (最终重建, ‖Δz‖/‖z‖)"""
    delta = torch.zeros_like(z_init, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR_Z)
    for _ in range(steps):
        opt.zero_grad()
        out = model.decoder(z_init + delta * z_std, db)
        ((out - xb)[:, band_dev, :] ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        z = z_init + delta * z_std
        rel = ((z - z_init).norm() / z_init.norm()).item()
        return model.decoder(z, db), rel


se_warm, se_cold, rel_w, rel_c, checked = {}, {}, [], [], False
for i in range(0, len(Xva), BS):
    xb = Xva[i:i + BS].to(device); db = Dva[i:i + BS].to(device)
    with torch.no_grad():
        rec_enc, z0 = model(xb, db)

    # ★ 自检（已知答案）：delta=0 时 decoder(z0) 必须和 model 的前向逐位相同
    if not checked:
        with torch.no_grad():
            e = (model.decoder(z0, db) - rec_enc).abs().max().item()
        print(f'\n[2] 自检: delta=0 时与 encoder 前向的最大差 = {e:.2e}   '
              f'{"✓" if e < 1e-4 else "★ 不一致，下面的数字不可信"}')
        checked = True

    rw, r1 = optimise_latent(xb, db, z0)
    rc, r2 = optimise_latent(xb, db, z_mean.unsqueeze(0).expand_as(z0).contiguous())
    with torch.no_grad():
        acc(se_warm, band_rmse(rw, xb), len(xb))
        acc(se_cold, band_rmse(rc, xb), len(xb))
    rel_w.append(r1); rel_c.append(r2)

enc = {k: (se_enc[k] / n_tot) ** .5 for k in se_enc}
warm = {k: (se_warm[k] / n_tot) ** .5 for k in se_warm}
cold = {k: (se_cold[k] / n_tot) ** .5 for k in se_cold}

print(f'    优化了 {OPT_STEPS} 步, lr={LR_Z} std/步')
print(f'    ‖Δz‖/‖z‖  warm {np.mean(rel_w):.3f} | cold {np.mean(rel_c):.3f}'
      f'   <- 若接近 0 说明优化器没动，结论无效')
print(f"\n    {'band':<14}{'encoder':>9}{'warm-opt':>10}{'cold-opt':>10}{'gap':>8}")
for k in list(BANDS) + ['overall']:
    print(f'    {k:<14}{enc[k]:>9.3f}{warm[k]:>10.3f}{cold[k]:>10.3f}'
          f'{enc[k]-warm[k]:>+8.3f}')

g = enc['overall'] - warm['overall']
print(f"\n  gap = {g:.3f} dB ({g/enc['overall']:.1%})")
print('  gap 大  -> encoder 是瓶颈，值得改 encoder / 换 auto-decoder')
print('  gap 小  -> encoder 已接近它能达到的最好，天花板在 decoder + latent 维度')
print(f"  cold 与 warm 的差 {abs(cold['overall']-warm['overall']):.3f} dB "
      '-> 接近 0 说明 encoder 只提供了一个好的初值，没有额外信息')

for p in model.parameters():
    p.requires_grad_(True)
