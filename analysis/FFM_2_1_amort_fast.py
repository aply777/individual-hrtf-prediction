# ==========================================================================
# Amortization gap（快速版）—— 接在 diagnostics 的 [1] 之后，单独跑
#
# 相对上一版的三处提速：
#   1. 优化过程套 bf16 autocast（上一版是 fp32，白白慢 2 倍）
#   2. BS 8 -> 22：decoder-only 不需要 encoder 激活，也不更新那 1 亿参数，
#      显存宽裕；44 只耳两个 batch 搞定，GPU 利用率高得多
#   3. 300 -> 200 步，每 50 步打印一次，能自己看收敛没收敛
#
# ★ 精度：优化过程用 bf16（只影响轨迹），最终评估强制 fp32（报出来的数字准确）
# ★ latent 优化在 per-dim std 单位里做 —— PROJECT_LOG §5 记录 1
# ★ 自检对已知答案：delta=0 时必须与 encoder 前向逐位相同 —— §5 记录 2
# ==========================================================================
import time
import numpy as np
import torch

BS_OPT, OPT_STEPS, LR_Z = 22, 200, 0.05
DO_COLD = True          # 关掉可再省一半时间（但就看不出 encoder 是否只提供初值）
DID0 = DATASETS[0]['id']
AMP = dict(device_type='cuda', dtype=torch.bfloat16,
           enabled=config['amp'] and torch.cuda.is_available())

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

MASKS = {'notch 6-12k': torch.tensor((freqs >= 6000) & (freqs < 12000)).to(device)}
Xva = torch.stack([va_parts[0][i][0] for i in range(len(va_parts[0]))])
print(f'val {len(Xva)} 只耳 | batch {BS_OPT} -> {int(np.ceil(len(Xva)/BS_OPT))} 个 batch '
      f'| 每个 batch {OPT_STEPS} 步 x {"2" if DO_COLD else "1"} 次')

# train latent 的每维 std（优化的单位）与人群均值（cold 起点）
with torch.no_grad():
    zs = []
    for i in range(0, len(tr_parts[0]), 8):
        xb = torch.stack([tr_parts[0][j][0]
                          for j in range(i, min(i + 8, len(tr_parts[0])))]).to(device)
        db = torch.full((len(xb),), DID0, dtype=torch.long, device=device)
        with torch.autocast(**AMP):
            zs.append(model(xb, db)[1].float().cpu())
    Z = torch.cat(zs); del zs
z_mean, z_std = Z.mean(0).to(device), Z.std(0).clamp_min(1e-6).to(device)
del Z


def se(pred, tgt):
    d = (pred - tgt) ** 2
    out = {'overall': (d[:, band_dev, :].sum().item(), d[:, band_dev, :].numel())}
    for k, m in MASKS.items():
        out[k] = (d[:, m, :].sum().item(), d[:, m, :].numel())
    return out


def optimise(xb, db, z_init, tag):
    delta = torch.zeros_like(z_init, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR_Z)
    for s in range(OPT_STEPS):
        opt.zero_grad()
        with torch.autocast(**AMP):
            out = model.decoder(z_init + delta * z_std, db)
            loss = ((out - xb)[:, band_dev, :] ** 2).mean()
        loss.backward(); opt.step()
        if (s + 1) % 50 == 0:
            print(f'      {tag} step {s+1:>3}/{OPT_STEPS}  rmse {loss.item()**.5:.4f}  '
                  f'({time.time()-t_b:.0f}s)', flush=True)
    with torch.no_grad():                       # ★ 最终评估强制 fp32
        z = z_init + delta * z_std
        return model.decoder(z, db), ((z - z_init).norm() / z_init.norm()).item()


tot = {}
rel_w, rel_c, checked = [], [], False
for bi, i in enumerate(range(0, len(Xva), BS_OPT)):
    t_b = time.time()
    xb = Xva[i:i + BS_OPT].to(device)
    db = torch.full((len(xb),), DID0, dtype=torch.long, device=device)
    print(f'  batch {bi+1} ({len(xb)} 只耳)')
    with torch.no_grad():
        rec_enc, z0 = model(xb, db)
        z0 = z0.float()

    if not checked:
        with torch.no_grad():
            e = (model.decoder(z0, db) - rec_enc).abs().max().item()
        print(f'    自检 delta=0 与 encoder 前向的最大差 = {e:.2e}  '
              f'{"✓" if e < 1e-4 else "★ 不一致，结果不可信"}')
        checked = True

    rw, r1 = optimise(xb, db, z0, 'warm')
    rel_w.append(r1)
    outs = {'encoder': rec_enc, 'warm': rw}
    if DO_COLD:
        rc, r2 = optimise(xb, db, z_mean.unsqueeze(0).expand_as(z0).contiguous(), 'cold')
        rel_c.append(r2); outs['cold'] = rc
    with torch.no_grad():
        for name, o in outs.items():
            for k, (s_, n_) in se(o, xb).items():
                a, b = tot.get((name, k), (0.0, 0))
                tot[(name, k)] = (a + s_, b + n_)
    if bi == 0:
        print(f'    本 batch 用时 {time.time()-t_b:.0f}s，'
              f'预计总计 {(time.time()-t_b)*int(np.ceil(len(Xva)/BS_OPT))/60:.1f} 分钟')

R = {k: (v[0] / v[1]) ** .5 for k, v in tot.items()}
names = ['encoder', 'warm'] + (['cold'] if DO_COLD else [])
print(f"\n  ‖Δz‖/‖z‖  warm {np.mean(rel_w):.3f}"
      + (f' | cold {np.mean(rel_c):.3f}' if DO_COLD else '')
      + '   <- 接近 0 就说明优化器没动，结论无效')
print(f"\n  {'band':<14}" + ''.join(f'{n:>10}' for n in names) + f"{'gap':>9}")
for k in ['overall'] + list(MASKS):
    row = ''.join(f'{R[(n, k)]:>10.3f}' for n in names)
    print(f"  {k:<14}{row}{R[('encoder', k)] - R[('warm', k)]:>+9.3f}")

g = R[('encoder', 'overall')] - R[('warm', 'overall')]
print(f"\n  gap = {g:.3f} dB ({g/R[('encoder','overall')]:.1%})")
print('  gap 大 -> encoder 是瓶颈；gap 小 -> 天花板在 decoder + 64 维/bin')
if DO_COLD:
    print(f"  cold - warm = {R[('cold','overall')] - R[('warm','overall')]:+.3f} dB "
          '-> 接近 0 说明 encoder 只提供了初值，没带额外信息')

for p in model.parameters():
    p.requires_grad_(True)
