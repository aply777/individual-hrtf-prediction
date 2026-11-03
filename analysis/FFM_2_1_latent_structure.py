# ==========================================================================
# latent 的人群结构 —— 下游到底该预测多少个数？
#
# 单人 latent 标称 93x64 = 5952 维，但 174 个训练受试者只能撑起【至多 173 维】
# 的子流形。下游不该回归 5952 个目标，而该回归前 K 个主成分系数。
#
# K 用【截断到 K 维后解码出来的 HRTF 差多少 dB】来定，不用方差解释率 ——
# PROJECT_LOG §2 已记录：方差解释率系统性高估充分性（notch 低方差但关键）。
#
# PCA 在 train 上算，在 val 上评估（held-out），否则 K 会被高估。
# ==========================================================================
import numpy as np
import torch

BS = 8
DID0 = DATASETS[0]['id']
model.eval()


def encode_all(part):
    zs, xs = [], []
    with torch.no_grad():
        for i in range(0, len(part), BS):
            xb = torch.stack([part[j][0] for j in range(i, min(i + BS, len(part)))]).to(device)
            db = torch.full((len(xb),), DID0, dtype=torch.long, device=device)
            zs.append(model(xb, db)[1].cpu())
            xs.append(xb.cpu())
    return torch.cat(zs), torch.cat(xs)


Ztr, _ = encode_all(tr_parts[0])
Zva, Xva = encode_all(va_parts[0])
F_, K_ = Ztr.shape[1], Ztr.shape[2]
n_sub_tr = len(SPLIT[DATASETS[0]['tag']]['train'])

print(f'单人 latent : ({F_}, {K_}) = {F_*K_} 个数')
print(f'训练样本    : {len(Ztr)} 只耳 / {n_sub_tr} 人')
print(f'★ 秩上限    : min({len(Ztr)}-1, {F_*K_}) = {min(len(Ztr)-1, F_*K_)}，'
      f'但左右耳镜像后相关 0.978，【有效】上限接近 {n_sub_tr - 1}')

# ---------------- train 上做 PCA ----------------
A = Ztr.reshape(len(Ztr), -1).double()
mu = A.mean(0, keepdim=True)
U, S, Vt = torch.linalg.svd(A - mu, full_matrices=False)
var = (S ** 2) / (S ** 2).sum()
cum = var.cumsum(0)
print(f'\n方差解释率: 90% 需要 {int((cum<0.90).sum())+1} 维 | '
      f'99% 需要 {int((cum<0.99).sum())+1} 维 | 数值秩 {int((S>S[0]*1e-10).sum())}')
print('  （仅供参考 —— 下面 HRTF 域的数字才是判据）')

# ---------------- val 上：截断到 K 维 -> 解码 -> 量 HRTF 误差 ----------------
B = Zva.reshape(len(Zva), -1).double() - mu
band_cpu = band_dev.cpu()


def decode_rmse(Zmat):
    """Zmat: (N, F*K) 已加回 mu。返回 (overall, notch) RMSE"""
    Z = Zmat.reshape(-1, F_, K_).float()
    se_a = se_n = n_a = n_n = 0.0
    mn = torch.tensor((freqs >= 6000) & (freqs < 12000))
    with torch.no_grad():
        for i in range(0, len(Z), BS):
            zb = Z[i:i + BS].to(device)
            db = torch.full((len(zb),), DID0, dtype=torch.long, device=device)
            d = (model.decoder(zb, db).cpu() - Xva[i:i + BS]) ** 2
            se_a += d[:, band_cpu, :].sum().item(); n_a += d[:, band_cpu, :].numel()
            se_n += d[:, mn, :].sum().item();       n_n += d[:, mn, :].numel()
    return (se_a / n_a) ** .5, (se_n / n_n) ** .5


KS = [1, 2, 4, 8, 16, 32, 64, 128, min(256, len(S)), len(S)]
KS = sorted({k for k in KS if k <= len(S)})

full_a, full_n = decode_rmse(B + mu)                       # 未截断 = encoder 本身
mean_a, mean_n = decode_rmse(mu.expand(len(B), -1))        # K=0，人人都用人群均值

print(f"\n{'K':>6}{'overall':>10}{'notch 6-12k':>13}{'方差解释':>10}"
      f"{'相对 K=0 的收益':>16}")
print(f"{0:>6}{mean_a:>10.3f}{mean_n:>13.3f}{0.0:>10.1%}{'—':>16}")
for k in KS:
    P = Vt[:k]
    rec = (B @ P.T) @ P + mu
    a, n = decode_rmse(rec)
    print(f'{k:>6}{a:>10.3f}{n:>13.3f}{cum[k-1]:>10.1%}'
          f'{(mean_a - a) / (mean_a - full_a):>15.1%}')
print(f"{'full':>6}{full_a:>10.3f}{full_n:>13.3f}{1.0:>10.1%}{1.0:>15.1%}")

print(f'\n读法：找 overall 开始走平的那个 K —— 那就是下游该预测的目标维度。')
print(f'      最右列 = "截断到 K 维能拿回多少 encoder 相对人群均值的收益"。')
print(f'      如果 K=173 附近还没走平，说明是【样本量】不够而不是维度不够，')
print(f'      结论就是"先去要更多人"，而不是改架构。')

# ---------------- 下游要用的东西，存下来 ----------------
K_PICK = 64          # ← 看完表格再改成走平的那个 K
np.savez(os.path.join(config['save_dir'], 'latent_pca.npz'),
         mean=mu.squeeze(0).numpy().astype(np.float32),
         components=Vt[:K_PICK].numpy().astype(np.float32),
         sing=S[:K_PICK].numpy().astype(np.float32),
         F=F_, K=K_, K_pick=K_PICK,
         train_ids=np.array(SPLIT[DATASETS[0]['tag']]['train']))
print(f'\nPCA 基已存 (K={K_PICK}) -> latent_pca.npz')
print('下游用法:  mesh -> 网络 -> c (K维) -> z = mean + c @ components -> '
      'reshape(93,64) -> model.decoder(z, did)')
