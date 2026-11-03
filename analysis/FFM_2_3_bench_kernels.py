# ==========================================================================
# 定位 1136 ms 花在哪一步
#
# 已知：DataLoader 8 ms、GPU 前向反向 1136 ms、显存峰值 12.1 GB / 40 GB。
#       算力利用率 ~0.2% —— 不是卡不够快，是某个 kernel 在做低效的事。
#
# 嫌疑排序：
#   1. OSLOConv 的 gather 反向 = scatter-add = 大量原子加（最慢的一类操作）
#   2. einsum('bcqn,ocq->bon') 内部可能对 2.6 GB 张量做 permute
#   3. 两个 Linear(98304, 512)（这两个是正常 GEMM，应该很快）
#
# 跑完看哪一行最大，再决定动代码还是换卡。
# ==========================================================================
import time
import torch

sync = lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None
AMP = dict(device_type='cuda', dtype=torch.bfloat16,
           enabled=config['amp'] and torch.cuda.is_available())

B, F_, NP = config['batch_size'], config['num_bins'], 3072
BF = B * F_
print(f'B={B}  F={F_}  ->  B*F={BF} 行, 每行 {NP} 像素\n')


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync(); t = time.time()
    for _ in range(n):
        fn()
    sync(); return (time.time() - t) / n * 1000


# ---------------- 最大的一个 OSLOConv: enc2.conv2 (32->32) ----------------
conv = model.encoder.enc2.conv2
xt = torch.randn(BF, 32, NP, device=device, requires_grad=True)
nb = conv.neighborhood
w = conv.weight

def gather_only():
    pad = torch.zeros(BF, 32, 1, device=device, dtype=xt.dtype)
    return torch.cat([xt.detach(), pad], dim=2)[:, :, nb]

g = gather_only().detach()
gb = g.to(torch.bfloat16)

def einsum_only():
    with torch.autocast(**AMP):
        return torch.einsum('bcqn,ocq->bon', g, w)

def matmul_alt():
    """不用 einsum：view 成 (b, c*q, n) 后直接 matmul，避开可能的 permute"""
    with torch.autocast(**AMP):
        return torch.matmul(w.reshape(w.shape[0], -1), g.reshape(BF, 32 * 9, NP))

def conv_fwd():
    with torch.autocast(**AMP):
        return conv(xt)

def conv_fwdbwd():
    with torch.autocast(**AMP):
        y = conv(xt)
    y.float().sum().backward()
    xt.grad = None

lin = model.encoder.fc_encode
h = torch.randn(BF, 32, NP, device=device, requires_grad=True)

def lin_fwd():
    with torch.autocast(**AMP):
        return lin(h)

def lin_fwdbwd():
    with torch.autocast(**AMP):
        y = lin(h)
    y.float().sum().backward()
    h.grad = None

rows = [
    ('gather 单独（前向）',            gather_only),
    ('einsum 单独（前向）',            einsum_only),
    ('matmul 替代 einsum（前向）',     matmul_alt),
    ('OSLOConv 32->32 前向',           conv_fwd),
    ('OSLOConv 32->32 前向+反向',      conv_fwdbwd),
    ('fc_encode Linear 前向',          lin_fwd),
    ('fc_encode Linear 前向+反向',     lin_fwdbwd),
]
print(f"  {'部件':<30}{'ms':>9}{'占 1136 ms':>12}")
res = {}
for name, fn in rows:
    try:
        ms = bench(fn)
        res[name] = ms
        print(f'  {name:<30}{ms:>9.1f}{ms/1136:>11.1%}')
    except RuntimeError as e:
        print(f'  {name:<30}   跳过: {str(e)[:44]}')

print(f'\n  gather 张量: {g.numel()*4/1e9:.2f} GB (fp32) / '
      f'{g.numel()*2/1e9:.2f} GB (bf16)')
print(f'  模型里共 7 个 OSLOConv，其中 2 个是 C_in=32 这一档')

if 'OSLOConv 32->32 前向+反向' in res and 'OSLOConv 32->32 前向' in res:
    f_, fb = res['OSLOConv 32->32 前向'], res['OSLOConv 32->32 前向+反向']
    print(f'\n  反向 / 前向 = {(fb-f_)/f_:.1f}x'
          '   （正常应 ~2x；远大于 2 就是 scatter-add 的原子加在拖后腿）')
if 'einsum 单独（前向）' in res and 'matmul 替代 einsum（前向）' in res:
    e_, m_ = res['einsum 单独（前向）'], res['matmul 替代 einsum（前向）']
    print(f'  einsum / matmul = {e_/m_:.2f}x'
          f'   （>1.3 说明 einsum 在内部多做了一次 permute，值得替换）')
