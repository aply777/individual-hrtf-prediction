"""
OSLOConv 的快速版：把反向里的 scatter-add（原子加）换成 gather。

现状（A100 实测，B*F=744, npix=3072, C_in=32）：
    前向      17.2 ms
    前向+反向 260.0 ms      ->  反向是前向的 14.1 倍

原因：`x_pad[:, :, neighborhood]` 是 advanced indexing，它的反向是
`index_put_(accumulate=True)`，即 6.58 亿次原子浮点加，而且 9 个邻居会
写进同一个像素，冲突严重。

修法：HEALPix 的邻居关系是"几乎对称"的 —— 实测 nside=16 时，每个像素被
其它像素指为邻居的次数最多是 9（均值 8.99）。所以可以一次性预计算一张
【逆索引】：对每个像素 m，列出所有指向它的 (q, n) 对。有了它，
grad_x 就能写成一次 gather（大小和前向的 gather 一样），完全不用原子加。
"""
import numpy as np
import torch
import torch.nn as nn


def build_inverse_index(nb, npix):
    """nb: (Q, npix) int，值域 [0, npix]（npix 是 padding 槽）。
    返回 inv: (K, npix)，每个元素是扁平下标 q*npix + n；空位指向 Q*npix（零槽）。"""
    Q = nb.shape[0]
    buckets = [[] for _ in range(npix)]
    for q in range(Q):
        row = nb[q]
        for n in range(npix):
            m = int(row[n])
            if m < npix:                       # 指向 padding 的不需要梯度
                buckets[m].append(q * npix + n)
    K = max(len(b) for b in buckets)
    inv = np.full((K, npix), Q * npix, dtype=np.int64)
    for m, lst in enumerate(buckets):
        inv[:len(lst), m] = lst
    return inv, K


# --- autocast 兼容：自定义 Function 需要显式声明，否则 forward/backward 的
#     dtype 行为不确定。cast_inputs=bfloat16 顺带让 gather 也在 bf16 上做，
#     访存量直接减半。autocast 关闭时这两个装饰器是空操作。
try:                                              # torch >= 2.4
    _fwd = torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.bfloat16)
    _bwd = torch.amp.custom_bwd(device_type='cuda')
except (AttributeError, TypeError):               # 老版本
    _fwd = torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    _bwd = torch.cuda.amp.custom_bwd


class _OSLOConvFn(torch.autograd.Function):
    @staticmethod
    @_fwd
    def forward(ctx, x, weight, bias, nb, inv):
        B, C, N = x.shape
        g = torch.cat([x, x.new_zeros(B, C, 1)], dim=2)[:, :, nb]      # (B,C,Q,N)
        out = torch.einsum('bcqn,ocq->bon', g, weight) + bias.view(1, -1, 1)
        ctx.save_for_backward(g, weight, inv)
        ctx.N = N
        return out

    @staticmethod
    @_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        g, weight, inv = ctx.saved_tensors
        B, C, Q, N = g.shape
        grad_w = torch.einsum('bon,bcqn->ocq', grad_out, g)
        grad_b = grad_out.sum(dim=(0, 2))
        gg = torch.einsum('bon,ocq->bcqn', grad_out, weight)           # (B,C,Q,N)
        gg = torch.cat([gg.reshape(B, C, Q * N), gg.new_zeros(B, C, 1)], dim=2)
        grad_x = gg[:, :, inv].sum(dim=2)                              # ★ gather，不是 scatter
        return grad_x, grad_w, grad_b, None, None


class OSLOConvFast(nn.Module):
    """与原 OSLOConv 数学等价，只是自定义了反向。"""
    def __init__(self, in_channels, out_channels, nside=16):
        super().__init__()
        import healpy as hp
        self.npix = hp.nside2npix(nside)
        neighbors = hp.get_all_neighbours(nside, np.arange(self.npix))
        rf = np.concatenate([neighbors, np.arange(self.npix)[np.newaxis, :]], axis=0)
        rf[rf == -1] = self.npix
        inv, K = build_inverse_index(rf, self.npix)
        self.register_buffer('neighborhood', torch.tensor(rf, dtype=torch.int64))
        self.register_buffer('inv_index', torch.tensor(inv, dtype=torch.int64))
        self.K = K
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 9))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return _OSLOConvFn.apply(x, self.weight, self.bias,
                                 self.neighborhood, self.inv_index)


# ==========================================================================
# 怎么用（Cell 2 里）
#
# 1. 把上面的 build_inverse_index / _OSLOConvFn / OSLOConvFast 三段贴进 Cell 2，
#    放在原来的 OSLOConv 后面。
# 2. 把 OSLOResBlock 里两处 `OSLOConv(...)` 改成 `OSLOConvFast(...)`，
#    以及 Conditioned_OSLO_Decoder 里的 `self.dec_final = OSLOConv(16, 1, nside)`。
#    —— 或者更省事：在 Cell 2 末尾加一行 `OSLOConv = OSLOConvFast`（别名覆盖），
#       前提是这行在所有用到它的类【定义之后、实例化之前】。
# 3. 参数名和形状完全一致（weight (O,C,9) + bias (O,)），所以
#    【旧 checkpoint 可以直接 load】，不用重训。
#
# 数值等价性已验（fp64，nside=4）：
#    前向 max|Δ| = 0.000e+00
#    grad_x max|Δ| = 4.44e-16（机器精度）
#    grad_weight / grad_bias max|Δ| = 0.000e+00
#    torch.autograd.gradcheck（数值微分对拍）= True
#
# 上到 GPU 后先跑一次对拍再信它：
#
#   old = OSLOConv(32, 32).to(device);  new = OSLOConvFast(32, 32).to(device)
#   new.weight.data.copy_(old.weight.data); new.bias.data.copy_(old.bias.data)
#   x1 = torch.randn(64, 32, 3072, device=device, requires_grad=True)
#   x2 = x1.detach().clone().requires_grad_(True)
#   y1, y2 = old(x1), new(x2); g = torch.randn_like(y1)
#   y1.backward(g); y2.backward(g)
#   print('fwd', (y1-y2).abs().max().item(), 'grad', (x1.grad-x2.grad).abs().max().item())
#   # 再用 FFM_2_3_bench_kernels.py 量一次 260 ms 变成多少
# ==========================================================================
