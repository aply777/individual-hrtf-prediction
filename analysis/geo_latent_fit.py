# ==========================================================================
# geo_latent_fit.py — 量化「几何特征 与 latent z 的贴合程度」
#
# 输出三个数：
#   dCor²ᵤ   无偏距离相关平方。不拟合任何模型 -> 不会过拟合。
#            = 0 当且仅当两者【独立】（比相关系数强得多，非线性也抓得到）。
#            零假设分布只取决于 N，与特征维度无关：
#              N=57  -> 0.000 ± 0.025，95 分位 0.040
#              N=115 -> 0.000 ± 0.013，95 分位 0.021
#              N=200 -> 0.000 ± 0.007，95 分位 0.012
#   p        置换检验 p 值。这才是判据，不是 dCor 本身的大小。
#   CV-R²    交叉验证 ridge 的 R²。dCor 说"有关系"，这个说"能用上多少"。
#
# ⚠️ 不要用 dcor.distance_correlation（有偏版）也不要用 CKA：
#    实测在 N=57、独立数据上它们分别读出 0.830 和 0.378。
#
# pip install dcor scikit-learn
# ==========================================================================
import numpy as np, warnings
warnings.filterwarnings('ignore')
import dcor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

NULL95 = {57: 0.040, 115: 0.021, 200: 0.012}     # 上表，N 不在表里就看 1.65/sqrt-ish


# ==========================================================================
# 1. 准备 Z —— ★ 必须先去掉「左右耳」那一维
# ==========================================================================
def person_subspace(Z, side, force=None):
    """Z: (2N, d) 每只耳一行；side: (2N,) bool，True=右耳

    ★ 2026-08-31 更新 —— 这个函数的前提随模型版本变化，不要照抄旧数字：
        v1   : side 方向占 latent 方差 71.7%，person 分量只有 18.6%。
               不投影的话 71.7% 的目标是几何平凡就知道的一个比特，R² 会虚高。
        v2.3 : 镜像生效后实测 **1.4%**。此时投影几乎去不掉任何东西，
               只白白损失一个自由度，默认【不投影】。
    所以判据是实测值本身，不是和某个历史数字对不对得上。
    force=True/False 可以强制；默认 >5% 才投影。"""
    v = Z[side].mean(0) - Z[~side].mean(0)        # side 方向
    v = v / np.linalg.norm(v)
    Zp = Z - np.outer(Z @ v, v)                   # 投影掉
    var_removed = 1 - Zp.var() / Z.var()
    do = (var_removed > 0.05) if force is None else force
    print(f'  side 方向占 latent 方差 {var_removed:.1%} '
          f'(v1=71.7% / v2.3 实测 1.4%) -> {"投影掉" if do else "不投影"}')
    if var_removed > 0.5:
        print('  ⚠️ 超过 50%：镜像可能没生效，或 side 标签接反了。先查 MonauralHRTFDataset。')
    return Zp if do else Z


def whiten(X, keep=0.95):
    """PCA 白化并截断。不做的话，几个大方差维会主导距离，小维度的信息被淹掉。"""
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), keep) + 1)
    return U[:, :k] * np.sqrt(len(X)), k


# ==========================================================================
# 2. 三个指标
# ==========================================================================
def dcor_test(L, Z, nperm=999, seed=0):
    rng = np.random.default_rng(seed)
    obs = dcor.u_distance_correlation_sqr(L, Z)
    null = np.array([dcor.u_distance_correlation_sqr(L, Z[rng.permutation(len(Z))])
                     for _ in range(nperm)])
    p = (np.sum(null >= obs) + 1) / (nperm + 1)
    return obs, p, null.mean(), null.std()


def cv_r2(L, Z, k=5, seed=0):
    """留出折上的 R²。k=len(L) 就是 leave-one-out。"""
    kf = KFold(min(k, len(L)), shuffle=True, random_state=seed)
    P = np.zeros_like(Z)
    for tr, te in kf.split(L):
        P[te] = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(L[tr], Z[tr]).predict(L[te])
    return 1 - ((Z - P) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum(), P


def report(name, L, Z, nperm=999):
    N = len(L)
    d, p, m0, s0 = dcor_test(L, Z, nperm)
    r2, _ = cv_r2(L, Z)
    thr = NULL95.get(min(NULL95, key=lambda k: abs(k - N)))
    z = (d - m0) / s0 if s0 > 0 else float('nan')
    print(f'\n【{name}】 N={N}  landmark {L.shape[1]}维  latent {Z.shape[1]}维')
    print(f'  dCor²ᵤ  = {d:+.4f}   (置换零分布 {m0:+.4f} ± {s0:.4f}, '
          f'z = {z:+.1f}σ)')
    print(f'  p       = {p:.4f}   {"★ 显著" if p < 0.05 else "  不显著"}'
          f'   [N≈{N} 的 95 分位参考 {thr:.3f}]')
    print(f'  CV-R²   = {r2:+.4f}   {"—— 有可用信号" if r2 > 0.05 else "—— 回归器抓不到"}')
    return dict(name=name, N=N, dcor=d, p=p, r2=r2)


# ==========================================================================
# 3. 逐频带拆开 —— 这才是真正的科学结论
# ==========================================================================
def per_band(L, Z_fd, freqs, edges=((0, 500), (500, 6000), (6000, 12000),
                                    (12000, 16000))):
    """Z_fd: (N, num_bins, latent_dim) 保留频率轴的 latent
    问题不是「几何能不能预测 z」，是「几何能预测 z 的哪一段」。
    如果 notch 带(6-12k)预测不动而低频能，那就是 Thread A 的直接论据。"""
    print(f"\n{'频带':>14}{'dCor²ᵤ':>10}{'p':>8}{'CV-R²':>9}")
    out = []
    for lo, hi in edges:
        m = (freqs >= lo) & (freqs < hi)
        Zb, _ = whiten(Z_fd[:, m, :].reshape(len(Z_fd), -1))
        d, p, _, _ = dcor_test(L, Zb, 499)
        r2, _ = cv_r2(L, Zb)
        print(f'{lo/1000:>6.1f}-{hi/1000:<5.1f}k{d:>10.4f}{p:>8.3f}{r2:>9.4f}')
        out.append((lo, hi, d, p, r2))
    return out


# ==========================================================================
# 4. 必须一起报的对照
# ==========================================================================
def controls(L, Z, seed=0):
    rng = np.random.default_rng(seed)
    print('\n对照：')
    d0, p0, _, _ = dcor_test(L, Z[rng.permutation(len(Z))], 499)
    print(f'  打乱受试者标签         dCor²ᵤ {d0:+.4f}  p {p0:.3f}   (应≈0)')
    Lr = rng.normal(size=L.shape)
    d1, p1, _, _ = dcor_test(Lr, Z, 499)
    print(f'  随机特征代替 landmark  dCor²ᵤ {d1:+.4f}  p {p1:.3f}   (应≈0)')
    # 最近邻 baseline：找几何上最像的人，直接抄他的 z
    from scipy.spatial.distance import cdist
    D = cdist(L, L); np.fill_diagonal(D, np.inf)
    P = Z[D.argmin(1)]
    r2nn = 1 - ((Z - P) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()
    print(f'  最近邻抄答案 baseline  CV-R² {r2nn:+.4f}   '
          f'(打不过它，回归就没意义)')


# ==========================================================================
# 5. 用法
# ==========================================================================
if __name__ == '__main__':
    # ---- 自检：造一份已知答案的数据，确认整条链路没写错 ----
    rng = np.random.default_rng(0)
    N, k = 57, 8
    F = rng.normal(size=(N, k))
    for truth, tag in ((0.0, '独立（应不显著）'), (0.25, '有信号（应显著）')):
        Zt = np.sqrt(truth) * (F @ rng.normal(size=(k, 20))) + \
             np.sqrt(1 - truth) * rng.normal(size=(N, 20))
        Lt = np.sqrt(truth) * (F @ rng.normal(size=(k, 60))) + \
             np.sqrt(1 - truth) * rng.normal(size=(N, 60))
        report(f'自检 {tag}', *[whiten(x)[0] for x in (Lt, Zt)], nperm=499)

    print('\n' + '=' * 62)
    print('自检通过后，把下面三行换成你的真实数据：')
    print("""
    L   = landmarks.reshape(N, -1)      # (N, 3*n_landmark) 归一化后的坐标
    Z   = latents                       # (2N, d) 每只耳一行
    Zp  = person_subspace(Z, side_mask) # ★ 先去掉 side
    Zp  = 0.5*(Zp[::2] + Zp[1::2])      # 同一人的左右耳合并 -> (N, d)
    Lw,_ = whiten(L);  Zw,_ = whiten(Zp)
    report('SYMARE57 landmark -> latent', Lw, Zw)
    controls(Lw, Zw)
    """)
