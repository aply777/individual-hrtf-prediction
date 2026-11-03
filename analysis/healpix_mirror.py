# ==========================================================================
# HEALPix 左右镜像：把右耳的 HRTF 图翻成"看起来像左耳"
#
# 动机：方差分解显示 71.7% 的 latent 方差在编码 side（PC1 corr=0.997）。
#       镜像后这个自由度消失，容量释放给个体差异。
#
# 关键：必须验证 phi -> -phi 在 HEALPix 上是【精确置换】。
#       如果不是精确的，最近邻映射会引入插值误差，要如实报告。
# ==========================================================================
import numpy as np
import healpy as hp

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)

def build_mirror_index(nside=NSIDE):
    """返回置换 idx，使得 mirrored_map = m[idx]（沿中矢面 phi -> -phi 反射）"""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    idx = hp.ang2pix(nside, theta, (-phi) % (2 * np.pi))
    return idx

idx = build_mirror_index()

# ---- 验证 1：是不是双射（精确置换）？----
is_perm = len(np.unique(idx)) == NPIX
print(f"是精确置换: {is_perm}   (unique={len(np.unique(idx))}/{NPIX})")

# ---- 验证 2：镜像两次应回到原图 ----
print(f"镜像两次 == 恒等: {np.array_equal(idx[idx], np.arange(NPIX))}")

# ---- 验证 3：角度误差有多大 ----
theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
th2, ph2 = hp.pix2ang(NSIDE, idx)
d_theta = np.abs(theta - th2)
d_phi = np.abs(((-phi) % (2*np.pi)) - ph2)
d_phi = np.minimum(d_phi, 2*np.pi - d_phi)
print(f"角度误差: theta max {np.degrees(d_theta).max():.4f} deg, "
      f"phi max {np.degrees(d_phi).max():.4f} deg")
if not is_perm:
    print("★ 不是精确置换 -> 用的是最近邻，误差如上，需在论文中说明")

# ---- 用法 ----
def mirror_hrtf(x):
    """x: (..., 129, 3072) 右耳 -> 镜像成'左耳视角'"""
    return x[..., idx]

np.save("/content/drive/MyDrive/HRTF_Checkpoints/healpix_mirror_idx.npy", idx)
print("\n置换已存 -> healpix_mirror_idx.npy")
print("Dataset 里用法:  right = right[..., idx]   然后左右耳一起当同分布样本训练")

# ---- 快速验证镜像是否真的有效：ILD 的横向轴应该翻过来 ----
# （拿一个受试者的左右耳，镜像右耳后两者应当高度相似）
try:
    L, R, _ = train_dataset[0]
    L, R = L.numpy(), R.numpy()
    Rm = R[..., idx]
    c_raw = np.corrcoef(L.ravel(), R.ravel())[0, 1]
    c_mir = np.corrcoef(L.ravel(), Rm.ravel())[0, 1]
    print(f"\n同一受试者 corr(左耳, 右耳)        = {c_raw:.3f}")
    print(f"           corr(左耳, 镜像后右耳)  = {c_mir:.3f}   <- 应显著更高")
except Exception as e:
    print("（跳过样本验证:", e, "）")
