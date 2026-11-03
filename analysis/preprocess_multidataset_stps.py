"""SOFA / .mat -> HEALPix dB-magnitude preprocessing for SYMARE, HUTUBS and SONICOM.

This is the reader that produced the HUTUBS, SONICOM and SYMARE `.npy` archives listed in
docs/DATA.md. CARLab has its own path (carlab_export.py -> export_carlab_44k1.py) because
it arrives as the lab's binary `hrirfilt`/`.imp` format rather than SOFA.

All four databases end up in the same representation: (129, 3072) float32, dB magnitude,
44.1 kHz, nfft=256 (df = 172.27 Hz), HEALPix nside=16, spline_order=2, smoothing=1e-5.

Four things to know before trusting or re-running it:

1. The STPS weight matrix is built ONCE per database, from the FIRST subject's direction
   grid, and reused for every other subject. That is correct only while a database uses a
   single grid. CARLab does not - it has 393/475/412/392-direction subjects, which is why
   its own exporter caches one matrix per grid instead.

2. The dB floor here is DB_FLOOR = 1e-6 (-120 dB); the CARLab path uses 1e-9 (-180 dB).
   Neither appears to be active in practice - the lowest value measured anywhere in the
   CARLab archive is about -80 dB, 40 dB above the shallower floor - but the two paths are
   not identical and the difference is recorded here rather than left to be rediscovered.

3. SONICOM is read from FreeFieldCompMinPhase_44kHz and HUTUBS from the raw measured
   SOFA. Those are different processing conventions; the free-field compensation in
   particular changes the magnitude spectrum. This is part of what per-dataset FiLM
   absorbs, and part of why a pooled cross-database baseline is meaningless.

4. Impulse responses longer than 256 samples are TRUNCATED, not windowed (a warning is
   printed). Minimum-phase SONICOM IRs are onset-aligned so they fit; measured HUTUBS IRs
   should be checked if this is ever re-run.

The azimuth-convention question flagged in the comments below was checked separately -
see notebooks/Check_Direction_Alignment.ipynb.

Comments below are the author's working notes and are left in the original Chinese.
"""

"""
统一多数据集预处理：SYMARE(.mat) + HUTUBS/SONICOM(.sofa) -> HEALPix dB 幅度
================================================================================
核心约定（保证跨库一致、无需 PCHIP）：
  - 所有数据重采样到 44.1 kHz
  - 统一 FFT 长度 N_FFT = 256  ->  129 bins @ 172.27 Hz，0–22.05 kHz
  - 同一套 STPS 参数（spline_order=2, smoothing=1e-5）处理所有库
  - 每个库的方向网格不同 -> 各自构建一次 W_matrix，库内所有受试者复用

依赖：pip install netCDF4 scipy healpy numpy
"""

import os
import numpy as np
import scipy.io as sio
import healpy as hp
from scipy.signal import resample_poly
from math import gcd

try:
    import netCDF4
except ImportError:
    netCDF4 = None

# ==========================================
# 全局约定
# ==========================================
TARGET_FS = 44100        # 统一采样率（最低共同采样率，只降不升）
N_FFT     = 256          # 统一 FFT 长度 -> 129 bins
N_BINS    = N_FFT // 2 + 1
NSIDE     = 16
SPLINE_ORDER = 2
SMOOTHING    = 1e-5
DB_FLOOR     = 1e-6      # 比原来的 1e-9 (=-180dB) 浅，避免极端离群点

# ==========================================
# 1. STPS 球面薄板样条（用 solve 替代 inv，更稳）
# ==========================================
def _calc_QR(D, spline_order):
    I = D < 1.0
    D_safe = np.copy(D); D_safe[~I] = 1.0
    W = (1.0 - D_safe) / 2.0
    C = 2.0 * np.sqrt(W)
    A = np.zeros_like(D_safe)
    A[I] = np.log(1.0 + 2.0 / C[I])
    if spline_order == 1:
        Q = 2.0 * A * W - C + 1.0
        R = (Q - 0.5) / (2.0 * np.pi)
    elif spline_order == 2:
        Q = (A * (12.0 * W**2 - 4.0 * W) - 6.0 * C * W + 6.0 * W + 1.0) / 2.0
        R = (Q / 2.0 - 1.0 / 6.0) / (2.0 * np.pi)
    else:
        raise ValueError("Spline order must be 1 or 2")
    return R

def build_interpolation_matrix(xyz_scatter, xyz_hp,
                                spline_order=SPLINE_ORDER, smoothing=SMOOTHING):
    """构建 STPS 权重矩阵 W: (npix, n_dir)。用 np.linalg.solve 而非 inv。"""
    N = xyz_scatter.shape[0]
    P = xyz_hp.shape[0]

    R_inp = _calc_QR(xyz_scatter @ xyz_scatter.T, spline_order)
    M = R_inp + smoothing * np.eye(N)

    ones_N = np.ones((N, 1))
    Minv   = np.linalg.solve(M, np.eye(N))          # 一次分解求逆，仍比 inv 稳
    sum_total = float((ones_N.T @ (Minv @ ones_N)).item())
    w_T = Minv.sum(axis=0) / sum_total

    H = Minv @ (np.eye(N) - ones_N @ w_T.reshape(1, N))
    R_cross = _calc_QR(xyz_scatter @ xyz_hp.T, spline_order)
    W_interp = (R_cross.T @ H) + (np.ones((P, 1)) @ w_T.reshape(1, N))
    return W_interp

# ==========================================
# 2. 坐标 -> 单位向量
#    约定：azimuth 从正前方(+x)逆时针，elevation 从水平面起算
#    !!! 必须确认三个库的 azimuth 零点/方向一致，否则空间会错位 !!!
# ==========================================
def sph_to_xyz(az_rad, el_rad):
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    return np.column_stack([x, y, z])

def healpix_xyz(nside=NSIDE):
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    return np.column_stack([x, y, z])

# ==========================================
# 3. 统一的 HRIR -> HEALPix dB 幅度
# ==========================================
def hrir_to_mag_healpix(hrir, fs, W_matrix):
    """
    hrir: (n_dir, n_samples) 单耳时域 HRIR
    返回: (129, npix) dB 幅度
    """
    fs = int(round(fs))
    if fs != TARGET_FS:                              # 只对非 44.1k 的库重采样（如 SYMARE 48k）
        g = gcd(fs, TARGET_FS)
        hrir = resample_poly(hrir, TARGET_FS // g, fs // g, axis=1)

    if hrir.shape[1] > N_FFT:
        print(f"    [warn] IR 长度 {hrir.shape[1]} > {N_FFT}，尾部被截断；"
              f"建议用 NoITD/加窗版本，或先对齐 onset")

    spec = np.fft.rfft(hrir, n=N_FFT, axis=1)        # 自动补零到 256 -> (n_dir, 129)
    mag_dB = 20.0 * np.log10(np.abs(spec) + DB_FLOOR)
    return (W_matrix @ mag_dB).T.astype(np.float32)  # (129, npix)

# ==========================================
# 4. 各数据集的读取器（统一返回 hL, hR (n_dir, n_samp), fs, xyz_scatter）
# ==========================================
def read_symare(mat_path, xyz_scatter, fs=48000):
    """SYMARE: .mat 含 hL,hR = (256, 393)。坐标来自外部 azim/elev (radians)。"""
    d = sio.loadmat(mat_path)
    hL = np.asarray(d['hL']).T   # (393, 256)
    hR = np.asarray(d['hR']).T
    return hL, hR, fs, xyz_scatter

def _load_sofa(sofa_path):
    if netCDF4 is None:
        raise ImportError("需要 netCDF4 读取 SOFA：pip install netCDF4")
    ds = netCDF4.Dataset(sofa_path, 'r')
    ir  = np.array(ds.variables['Data.IR'][:])                  # (M, R, N)
    fs  = float(np.array(ds.variables['Data.SamplingRate'][:]).flatten()[0])
    pos = np.array(ds.variables['SourcePosition'][:])           # (M, 3)
    ptype = getattr(ds.variables['SourcePosition'], 'Type', 'spherical')
    ds.close()
    return ir, fs, pos, str(ptype)

def positions_to_xyz(pos, ptype):
    if 'cart' in ptype.lower():                                 # cartesian
        xyz = pos[:, :3].astype(float)
        return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    az = np.deg2rad(pos[:, 0].astype(float))                    # spherical: [az_deg, el_deg, r]
    el = np.deg2rad(pos[:, 1].astype(float))
    return sph_to_xyz(az, el)

def read_sofa_subject(sofa_path):
    """HUTUBS/SONICOM: 每个受试者一个 SOFA。左耳=receiver0, 右耳=receiver1。"""
    ir, fs, pos, ptype = _load_sofa(sofa_path)
    hL = ir[:, 0, :]                                            # (M, N)
    hR = ir[:, 1, :]
    xyz = positions_to_xyz(pos, ptype)
    return hL, hR, fs, xyz

# ==========================================
# 5. 批处理：每个库一个配置
# ==========================================
def process_dataset(name, file_list, reader, out_left, out_right,
                     xyz_scatter=None, dataset_id=None):
    os.makedirs(out_left, exist_ok=True)
    os.makedirs(out_right, exist_ok=True)

    W_matrix = None
    if xyz_scatter is not None:                                 # SYMARE：网格已知，预先建一次
        W_matrix = build_interpolation_matrix(xyz_scatter, healpix_xyz())
        print(f"[{name}] W_matrix 预构建: {W_matrix.shape}")

    print(f"\n=== 处理 {name} ({len(file_list)} 个受试者) ===")
    for subj_id, path in file_list:
        if not os.path.exists(path):
            print(f"  跳过(文件不存在): {subj_id}"); continue
        try:
            hL, hR, fs, xyz = reader(path)

            if W_matrix is None:                                # SOFA 库：用第一个受试者的网格建一次后复用
                W_matrix = build_interpolation_matrix(xyz, healpix_xyz())
                print(f"[{name}] W_matrix 由首个受试者网格构建: {W_matrix.shape} "
                      f"(n_dir={xyz.shape[0]}, fs={fs})")

            mag_L = hrir_to_mag_healpix(hL, fs, W_matrix)
            mag_R = hrir_to_mag_healpix(hR, fs, W_matrix)

            np.save(os.path.join(out_left,  f"{subj_id}_left_full.npy"),  mag_L)
            np.save(os.path.join(out_right, f"{subj_id}_right_full.npy"), mag_R)
            print(f"  [OK] {subj_id}  (L {mag_L.shape}, R {mag_R.shape})")
        except Exception as e:
            print(f"  [ERR] {subj_id}: {e}")

    # 顺手写一个 dataset_id 标签文件，供训练时给 dataset-specific bias 索引
    if dataset_id is not None:
        with open(os.path.join(out_left, "_dataset_id.txt"), "w") as f:
            f.write(str(dataset_id))

# ==========================================
# 6. 主流程
# ==========================================
if __name__ == "__main__":

    # ---------- SYMARE (48 kHz, .mat) ----------
    SYMARE_DATA  = "/Users/harry/Documents/26S1/ELEC5020/SYMARE/HRIRs/Acoustic"
    SYMARE_PARAM = "/Users/harry/Documents/26S1/ELEC5020/SYMARE/HRIRs/Parameters"

    azim = np.squeeze(sio.loadmat(os.path.join(SYMARE_PARAM, 'azim.mat'))['azim'])
    elev = np.squeeze(sio.loadmat(os.path.join(SYMARE_PARAM, 'elev.mat'))['elev'])
    # SYMARE readme 确认 elev 是仰角(-pi/2~pi/2)，故 z=sin(elev)
    symare_xyz = sph_to_xyz(azim, elev)

    symare_files = [(f"SYM_HA{i:02d}",
                     os.path.join(SYMARE_DATA, f"HA{i:02d}.mat")) for i in range(1, 62)]
    process_dataset(
        "SYMARE", symare_files,
        reader=lambda p: read_symare(p, symare_xyz, fs=48000),
        out_left="/Users/harry/Documents/26S1/ELEC5020/SYMARE/npy_left",
        out_right="/Users/harry/Documents/26S1/ELEC5020/SYMARE/npy_right",
        xyz_scatter=symare_xyz, dataset_id=0,
    )

    # ---------- HUTUBS (44.1 kHz native, .sofa) ----------
    HUTUBS_DIR = "/Users/harry/Documents/26S1/ELEC5020/HUTUBS/HRIRs"   # pp1_HRIRs_measured.sofa 之类
    hutubs_files = [(f"HU_pp{i}",
                     os.path.join(HUTUBS_DIR, f"pp{i}_HRIRs_measured.sofa"))
                    for i in range(1, 97)]
    process_dataset(
        "HUTUBS", hutubs_files, reader=read_sofa_subject,
        out_left="/Users/harry/Documents/26S1/ELEC5020/HUTUBS/npy_left", out_right="/Users/harry/Documents/26S1/ELEC5020/HUTUBS/npy_right",
        dataset_id=1,
    )

    # ---------- SONICOM (用自带 44.1 kHz 文件，.sofa) ----------
    # 建议用 NoITD 版：onset 对齐到起点，IR 短，能完整落进 256 点
    SONICOM_DIR = "/Users/harry/Documents/26S1/ELEC5020/SONICOM/dataset"
    sonicom_files = [(f"SON_P{i:04d}",
                      os.path.join(SONICOM_DIR, f"P{i:04d}",
                                   "HRTF", "HRTF", "44kHz",
                                   f"P{i:04d}_FreeFieldCompMinPhase_44kHz.sofa"))
                     for i in range(1, 501)]
    process_dataset(
        "SONICOM", sonicom_files, reader=read_sofa_subject,
        out_left="/Users/harry/Documents/26S1/ELEC5020/SONICOM/npy_left", out_right="/Users/harry/Documents/26S1/ELEC5020/SONICOM/npy_right",
        dataset_id=2,
    )

    print("\n全部完成。所有库均为 129 bins @ 172.27 Hz (0–22.05 kHz)，可直接合并训练。")
