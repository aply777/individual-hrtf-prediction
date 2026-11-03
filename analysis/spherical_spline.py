"""Spherical thin-plate spline (STPS) projection onto a HEALPix grid.

Every database in this project reaches its (129, 3072) dB-magnitude array through the
same three steps: FFT -> 20 log10 |.| -> STPS projection onto HEALPix nside=16. This
module is that third step, and is imported by the CARLab export scripts in this folder.

build_interpolation_matrix() returns a (npix, n_directions) weight matrix that depends
only on the measurement grid, so it is computed once per grid and cached; projecting a
subject is then a single matmul.

The __main__ block at the bottom is the SYMARE .mat driver, kept for provenance. Note
that its paths are local and that mat2HEALPix.py, an earlier scipy Rbf version of the
same idea, is superseded by this file. See docs/DATA.md.
"""
import os
import numpy as np
import scipy.io as sio
import healpy as hp

# ==========================================
# Core algorithm: spherical thin-plate spline (STPS), pure Python
# ==========================================

def _calc_QR(D, spline_order):
    """Kernel matrix of Wahba's spherical spline"""
    I = D < 1.0
    D_safe = np.copy(D)
    D_safe[~I] = 1.0

    W = (1.0 - D_safe) / 2.0
    C = 2.0 * np.sqrt(W)

    A = np.zeros_like(D_safe)
    # Evaluate only where D < 1, to avoid dividing by zero when C = 0
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

def build_interpolation_matrix(xyz_scatter, xyz_hp, spline_order=2, smoothing=1e-5):
    """Build the STPS spatial mapping weight matrix"""
    N = xyz_scatter.shape[0]
    P = xyz_hp.shape[0]
    
    # 1. Source-to-source relation (corresponds to SphericalSplineParam)
    D_inp = xyz_scatter @ xyz_scatter.T
    R_inp = _calc_QR(D_inp, spline_order)
    
    M = R_inp + smoothing * np.eye(N)
    Minv = np.linalg.inv(M)
    
    sum_Minv_cols = np.sum(Minv, axis=0)
    sum_Minv_total = np.sum(Minv)
    w_T = sum_Minv_cols / sum_Minv_total 
    
    I_mat = np.eye(N)
    ones_N = np.ones((N, 1))
    H = Minv @ (I_mat - ones_N @ w_T.reshape(1, N)) 
    
    # 2. Source-to-target relation (corresponds to SphericalSplineEval)
    D_cross = xyz_scatter @ xyz_hp.T
    R_cross = _calc_QR(D_cross, spline_order) 
    
    # 3. Assemble the final matrix
    ones_P = np.ones((P, 1))
    W_interp = (R_cross.T @ H) + (ones_P @ w_T.reshape(1, N)) 
    
    return W_interp

# ==========================================
# Data processing
# ==========================================

def process_full_spectrum_to_healpix_stps(hrir_file, W_matrix, ear_key='hL'):
    """
    Load HRIR data, FFT it, and map it onto HEALPix with a precomputed STPS matrix.
    Output shape: (129, npix)
    """
    audio_data = sio.loadmat(hrir_file)
    if ear_key not in audio_data:
        raise KeyError(f"Data for '{ear_key}' not found in the .mat file.")
    h_data = audio_data[ear_key] 
    
    # FFT Transformation 
    fft_h_data = np.fft.fft(h_data, axis=0)
    fft_positive = fft_h_data[:129, :]
    mag_dB = 20 * np.log10(np.abs(fft_positive) + 1e-9) 
    
    # One matmul interpolates all 129 frequency layers at once
    full_spectrum_matrix = mag_dB @ W_matrix.T
    
    return full_spectrum_matrix.astype(np.float32)

# ==========================================
# Batch Processing Execution Block
# ==========================================
if __name__ == "__main__":
    # --- paths (replace with your own) ---
    data_directory = "/Users/harry/Documents/26S1/ELEC5020/SYMARE/HRIRs/Acoustic"
    param_directory = "/Users/harry/Documents/26S1/ELEC5020/SYMARE/HRIRs/Parameters" 
    out_dir_left = "./npy_full_spectrum_left_stps"
    out_dir_right = "./npy_full_spectrum_right_stps"
    
    for d in [out_dir_left, out_dir_right]:
        os.makedirs(d, exist_ok=True)

    print("Phase 1: Pre-computing Spherical Spline Matrix...")
    
    # Load the measurement directions once
    azim = np.squeeze(sio.loadmat(os.path.join(param_directory, 'azim.mat'))['azim'])
    elev = np.squeeze(sio.loadmat(os.path.join(param_directory, 'elev.mat'))['elev'])
    
    # Assumes elev is elevation (equator = 0, north pole = pi/2).
    # If your elev is colatitude instead, use z_scatter = np.cos(elev).
    x_scatter = np.cos(elev) * np.cos(azim)
    y_scatter = np.cos(elev) * np.sin(azim)
    z_scatter = np.sin(elev)
    xyz_scatter = np.column_stack((x_scatter, y_scatter, z_scatter))
    
    # Build the HEALPix target grid
    nside = 16
    npix = hp.nside2npix(nside)
    x_hp, y_hp, z_hp = hp.pix2vec(nside, np.arange(npix))
    xyz_hp = np.column_stack((x_hp, y_hp, z_hp))
    
    # Compute the STPS mapping matrix
    W_matrix = build_interpolation_matrix(xyz_scatter, xyz_hp, spline_order=2, smoothing=1e-5)
    print(f"Matrix built! Shape: {W_matrix.shape}. Ready for processing.")
    
    print("\nPhase 2: Starting batch processing...")
    
    # Loop over HA01 .. HA61
    for i in range(1, 62):
        subject_id = f"HA{i:02d}" 
        hrir_file_path = os.path.join(data_directory, f"{subject_id}.mat")
        
        if not os.path.exists(hrir_file_path):
            continue
            
        print(f"--> Processing Subject: {subject_id}")
            
        try:
            matrix_left = process_full_spectrum_to_healpix_stps(hrir_file_path, W_matrix, ear_key='hL')
            np.save(os.path.join(out_dir_left, f"{subject_id}_left_full.npy"), matrix_left)
            
            matrix_right = process_full_spectrum_to_healpix_stps(hrir_file_path, W_matrix, ear_key='hR')
            np.save(os.path.join(out_dir_right, f"{subject_id}_right_full.npy"), matrix_right)
            
            print(f"    [Success] {subject_id} Left and Right channels saved.")
        except Exception as e:
            print(f"    [Error] Failed on {subject_id}: {e}")

    print("\nAll conversions completed using pure Python STPS.")