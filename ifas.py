import numpy as np
from scipy.fft import fft

def extract_IFAS(x_vec, fs_sca, K_sca=2048, B_sca=1025, a_sca=3.0):
    x_vec = x_vec.astype(np.float32) / np.iinfo(np.int16).max
    S_sca = fs_sca // 1000

    # Zero-padding
    y_vec = np.pad(x_vec.flatten(), (B_sca // 2, B_sca // 2 - 1), mode='constant')

    # Frame extraction
    C_sca = int(np.floor((len(y_vec) - B_sca) / S_sca)) + 1
    frames = np.lib.stride_tricks.sliding_window_view(y_vec, window_shape=B_sca)[::S_sca][:C_sca].T

    if_mat, as_mat = ifas_instfreq_gausswin(frames, K_sca, fs_sca, a_sca)

    return if_mat

def IFAS_IF_WD(a):
    return 0.30 * a

def ifas_instfreq_gausswin(x_mat, K_sca, fs_sca, a_sca, if_min_mlbpoints=5):
    B_sca, C_sca = x_mat.shape
    w_sca = (B_sca - 1) // 2 if B_sca % 2 == 1 else B_sca // 2
    ru_sca = (w_sca / (a_sca - IFAS_IF_WD(a_sca)))**2
    rd_sca = (w_sca / (a_sca + IFAS_IF_WD(a_sca)))**2

    t = (np.arange(B_sca) - w_sca) / w_sca
    g_sca = np.exp(-0.5 * (a_sca * t)**2).reshape(-1, 1)
    pos = (np.arange(B_sca) - w_sca).reshape(-1, 1)

    # Apply window
    x0_mat = x_mat * g_sca
    x1_mat = x0_mat * pos
    x2_mat = x0_mat * (pos ** 2)

    # Apply FFT
    X0 = fft(x0_mat, n=K_sca, axis=0)
    X1 = fft(x1_mat, n=K_sca, axis=0)
    X2 = fft(x2_mat, n=K_sca, axis=0)

    eps = np.finfo(float).eps
    abs_X0_sq = np.abs(X0)**2
    valid_mask = abs_X0_sq > eps

    r1 = np.zeros_like(X0, dtype=np.complex128)
    r2 = np.zeros_like(X0, dtype=np.complex128)
    rx = np.zeros_like(X0.real)

    r1[valid_mask] = -1j * X1[valid_mask] / X0[valid_mask]
    r2[valid_mask] = -X2[valid_mask] / X0[valid_mask]
    rx[valid_mask] = (r1[valid_mask]**2 - r2[valid_mask]).real

    log_abs_X0 = np.full_like(X0.real, np.nan)
    in_range_mask = (rx > rd_sca) & (rx < ru_sca) & valid_mask
    log_abs_X0[in_range_mask] = np.log(np.abs(X0[in_range_mask]))
    as_mat = np.abs(X0)

    # --- Remove sidelobes ---
    as_tmp = log_abs_X0.copy()
    for c in range(C_sca):
        tmp = as_tmp[:, c]
        isn = np.isnan(tmp)
        c_count = 0
        for k in range(K_sca):
            if not isn[k]:
                c_count += 1
            elif c_count > 0:
                if c_count < if_min_mlbpoints:
                    tmp[k - c_count:k] = np.nan
                c_count = 0
        as_tmp[:, c] = tmp

    # --- Instantaneous Frequency estimation ---
    if_mat = np.full_like(as_tmp, np.nan)
    w = if_min_mlbpoints // 2
    for c in range(C_sca):
        tmp = as_tmp[:, c]
        c_count = 0
        as_max = -np.inf
        i_sca = 0
        for k in range(K_sca):
            if not np.isnan(tmp[k]):
                c_count += 1
                if tmp[k] > as_max:
                    as_max = tmp[k]
                    i_sca = k
            elif c_count > 0:
                if i_sca - w > 0 and i_sca + w < K_sca:
                    alp = tmp[i_sca - 1]
                    bta = tmp[i_sca]
                    gma = tmp[i_sca + 1]
                    denom = alp - 2 * bta + gma
                    p = 0.5 * (alp - gma) / denom if denom != 0 else 0
                    for n in range(-w, w):
                        if_mat[i_sca + n, c] = (i_sca + p) * fs_sca / K_sca
                as_max = -np.inf
                c_count = 0

    return if_mat, as_mat