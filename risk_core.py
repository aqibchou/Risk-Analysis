import numpy as np

def _shannon_entropy(x: np.ndarray, bins: int = 30, normalize: bool = True) -> float:
 
    x = np.asarray(x, dtype=float)
    if np.allclose(np.std(x), 0):
        x = x + 1e-12 * np.random.randn(*x.shape)  # éviter variance nulle
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]  # éviter log(0)
    H = -np.sum(p * np.log2(p))
    if not normalize:
        return float(H)
    Hmax = np.log2(len(counts))
    return float(H / Hmax) if Hmax > 0 else np.nan

def _mutual_information_norm(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:

    counts_x, _ = np.histogram(x, bins=bins)
    counts_y, _ = np.histogram(y, bins=bins)
    counts_xy, _, _ = np.histogram2d(x, y, bins=(bins, bins))

 
    def H(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    Hx = H(counts_x)
    Hy = H(counts_y)
    Hxy = H(counts_xy.ravel())

    MI = Hx + Hy - Hxy
    return float(MI / Hx) if Hx > 0 else np.nan

def _hurst_exponent_rs(x: np.ndarray, min_window: int = 16) -> float:

    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    max_window = N // 2
    sizes = np.floor(np.logspace(np.log10(min_window), np.log10(max_window), num=10)).astype(int)

    RS = []
    for w in sizes:
        if w < 2:
            continue
        n_seg = N // w
        rs_vals = []
        for i in range(n_seg):
            seg = x[i*w:(i+1)*w]
            Y = np.cumsum(seg - np.mean(seg))
            R = np.max(Y) - np.min(Y)
            S = np.std(seg)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append((w, np.mean(rs_vals)))

    if not RS:
        return np.nan
    w_arr, rs_arr = zip(*RS)
    H, _ = np.polyfit(np.log(w_arr), np.log(rs_arr), 1)
    return float(H)

def detect_risk_factors(returns: np.ndarray,
                        bins: int = 30,
                        mi_lag: int = 1) -> dict:
    H_ent = _shannon_entropy(returns, bins=bins, normalize=True)
    mi_norm = _mutual_information_norm(returns[mi_lag:], returns[:-mi_lag], bins=bins)
    hurst = _hurst_exponent_rs(returns)

    risk_flag = "latent"
    if hurst > 0.5 and H_ent > 0.9 and mi_norm < 0.05:
        risk_flag = "structural_time_bomb"
    elif hurst > 0.5 and H_ent > 0.9:
        risk_flag = "chaotic_memory"
    elif mi_norm < 0.05:
        risk_flag = "blocked_flow"

    return {
        "entropy": H_ent,
        "mi_norm": mi_norm,
        "hurst": hurst,
        "risk_flag": risk_flag
    }

