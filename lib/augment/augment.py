# augmentations.py

import numpy as np
from scipy import fftpack, signal

def add_gaussian_noise(epoch: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*epoch.shape) * (np.std(epoch) * sigma)
    return epoch + noise

def time_warp(epoch: np.ndarray, warp_ratio: float, max_seg: float=0.5):
    ch, T = epoch.shape
    seg_len = int(T * max_seg)
    start = np.random.randint(0, T - seg_len)
    factor = 1 + np.random.uniform(-warp_ratio, warp_ratio)
    warped = signal.resample(epoch[:, start:start+seg_len], int(seg_len * factor), axis=1)
    # pad/crop back to seg_len
    if warped.shape[1] < seg_len:
        pad = np.zeros((ch, seg_len - warped.shape[1]))
        warped = np.concatenate([warped, pad], axis=1)
    else:
        warped = warped[:, :seg_len]
    return np.concatenate([epoch[:, :start], warped, epoch[:, start+seg_len:]], axis=1)

def frequency_shift(epoch: np.ndarray, fs: float, shift_hz: float):
    ch, T = epoch.shape
    spec = fftpack.fft(epoch, axis=1)
    bin_shift = int(np.round(shift_hz / (fs / T)))
    spec_shifted = np.roll(spec, bin_shift, axis=1)
    return np.real(fftpack.ifft(spec_shifted, axis=1))

def mixup_features(x1: np.ndarray, y1: np.ndarray,
                   x2: np.ndarray, y2: np.ndarray,
                   alpha: float):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix

def mixup_batch(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Batch-level mixup on raw EEG windows (or feature batches).
    X: (B, C, T) or (B, D)
    y: (B,) integer labels
    alpha: mixup Beta parameter
    Returns:
      X_mix: (B, …)
      y_a:   (B,) original labels
      y_b:   (B,) permuted labels
      lam:   mixing coefficient
    """
    if alpha <= 0:
        return X, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    X2 = X[idx]
    y2 = y[idx]
    X_mix = lam * X + (1 - lam) * X2
    return X_mix, y, y2, lam

def apply_raw_augmentations(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    X: (n_trials, n_ch, n_times)
    cfg: config.augment.augmentations dictionary
    """
    out = []
    for epoch in X:
        e = epoch
        if cfg["gaussian_noise"]["enabled"]:
            e = add_gaussian_noise(e, cfg["gaussian_noise"]["sigma"])
        if cfg["time_warp"]["enabled"]:
            tp = cfg["time_warp"]
            e = time_warp(e, tp["warp_ratio"], tp["max_seg"])
        if cfg["frequency_shift"]["enabled"]:
            fsf = cfg["frequency_shift"]
            e = frequency_shift(e, fsf["fs"], fsf["shift_hz"])
        out.append(e)
    return np.stack(out, axis=0)

def apply_mixup(X: np.ndarray, y: np.ndarray, cfg: dict):
    """
    Batch-level mixup wrapper.
    X: (B, n_ch, n_times)
    y: (B,)
    cfg: cfg.augment.augmentations["mixup"]
    """
    from lib.augment.augment import mixup_batch
    Xm, ya, yb, lam = mixup_batch(X, y, cfg["alpha"])
    return Xm, ya, yb, lam