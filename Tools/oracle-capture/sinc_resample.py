"""torchaudio.functional.resample, re-derived in numpy (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99) so the
pure-MLX tokenizer path can resample exactly the way the reference does.  Verified against torchaudio in .venv-editx."""
import math, numpy as np

def sinc_resample(wav: np.ndarray, orig_freq: int, new_freq: int, lowpass_filter_width: int = 6, rolloff: float = 0.99) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if orig_freq == new_freq: return wav
    g = math.gcd(int(orig_freq), int(new_freq)); orig, new = orig_freq // g, new_freq // g
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig                     # (1, K)
    t = (-np.arange(new, dtype=np.float64)[:, None] / new + idx) * base_freq                    # (new, K)
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    kernels = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t)) * window * (base_freq / orig)   # (new, K)
    n = wav.shape[0]; target = math.ceil(new * n / orig)
    x = np.pad(wav.astype(np.float64), (width, width + orig))
    # conv1d with stride `orig`: out[p, j] = sum_k kernels[p, k] * x[j*orig + k]
    frames = np.lib.stride_tricks.sliding_window_view(x, kernels.shape[1])[::orig]              # (T', K)
    out = frames @ kernels.T                                                                   # (T', new)
    return out.reshape(-1)[:target].astype(np.float32)
