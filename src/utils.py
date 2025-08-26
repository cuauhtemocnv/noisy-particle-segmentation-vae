import numpy as np
import torch.nn.functional as F


def add_fourier_noise(img, sigma=0.5, low_freq_filter=0.2):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    noise = np.random.normal(0, sigma, fshift.shape) + 1j * np.random.normal(0, sigma, fshift.shape)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    l = int(min(crow, ccol) * low_freq_filter)
    mask[crow - l:crow + l, ccol - l:ccol + l] = 0
    fshift_noisy = fshift + noise * mask
    f_ishift = np.fft.ifftshift(fshift_noisy)
    img_noisy = np.fft.ifft2(f_ishift).real
    return img_noisy


def weighted_mse(output, target, weight=10.0):
    """Weighted MSE, foreground pixels weighted more."""
    fg = target > 0.1
    loss_fg = F.mse_loss(output[fg], target[fg]) if fg.any() else 0
    loss_bg = F.mse_loss(output[~fg], target[~fg])
    return loss_bg + weight * loss_fg
