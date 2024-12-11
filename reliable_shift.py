import numpy as np
from numpy.fft import fft2, ifft2, fftshift



def calculate_shift_reliability(template, target,):
    # Step 2: Compute the Cross Power Spectrum (CPS)
    fft_template = fft2(template)
    fft_target = fft2(target)
    eps = 1e-15  # Avoid division by zero
    cps = fft_template * np.conjugate(fft_target) / (np.abs(fft_template) * np.abs(fft_target) + eps)
    correlation_map = fftshift(np.abs(ifft2(cps)))

    # Step 3: Locate the peak in the correlation map
    peak_row, peak_col = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    peak_strength = np.mean(correlation_map[peak_row - 1:peak_row + 2, peak_col - 1:peak_col + 2])

    # Step 4: Mask the peak and calculate background statistics
    correlation_map_masked = correlation_map.copy()
    correlation_map_masked[peak_row - 1:peak_row + 2, peak_col - 1:peak_col + 2] = -9999
    background_values = correlation_map_masked[correlation_map_masked != -9999]
    background_mean = np.mean(background_values)
    background_std = np.std(background_values)

    # Step 5: Calculate reliability
    reliability = 100 - ((background_mean + 2 * background_std) / peak_strength * 100)
    reliability = max(0, min(100, reliability))  # Clamp between 0 and 100

    return  reliability