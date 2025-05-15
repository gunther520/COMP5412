import numpy as np

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in decibels"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')  # No noise case

def calculate_snr_improvement(clean, noisy, denoised):
    """Calculate SNR improvement after denoising"""
    # Calculate original noise
    original_noise = noisy - clean
    
    # Calculate remaining noise after denoising
    remaining_noise = denoised - clean
    
    # Calculate SNR before and after
    snr_before = calculate_snr(clean, original_noise)
    snr_after = calculate_snr(clean, remaining_noise)
    
    return snr_before, snr_after, snr_after - snr_before