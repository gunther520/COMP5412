import numpy as np
from scipy.fftpack import dct, idct

def dct_denoise(signal, threshold_factor=0.1):
    """
    Denoise a signal using DCT (Discrete Cosine Transform).
    
    Args:
        signal (np.ndarray): The input signal to be denoised
        threshold_factor (float): Factor to determine the threshold value (0-1)
            Higher values result in more aggressive noise removal
    
    Returns:
        np.ndarray: The denoised signal
    """
    # Convert signal to numpy array if it isn't already
    signal = np.array(signal, dtype=float)
    
    # Apply DCT to the signal
    dct_coeffs = dct(signal, type=2, norm='ortho')
    
    # Calculate threshold based on the coefficients
    threshold = threshold_factor * np.max(np.abs(dct_coeffs))
    
    # Apply hard thresholding - set coefficients below threshold to zero
    dct_coeffs_thresholded = dct_coeffs * (np.abs(dct_coeffs) > threshold)
    
    # Apply inverse DCT to get the denoised signal
    denoised_signal = idct(dct_coeffs_thresholded, type=2, norm='ortho')
    
    return denoised_signal

def dct_denoise_advanced(signal, threshold_type='hard', threshold_factor=0.1, keep_n=None):
    """
    Advanced version of DCT denoising with more parameters.
    
    Args:
        signal (np.ndarray): The input signal to be denoised
        threshold_type (str): Type of thresholding - 'hard', 'soft', or 'keep_n'
        threshold_factor (float): Factor to determine threshold value (0-1)
        keep_n (int, optional): Number of largest coefficients to keep (if threshold_type is 'keep_n')
    
    Returns:
        np.ndarray: The denoised signal
    """
    # Convert signal to numpy array if it isn't already
    signal = np.array(signal, dtype=float)
    
    # Apply DCT to the signal
    dct_coeffs = dct(signal, type=2, norm='ortho')
    
    if threshold_type == 'keep_n' and keep_n is not None:
        # Keep only the N largest coefficients
        idx = np.argsort(np.abs(dct_coeffs))[::-1]
        mask = np.zeros_like(dct_coeffs, dtype=bool)
        mask[idx[:keep_n]] = True
        dct_coeffs_thresholded = dct_coeffs * mask
    else:
        # Calculate threshold based on the coefficients
        threshold = threshold_factor * np.max(np.abs(dct_coeffs))
        
        if threshold_type == 'hard':
            # Hard thresholding
            dct_coeffs_thresholded = dct_coeffs * (np.abs(dct_coeffs) > threshold)
        elif threshold_type == 'soft':
            # Soft thresholding
            dct_coeffs_thresholded = np.sign(dct_coeffs) * np.maximum(0, np.abs(dct_coeffs) - threshold)
        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")
    
    # Apply inverse DCT to get the denoised signal
    denoised_signal = idct(dct_coeffs_thresholded, type=2, norm='ortho')
    
    return denoised_signal