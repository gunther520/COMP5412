import numpy as np
from pesq import pesq

def calculate_pesq(reference_array, degraded_array, fs=16000, mode='wb'):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score between reference
    and degraded audio arrays.
    
    Parameters:
    -----------
    reference_array : numpy.ndarray
        Reference/original audio signal array
    degraded_array : numpy.ndarray
        Degraded/processed audio signal array
    fs : int, optional
        Sampling rate (default: 16000 Hz)
    mode : str, optional
        PESQ mode - 'nb' for narrowband (8000 Hz) or 'wb' for wideband (16000 Hz)
        
    Returns:
    --------
    float
        PESQ score ranging from -0.5 to 4.5, higher is better
    
    Notes:
    ------
    - For narrowband mode, fs must be 8000 Hz
    - For wideband mode, fs must be 16000 Hz
    """
    
    # Calculate PESQ score
    try:
        score = pesq(fs, reference_array, degraded_array, mode)
        return score
    except Exception as e:
        raise RuntimeError(f"Error calculating PESQ: {str(e)}")