import numpy as np
from pystoi import stoi

def calculate_stoi(reference_array, degraded_array, fs=16000, extended=False):
    """
    Calculate STOI (Short-Time Objective Intelligibility) score between reference
    and degraded audio arrays.
    
    Parameters:
    -----------
    reference_array : numpy.ndarray
        Reference/original audio signal array
    degraded_array : numpy.ndarray
        Degraded/processed audio signal array
    fs : int, optional
        Sampling rate in Hz (default: 16000 Hz)
    extended : bool, optional
        Whether to use extended STOI (ESTOI) instead of traditional STOI
        (default: False)
        
    Returns:
    --------
    float
        STOI score ranging from 0 to 1, higher is better
        - Close to 1: High intelligibility
        - Close to 0: Low intelligibility
    
    Notes:
    ------
    - STOI is most accurate for speech signals
    - Both signals must have the same sampling rate
    - Minimum duration should be above 384ms (STOI) or 400ms (ESTOI)
    - ESTOI is more accurate for highly modulated signals
    """

    

    # Calculate STOI score
    try:
        score = stoi(reference_array, degraded_array, fs, extended=extended)
        return score
    except Exception as e:
        raise RuntimeError(f"Error calculating STOI: {str(e)}")

def calculate_estoi(reference_array, degraded_array, fs=16000):
    """
    Calculate Extended STOI (ESTOI) score between reference and degraded audio arrays.
    ESTOI is more accurate than STOI for highly modulated signals.
    
    Parameters:
    -----------
    reference_array : numpy.ndarray
        Reference/original audio signal array
    degraded_array : numpy.ndarray
        Degraded/processed audio signal array
    fs : int, optional
        Sampling rate in Hz (default: 16000 Hz)
        
    Returns:
    --------
    float
        ESTOI score ranging from 0 to 1, higher is better
    """
    return calculate_stoi(reference_array, degraded_array, fs, extended=True)

def stoi_for_different_lengths(reference_array, degraded_array, fs=16000, extended=False):
    """
    Calculate STOI score for audio arrays of different lengths by adjusting the
    degraded signal length to match the reference.
    
    Parameters:
    -----------
    reference_array : numpy.ndarray
        Reference/original audio signal array
    degraded_array : numpy.ndarray
        Degraded/processed audio signal array
    fs : int, optional
        Sampling rate in Hz (default: 16000 Hz)
    extended : bool, optional
        Whether to use extended STOI (ESTOI) (default: False)
        
    Returns:
    --------
    float
        STOI/ESTOI score ranging from 0 to 1
    """
    # Ensure audio is mono (single channel)
    if len(reference_array.shape) > 1:
        reference_array = np.mean(reference_array, axis=1)
    if len(degraded_array.shape) > 1:
        degraded_array = np.mean(degraded_array, axis=1)
    
    ref_len = len(reference_array)
    deg_len = len(degraded_array)
    
    # Handle different lengths
    if ref_len > deg_len:
        # Pad degraded signal with zeros
        degraded_array = np.pad(degraded_array, (0, ref_len - deg_len), mode='constant')
    elif ref_len < deg_len:
        # Truncate degraded signal
        degraded_array = degraded_array[:ref_len]
    
    return calculate_stoi(reference_array, degraded_array, fs, extended=extended)