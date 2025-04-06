import os
import numpy as np
import soundfile as sf
import glob
import random
from tqdm import tqdm
import librosa

def load_audio(file_path, target_sr=16000):
    """Load audio file and resample if necessary."""
    data, sr = sf.read(file_path)
    
    # Handle stereo to mono conversion if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    
    return data, target_sr

def adjust_noise_level(speech, noise, target_snr):
    """Adjust noise level to achieve target SNR."""
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor
    factor = np.sqrt(speech_power / (noise_power * (10 ** (target_snr / 10))))
    scaled_noise = noise * factor
    
    return scaled_noise

def mix_audio(speech, noise, target_snr):
    """Mix speech with noise at target SNR."""
    # First, ensure noise is at least as long as speech
    if len(noise) < len(speech):
        # Loop noise if needed
        times_to_repeat = int(np.ceil(len(speech) / len(noise)))
        repeated_noise = np.zeros(times_to_repeat * len(noise))
        
        # Apply crossfade between repetitions
        fade_len = min(1000, len(noise) // 4)  # 1000 samples or 1/4 of noise length
        for i in range(times_to_repeat):
            pos = i * len(noise)
            repeated_noise[pos:pos+len(noise)] += noise
            
            # Apply crossfade if not the first segment
            if i > 0:
                # Create fade-in window
                fade_in = np.linspace(0, 1, fade_len)
                # Apply fade-in to current segment
                repeated_noise[pos:pos+fade_len] *= fade_in
                # Apply fade-out to previous segment
                repeated_noise[pos-fade_len:pos] *= (1 - fade_in)
        
        noise = repeated_noise[:len(speech)]
    
    # Trim or select random segment of noise
    if len(noise) > len(speech):
        start = random.randint(0, len(noise) - len(speech))
        noise = noise[start:start + len(speech)]
    
    # Adjust noise level
    scaled_noise = adjust_noise_level(speech, noise, target_snr)
    
    # Mix speech and noise
    noisy_speech = speech + scaled_noise
    
    # Normalize to prevent clipping
    max_abs = np.max(np.abs(noisy_speech))
    if max_abs > 1.0:
        noisy_speech = noisy_speech / max_abs * 0.9
    
    return noisy_speech

def main():
    # Define paths
    base_dir = "/home/hkngae/COMP5412/AudioData"
    noise_dir = os.path.join(base_dir, "AudioSet_CSV/audioset_classes")
    speech_dir = os.path.join(base_dir, "PTDB_TUG/SPEECH_DATA")
    output_dir = "/home/hkngae/COMP5412/NoisySpeechDataset"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define SNR levels
    snr_levels = [20, 15, 10, 5, 0, -5]  # in dB
    
    # Get list of noise classes
    noise_classes = os.listdir(noise_dir)
    
    # Get list of speech files
    speech_files = []
    for gender in ["FEMALE", "MALE"]:
        gender_path = os.path.join(speech_dir, gender)
        speech_files.extend(glob.glob(os.path.join(gender_path, "**", "*.wav"), recursive=True))
    
    print(f"Found {len(speech_files)} speech files and {len(noise_classes)} noise classes")
    
    # Prepare metadata file
    metadata_file = os.path.join(output_dir, "metadata.csv")
    with open(metadata_file, 'w') as f:
        f.write("noisy_file,clean_file,noise_file,snr\n")
    
    # Process files
    for speech_file in tqdm(speech_files):
        # Load speech
        speech, speech_sr = load_audio(speech_file)
        
        # Get speech file metadata
        gender = "FEMALE" if "/FEMALE/" in speech_file else "MALE"
        speech_filename = os.path.basename(speech_file)
        speech_id = os.path.splitext(speech_filename)[0]
        
        # For each noise class
        for noise_class in noise_classes:
            noise_files = glob.glob(os.path.join(noise_dir, noise_class, "*.wav"))
            if not noise_files:
                continue
            
            # Select a random noise file
            noise_file = random.choice(noise_files)
            noise, noise_sr = load_audio(noise_file)
            
            # For each SNR level
            for snr in snr_levels:
                # Mix audio
                noisy_speech = mix_audio(speech, noise, snr)
                
                # More structured approach - modify your directory structure in main()
                output_clean_dir = os.path.join(output_dir, "clean", gender)
                output_noisy_dir = os.path.join(output_dir, "noisy", noise_class, f"SNR_{snr}", gender)
                os.makedirs(output_clean_dir, exist_ok=True)
                os.makedirs(output_noisy_dir, exist_ok=True)

                # Save clean speech once (to avoid duplicates)
                clean_filename = f"{speech_id}_clean.wav"
                clean_path = os.path.join(output_clean_dir, clean_filename)
                if not os.path.exists(clean_path):
                    sf.write(clean_path, speech, speech_sr)

                # Save noisy speech
                noisy_filename = f"{speech_id}_mixed_{noise_class}_{snr}dB.wav"
                noisy_path = os.path.join(output_noisy_dir, noisy_filename)
                sf.write(noisy_path, noisy_speech, speech_sr)
                
                # Update metadata
                with open(metadata_file, 'a') as f:
                    f.write(f"{noisy_path},{speech_file},{noise_file},{snr}\n")

if __name__ == "__main__":
    main()