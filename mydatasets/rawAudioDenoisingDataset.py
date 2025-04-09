import torch
import numpy as np
from torch.utils.data import Dataset

class RawAudioDenoisingDataset(Dataset):
    def __init__(self, hf_dataset, fixed_length=None, normalize=True, augment=False):
        self.dataset = hf_dataset
        self.fixed_length = fixed_length  # Set to None to keep original lengths
        self.normalize = normalize
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get audio data
        noisy_audio = sample["noisy_file"]["array"].astype(np.float32)
        clean_audio = sample["clean_file"]["array"].astype(np.float32)
        
        # Data augmentation (optional)
        if self.augment:
            noisy_audio = self._augment_audio(noisy_audio)
        
        # Handle different lengths
        if self.fixed_length is not None:
            noisy_audio = self._adjust_length(noisy_audio, self.fixed_length)
            clean_audio = self._adjust_length(clean_audio, self.fixed_length)
        
        # Normalize if requested
        if self.normalize:
            noisy_audio = self._normalize_audio(noisy_audio)
            clean_audio = self._normalize_audio(clean_audio)
        
        # Convert to tensors
        noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32)
        
        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "original_length": len(sample["noisy_file"]["array"])
        }
    
    def _adjust_length(self, audio, target_length): #2 seconds at 16kHz
        """Adjust audio to target length by padding or truncating"""
        if len(audio) > target_length:
            # Randomly select a segment of target_length
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start+target_length]
        else:
            # Pad with zeros to target_length
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding))
        return audio
    
    def _normalize_audio(self, audio):
        """Normalize audio to the range [-1, 1]"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def _augment_audio(self, audio):
        """Apply simple augmentations to audio"""
        # Random gain
        gain = np.random.uniform(0.8, 1.2)
        return audio * gain

    # Define a collate function for handling batches with variable lengths
    @staticmethod
    def variable_length_collate(batch):
        # Find the maximum length in this batch
        max_len = max([item["noisy"].shape[0] for item in batch])
        
        # Prepare lists for batch data
        noisy_batch = []
        clean_batch = []
        lengths = []
        
        # Pad all samples to the max length
        for item in batch:
            noisy = item["noisy"]
            clean = item["clean"]
            lengths.append(len(noisy))
            
            # Pad if needed
            if len(noisy) < max_len:
                noisy_padded = torch.nn.functional.pad(noisy, (0, max_len - len(noisy)))
                clean_padded = torch.nn.functional.pad(clean, (0, max_len - len(clean)))
                noisy_batch.append(noisy_padded)
                clean_batch.append(clean_padded)
            else:
                noisy_batch.append(noisy)
                clean_batch.append(clean)
        
        # Stack into tensors
        noisy_batch = torch.stack(noisy_batch)
        clean_batch = torch.stack(clean_batch)
        lengths = torch.tensor(lengths)
        
        return {
            "noisy": noisy_batch,
            "clean": clean_batch,
            "lengths": lengths
        }
