import torch
import numpy as np
from torch.utils.data import Dataset

class RawAudioDenoisingDataset(Dataset):
    def __init__(self, hf_dataset, fixed_length=None, input_length=None, output_length=None, 
                 overlap=0.5, normalize=True, augment=False):
        self.dataset = hf_dataset
        self.fixed_length = fixed_length  # Set to None to keep original lengths
        self.input_length = input_length  # Length for noisy input
        self.output_length = output_length  # Length for clean output
        self.overlap = overlap  # Overlap ratio for sliding window
        self.normalize = normalize
        self.augment = augment
        
        # Validate parameters
        if input_length is not None and output_length is not None:
            if output_length > input_length:
                raise ValueError("output_length should be less than or equal to input_length")
        
        # Create index mapping for sliding window approach
        if input_length is not None:
            self._create_index_mapping()
        
    def _create_index_mapping(self):
        """Create a mapping from new indices to (original_idx, start_pos) for sliding windows"""
        self.index_mapping = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            audio_length = len(sample["noisy"]["array"])
            
            if audio_length <= self.input_length:
                # If audio is shorter than input_length, just use it once
                self.index_mapping.append((idx, 0))
            else:
                # Calculate hop size based on overlap
                hop_size = int(self.input_length * (1 - self.overlap))
                if hop_size <= 0:
                    hop_size = 1  # Ensure at least 1 sample shift
                
                # Create sliding windows
                for start_pos in range(0, audio_length - self.input_length + 1, hop_size):
                    self.index_mapping.append((idx, start_pos))
                
                # Handle the last chunk if it doesn't align perfectly
                last_start = start_pos + hop_size
                if last_start < audio_length and audio_length - last_start > self.input_length / 3:
                    # Include the last chunk if it covers at least 1/3 of the input length
                    # This will need padding in __getitem__
                    self.index_mapping.append((idx, audio_length - self.input_length))
    
    def __len__(self):
        if hasattr(self, 'index_mapping'):
            return len(self.index_mapping)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Handle sliding window indexing if using input_length/output_length
        if hasattr(self, 'index_mapping'):
            original_idx, start_pos = self.index_mapping[idx]
            sample = self.dataset[original_idx]
            
            # Extract audio segments
            audio_length = len(sample["noisy"]["array"])
            
            # Check if we need padding for this segment
            if start_pos + self.input_length > audio_length:
                # Need to pad
                remaining = audio_length - start_pos
                noisy_segment = sample["noisy"]["array"][start_pos:].astype(np.float32)
                noisy_audio = np.pad(noisy_segment, (0, self.input_length - remaining))
            else:
                # Standard case, no padding needed
                noisy_audio = sample["noisy"]["array"][start_pos:start_pos + self.input_length].astype(np.float32)
            
            # For clean audio, use left-aligned approach if output_length is different
            if self.output_length is not None and self.output_length != self.input_length:
                # Use left alignment (no offset) - just take first output_length samples
                clean_start = start_pos
                
                # Handle padding for clean audio if needed
                if clean_start + self.output_length > audio_length:
                    remaining = audio_length - clean_start
                    clean_segment = sample["clean"]["array"][clean_start:].astype(np.float32)
                    clean_audio = np.pad(clean_segment, (0, self.output_length - remaining))
                else:
                    clean_audio = sample["clean"]["array"][clean_start:clean_start + self.output_length].astype(np.float32)
            else:
                if start_pos + self.input_length > audio_length:
                    # Need to pad
                    remaining = audio_length - start_pos
                    clean_segment = sample["clean"]["array"][start_pos:].astype(np.float32)
                    clean_audio = np.pad(clean_segment, (0, self.input_length - remaining))
                else:
                    clean_audio = sample["clean"]["array"][start_pos:start_pos + self.input_length].astype(np.float32)
                
            original_length = len(sample["noisy"]["array"])
        else:
            # Original behavior
            sample = self.dataset[idx]
            noisy_audio = sample["noisy"]["array"].astype(np.float32)
            clean_audio = sample["clean"]["array"].astype(np.float32)
            original_length = len(noisy_audio)
        
        # Data augmentation (optional)
        if self.augment:
            noisy_audio = self._augment_audio(noisy_audio)
        
        # Handle different lengths if not using sliding window approach
        if self.fixed_length is not None and not hasattr(self, 'index_mapping'):
            noisy_audio = self._adjust_length(noisy_audio, self.fixed_length)
            clean_audio = self._adjust_length(clean_audio, self.fixed_length)
        
        # Normalize if requested
        if self.normalize:
            noisy_audio = self._normalize_audio(noisy_audio)
            clean_audio = self._normalize_audio(clean_audio)
        
        # Convert to tensors
        noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32)
        
        # Fix for noise calculation when input_length != output_length
        if self.output_length is not None and self.output_length != self.input_length:
            # Only use the first output_length samples from noisy for noise calculation
            noise = noisy_tensor[:self.output_length] - clean_tensor
        else:
            noise = noisy_tensor - clean_tensor
        
        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "noise": noise,
            "mask": torch.ones_like(clean_tensor),  # Placeholder for mask
            "original_length": original_length,
            "window_start": start_pos if hasattr(self, 'index_mapping') else 0
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
        max_noisy_len = max([item["noisy"].shape[0] for item in batch])
        max_clean_len = max([item["clean"].shape[0] for item in batch])
        
        # Prepare lists for batch data
        noisy_batch = []
        clean_batch = []
        noisy_lengths = []
        clean_lengths = []
        
        # Pad all samples to the max length
        for item in batch:
            noisy = item["noisy"]
            clean = item["clean"]
            noisy_lengths.append(len(noisy))
            clean_lengths.append(len(clean))
            
            # Pad if needed
            if len(noisy) < max_noisy_len:
                noisy_padded = torch.nn.functional.pad(noisy, (0, max_noisy_len - len(noisy)))
                noisy_batch.append(noisy_padded)
            else:
                noisy_batch.append(noisy)
                
            if len(clean) < max_clean_len:
                clean_padded = torch.nn.functional.pad(clean, (0, max_clean_len - len(clean)))
                clean_batch.append(clean_padded)
            else:
                clean_batch.append(clean)
        
        # Stack into tensors
        noisy_batch = torch.stack(noisy_batch)
        clean_batch = torch.stack(clean_batch)
        noisy_lengths = torch.tensor(noisy_lengths)
        clean_lengths = torch.tensor(clean_lengths)
        
        # Create mask tensors: 1 for actual data, 0 for padding
        noisy_mask_batch = torch.zeros_like(noisy_batch)
        clean_mask_batch = torch.zeros_like(clean_batch)
        
        for i, length in enumerate(noisy_lengths):
            noisy_mask_batch[i, :length] = 1.0
            
        for i, length in enumerate(clean_lengths):
            clean_mask_batch[i, :length] = 1.0
        
        return {
            "noisy": noisy_batch,
            "clean": clean_batch,
            "noisy_lengths": noisy_lengths,
            "clean_lengths": clean_lengths,
            "noisy_mask": noisy_mask_batch,
            "clean_mask": clean_mask_batch
        }
