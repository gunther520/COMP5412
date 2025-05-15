import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionLayer(nn.Module):  
    def __init__(self, n_inputs, n_outputs, kernel_sizes=[3, 5, 7, 11], stride=1, conv_type="bn", transpose=False):  
        super(InceptionLayer, self).__init__()  
        self.transpose = transpose  
        self.stride = stride  
        self.kernel_sizes = kernel_sizes  
        self.conv_type = conv_type  
          
        # How many channels should be normalized as one group if GroupNorm is activated  
        NORM_CHANNELS = 8  
          
        # Calculate channels per branch - divide output channels among branches  
        channels_per_branch = n_outputs // len(kernel_sizes)  
        remainder = n_outputs % len(kernel_sizes) # Simpler way to get remainder
        self.branch_channels = [channels_per_branch + (1 if i < remainder else 0) for i in range(len(kernel_sizes))]  
          
        # Create parallel convolution branches with different kernel sizes  
        self.branches = nn.ModuleList()  
        for i, k_size in enumerate(kernel_sizes):  
            if self.transpose:  
                # Fixed padding for ConvTranspose1d to ensure consistent output lengths
                # L_out = (L_in - 1) * stride - 2 * padding + k_size + output_padding
                # With padding = (k_size - 1) // 2 and output_padding = 0:
                # L_out = (L_in - 1) * stride + 1 (for odd k_size)
                branch = nn.ConvTranspose1d(  
                    n_inputs, self.branch_channels[i], k_size,   
                    stride, padding=(k_size - 1) // 2,
                    output_padding=0 # Explicitly set output_padding
                )  
            else:  
                # Each branch gets the same input but uses different kernel size  
                # Padding is (kernel_size - 1) // 2 to maintain the same output size (for stride=1)
                # or for ceil(L_in / stride) (for stride > 1)
                branch = nn.Conv1d(  
                    n_inputs, self.branch_channels[i], k_size,   
                    stride, padding=(k_size - 1) // 2  
                )  
            self.branches.append(branch)  
          
        # Normalization and activation  
        if conv_type == "gn":  
            if n_outputs % NORM_CHANNELS != 0:
                # Provide a more informative error if assertion fails
                raise ValueError(f"For GroupNorm (gn), n_outputs ({n_outputs}) must be divisible by NORM_CHANNELS ({NORM_CHANNELS}).")
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)  
        elif conv_type == "bn":  
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)  
        elif conv_type != "normal": # Added check for invalid conv_type
            raise ValueError(f"Unknown conv_type: {conv_type}. Expected 'bn', 'gn', or 'normal'.")
      
    def forward(self, x):  
        #print("Inception input shape tett", x.shape) # Kept for debugging, can be removed
        # Process input through each branch  
        branch_outputs = [branch(x) for branch in self.branches]  
          
        # Concatenate outputs along channel dimension  
        out = torch.cat(branch_outputs, dim=1)  
          
        # Apply normalization and activation  
        if self.conv_type == "gn" or self.conv_type == "bn":  
            out = F.relu(self.norm(out))  
        else:  # This implies conv_type == "normal" due to init check
            # assert(self.conv_type == "normal") # Assertion already implicitly handled by init
            out = F.leaky_relu(out)  
        #print("Inception output shape", out.shape) # Kept for debugging, can be removed
        return out  
      
    def get_input_size(self, output_size):  
        """  
        Calculate required input size to get desired output size.  
        This is L_in such that get_output_size(L_in) == output_size.
        """  
        if not self.transpose:
            # For Conv1d with padding=(k-1)//2, L_out = ceil(L_in / stride)
            # So, L_out = ((L_in - 1) // stride) + 1
            # To find L_in for a given L_out:
            # Smallest L_in is (L_out - 1) * stride + 1
            curr_size = (output_size - 1) * self.stride + 1
            # The max_kernel term was incorrect as it made this function not an inverse of get_output_size.
        else: # transpose = True
            # For ConvTranspose1d with padding=(k-1)//2 and output_padding=0,
            # L_out = (L_in - 1) * stride + 1
            # To find L_in for a given L_out:
            # L_in - 1 = (L_out - 1) / stride
            # L_in = ((L_out - 1) // stride) + 1
            # This requires (L_out - 1) to be non-negative and divisible by stride.
            curr_size = output_size
            if curr_size < 1:
                 raise ValueError("output_size must be at least 1 for transpose=True.")
            if self.stride == 0:
                 raise ValueError("stride cannot be zero.") # PyTorch layers also enforce this
            if (curr_size - 1) % self.stride != 0:
                raise ValueError(
                    f"For transpose=True, (output_size - 1) ({curr_size - 1}) must be divisible by stride ({self.stride})."
                )
            curr_size = ((curr_size - 1) // self.stride) + 1
            # The max_kernel term was incorrect.
              
        if curr_size <= 0: # Final check, though individual paths might catch earlier
            raise ValueError(f"Calculated input_size ({curr_size}) must be positive.")
        return curr_size  
          
    def get_output_size(self, input_size):  
        """  
        Calculate output size for a given input size.  
        """  
        if self.transpose:
            # For ConvTranspose1d with padding=(k-1)//2 and output_padding=0:
            # L_out = (L_in - 1) * stride + 1
            if input_size < 1: # Changed from input_size > 1 to input_size >= 1
                raise ValueError("input_size must be at least 1 for transpose=True.")
            curr_size = (input_size - 1) * self.stride + 1
            # The line involving max_kernel was equivalent to curr_size = curr_size, removed for clarity.
        else: # not self.transpose
            # For Conv1d with padding=(k-1)//2: L_out = ceil(L_in / stride)
            # L_out = ((L_in - 1) // stride) + 1
            if input_size < 1: # Added for consistency
                raise ValueError("input_size must be at least 1 for non-transpose.")
            curr_size = input_size
            # The line involving max_kernel was equivalent to curr_size = curr_size, removed for clarity.
            # The assertion ((curr_size - 1) % self.stride == 0) was incorrect and removed.
            if self.stride == 0:
                 raise ValueError("stride cannot be zero.")
            curr_size = ((curr_size - 1) // self.stride) + 1
              
        return curr_size