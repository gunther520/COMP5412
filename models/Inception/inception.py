import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class Inception1d(nn.Module):  
    def __init__(self, in_channels, out_channels, conv_type="normal"):  
        super(Inception1d, self).__init__()  
          
        # Define channel allocation  
        self.out_channels = out_channels  
        channels_per_branch = out_channels // 4  
        remaining_channels = out_channels - 3 * channels_per_branch  
          
        # Branch 1: 1x1 convolution  
        self.branch1 = nn.Conv1d(in_channels, channels_per_branch, kernel_size=1)  
          
        # Branch 2: 1x1 convolution followed by 3x1 convolution  
        self.branch2_1 = nn.Conv1d(in_channels, channels_per_branch, kernel_size=1)  
        self.branch2_2 = nn.Conv1d(channels_per_branch, channels_per_branch, kernel_size=3, padding=1)  
          
        # Branch 3: 1x1 convolution followed by 5x1 convolution  
        self.branch3_1 = nn.Conv1d(in_channels, channels_per_branch, kernel_size=1)  
        self.branch3_2 = nn.Conv1d(channels_per_branch, channels_per_branch, kernel_size=5, padding=2)  
          
        # Branch 4: max pooling followed by 1x1 convolution  
        self.branch4_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)  
        self.branch4_2 = nn.Conv1d(in_channels, remaining_channels, kernel_size=1)  
          
        # Normalization layer  
        self.conv_type = conv_type  
        NORM_CHANNELS = 8  
          
        if conv_type == "gn":  
            assert(out_channels % NORM_CHANNELS == 0)  
            self.norm = nn.GroupNorm(out_channels // NORM_CHANNELS, out_channels)  
        elif conv_type == "bn":  
            self.norm = nn.BatchNorm1d(out_channels, momentum=0.01)  
              
    def forward(self, x):  
        # Process each branch  
        branch1 = self.branch1(x)  
          
        branch2 = self.branch2_1(x)  
        branch2 = self.branch2_2(branch2)  
          
        branch3 = self.branch3_1(x)  
        branch3 = self.branch3_2(branch3)  
          
        branch4 = self.branch4_1(x)  
        branch4 = self.branch4_2(branch4)  
          
        # Concatenate outputs along channel dimension  
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)  
          
        # Apply normalization if specified  
        if self.conv_type == "gn" or self.conv_type == "bn":  
            outputs = F.relu(self.norm(outputs))  
        else:  
            outputs = F.leaky_relu(outputs)  
              
        return outputs  
      
    def get_input_size(self, output_size):  
        # All branches maintain the same temporal dimension  
        return output_size  
          
    def get_output_size(self, input_size):  
        # All branches maintain the same temporal dimension  
        return input_size