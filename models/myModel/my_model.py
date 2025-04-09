import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception1DBlock(nn.Module):
    """Inception block for 1D audio signals with multiple kernel sizes in parallel"""
    def __init__(self, in_channels, out_channels):
        super(Inception1DBlock, self).__init__()
        
        # Different kernel sizes to capture patterns at multiple time scales
        self.branch1x1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        
        self.branch3x3_1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        self.branch3x3_2 = nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        
        self.branch5x5_1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        self.branch5x5_2 = nn.Conv1d(out_channels//4, out_channels//4, kernel_size=5, padding=2)
        
        self.branch_pool = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        output = torch.cat(outputs, 1)
        output = F.relu(self.bn(output))
        
        return output

class InceptionFCN(nn.Module):
    """Fully Convolutional Network with Inception blocks for audio denoising"""
    def __init__(self, residual_learning=True):
        super(InceptionFCN, self).__init__()
        
        self.residual_learning = residual_learning
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Downsample path
        self.inception1 = Inception1DBlock(16, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception2 = Inception1DBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception3 = Inception1DBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Center block
        self.inception4 = Inception1DBlock(128, 256)
        
        # Upsample path with skip connections
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.inception5 = Inception1DBlock(256, 128)  # 256 = 128 + 128 from skip connection
        
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.inception6 = Inception1DBlock(128, 64)   # 128 = 64 + 64 from skip connection
        
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.inception7 = Inception1DBlock(64, 32)    # 64 = 32 + 32 from skip connection
        
        # Final convolution
        self.conv_final = nn.Conv1d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Reshape input from [batch, time] to [batch, channels, time]
        x = x.unsqueeze(1)
        
        # If using residual learning, store the input for later
        input_audio = x if self.residual_learning else None
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Downsample path
        skip1 = self.inception1(x)
        x = self.pool1(skip1)
        
        skip2 = self.inception2(x)
        x = self.pool2(skip2)
        
        skip3 = self.inception3(x)
        x = self.pool3(skip3)
        
        # Center block
        x = self.inception4(x)
        
        # Upsample path with skip connections
        x = self.upconv3(x)
        # Handle potential size mismatch in skip connections
        if x.size() != skip3.size():
            x = F.interpolate(x, size=skip3.size(2), mode='linear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.inception5(x)
        
        x = self.upconv2(x)
        if x.size() != skip2.size():
            x = F.interpolate(x, size=skip2.size(2), mode='linear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.inception6(x)
        
        x = self.upconv1(x)
        if x.size() != skip1.size():
            x = F.interpolate(x, size=skip1.size(2), mode='linear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.inception7(x)
        
        # Final convolution
        x = self.conv_final(x)
        
        # Apply residual learning if enabled
        if self.residual_learning:
            # Handle potential size mismatch between input and output
            if x.size() != input_audio.size():
                input_audio = F.interpolate(input_audio, size=x.size(2), mode='linear', align_corners=False)
            x = input_audio - x  # Model predicts the noise, which we subtract
        
        # Reshape output from [batch, channels, time] back to [batch, time]
        return x.squeeze(1)

