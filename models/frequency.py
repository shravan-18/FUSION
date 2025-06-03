import torch
import torch.nn as nn

class FrequencyAttention(nn.Module):
    """
    Attention mechanism for frequency domain features.
    Applies adaptive channel attention to refine frequency features.
    """
    def __init__(self, in_channels, reduction=16):
        super(FrequencyAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FrequencyBranchProcessor(nn.Module):
    """
    Processes features in the frequency domain.
    Extracts magnitude and phase information, applies transformations to magnitude,
    and reconstructs the features.
    """
    def __init__(self, in_channels, out_channels):
        super(FrequencyBranchProcessor, self).__init__()
        self.freq_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.freq_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.freq_attention = FrequencyAttention(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        
    def forward(self, x):
        # Convert to frequency domain using FFT
        x_freq = torch.fft.rfft2(x, norm="ortho")
        # Extract magnitude and phase
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        
        # Process magnitude in spatial domain
        processed_magnitude = self.freq_conv1(magnitude)
        processed_magnitude = self.relu(self.bn(processed_magnitude))
        processed_magnitude = self.freq_conv2(processed_magnitude)
        processed_magnitude = self.freq_attention(processed_magnitude)
        
        # Reconstruct complex frequency representation
        real_part = processed_magnitude * torch.cos(phase)
        imag_part = processed_magnitude * torch.sin(phase)
        complex_output = torch.complex(real_part, imag_part)
        
        # Convert back to spatial domain
        spatial_output = torch.fft.irfft2(complex_output, s=(x.size(2), x.size(3)), norm="ortho")
        
        return spatial_output

class FrequencyGuidedFusion(nn.Module):
    """
    Fuses spatial and frequency domain features.
    Uses attention mechanisms to adaptively combine features from both domains.
    """
    def __init__(self, channels):
        super(FrequencyGuidedFusion, self).__init__()
        from .attention import CBAM
        
        self.spatial_attention = CBAM(channels)
        # Fix: The number of input channels should match the concatenated features
        # If spatial_features has 'channels' dimensions and frequency_features has 'channels' dimensions
        # Then the concatenated tensor will have '2*channels' dimensions
        self.fusion_conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.activate = nn.PReLU(channels)
        
    def forward(self, spatial_features, frequency_features):
        # Apply attention to spatial features
        attended_spatial = self.spatial_attention(spatial_features)
        
        # Ensure frequency_features has the same number of channels as attended_spatial
        # If needed, use 1x1 convolution to adjust channels
        if attended_spatial.size(1) != frequency_features.size(1):
            # This fixes the channel mismatch error
            adapter = nn.Conv2d(frequency_features.size(1), attended_spatial.size(1), kernel_size=1).to(spatial_features.device)
            frequency_features = adapter(frequency_features)
        
        # Concatenate and fuse features
        fused = torch.cat([attended_spatial, frequency_features], dim=1)
        output = self.fusion_conv(fused)
        output = self.activate(self.norm(output))
        
        return output
    