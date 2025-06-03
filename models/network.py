import torch
import torch.nn as nn
from .fusion import Conv2D_pxp
from .attention import CBAM
from .frequency import FrequencyBranchProcessor, FrequencyGuidedFusion

class EnhancedCC_Module(nn.Module):
    """
    Enhanced Color Correction Module with Frequency-Aware Attention for underwater images.
    This is the main network architecture for the FUSION model.
    """
    def __init__(self):
        super(EnhancedCC_Module, self).__init__()   

        print("Enhanced Color correction module with Frequency-Aware Attention for underwater images")

        # Spatial domain processing (original)
        self.layer1_1 = Conv2D_pxp(1, 32, 3, 1, 1)  # For R channel
        self.layer1_2 = Conv2D_pxp(1, 32, 5, 1, 2)  # For G channel
        self.layer1_3 = Conv2D_pxp(1, 32, 7, 1, 3)  # For B channel

        self.layer2_1 = Conv2D_pxp(96, 32, 3, 1, 1)
        self.layer2_2 = Conv2D_pxp(96, 32, 5, 1, 2)
        self.layer2_3 = Conv2D_pxp(96, 32, 7, 1, 3)
        
        self.local_attn_r = CBAM(64)
        self.local_attn_g = CBAM(64)
        self.local_attn_b = CBAM(64)

        self.layer3_1 = Conv2D_pxp(192, 1, 3, 1, 1)
        self.layer3_2 = Conv2D_pxp(192, 1, 5, 1, 2)
        self.layer3_3 = Conv2D_pxp(192, 1, 7, 1, 3)

        # Frequency domain processing (new)
        self.freq_branch_r = FrequencyBranchProcessor(1, 16)
        self.freq_branch_g = FrequencyBranchProcessor(1, 16)
        self.freq_branch_b = FrequencyBranchProcessor(1, 16)
        
        # Feature fusion after channel-specific processing
        self.fusion_r = FrequencyGuidedFusion(1)
        self.fusion_g = FrequencyGuidedFusion(1)
        self.fusion_b = FrequencyGuidedFusion(1)

        # Refined decoder with frequency-aware processing
        self.d_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.PReLU(32)

        # Enhanced global attention with frequency information
        self.global_attn_rgb = CBAM(38)
        
        # Fix: Calculate correct input channels for freq_spatial_fusion (32 from output_d1 + 48 from freq_features (16*3))
        self.freq_spatial_fusion = nn.Conv2d(32 + 48, 35, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(35)
        self.fusion_relu = nn.PReLU(35)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=38, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=3)
        self.d_relu2 = nn.PReLU(3)

        # Channel re-calibration for adaptively balancing RGB channels
        self.channel_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 6, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input_1 = torch.unsqueeze(input[:,0,:,:], dim=1)  # R
        input_2 = torch.unsqueeze(input[:,1,:,:], dim=1)  # G
        input_3 = torch.unsqueeze(input[:,2,:,:], dim=1)  # B
        
        # Spatial domain processing (original path)
        #layer 1
        l1_1 = self.layer1_1(input_1) 
        l1_2 = self.layer1_2(input_2) 
        l1_3 = self.layer1_3(input_3) 

        #Input to layer 2
        input_l2 = torch.cat((l1_1, l1_2, l1_3), 1)
        
        #layer 2
        l2_1 = self.layer2_1(input_l2) 
        l2_1 = self.local_attn_r(torch.cat((l2_1, l1_1), 1))

        l2_2 = self.layer2_2(input_l2) 
        l2_2 = self.local_attn_g(torch.cat((l2_2, l1_2), 1))

        l2_3 = self.layer2_3(input_l2) 
        l2_3 = self.local_attn_b(torch.cat((l2_3, l1_3), 1))
        
        #Input to layer 3
        input_l3 = torch.cat((l2_1, l2_2, l2_3), 1)
        
        #layer 3
        l3_1 = self.layer3_1(input_l3) 
        l3_2 = self.layer3_2(input_l3) 
        l3_3 = self.layer3_3(input_l3) 

        # Frequency domain processing (new path)
        freq_1 = self.freq_branch_r(input_1)
        freq_2 = self.freq_branch_g(input_2)
        freq_3 = self.freq_branch_b(input_3)
        
        # Fusion of spatial and frequency information for each channel
        fused_1 = self.fusion_r(l3_1, freq_1)
        fused_2 = self.fusion_g(l3_2, freq_2)
        fused_3 = self.fusion_b(l3_3, freq_3)
        
        # Add residual connections (like in original)
        temp_d1 = torch.add(input_1, fused_1)
        temp_d2 = torch.add(input_2, fused_2)
        temp_d3 = torch.add(input_3, fused_3)

        # Concatenate channel features
        input_d1 = torch.cat((temp_d1, temp_d2, temp_d3), 1)
        
        # Original decoder path
        output_d1 = self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))
        
        # Frequency-aware feature fusion in decoder
        freq_features = torch.cat((freq_1, freq_2, freq_3), 1)
        
        # Concatenate and then apply fusion
        concat_features = torch.cat((output_d1, freq_features), 1)
        fusion_features = self.fusion_relu(self.fusion_norm(self.freq_spatial_fusion(concat_features)))
        
        # Apply global attention
        output_d1_with_attn = self.global_attn_rgb(torch.cat((fusion_features, input_d1), 1))
        final_output = self.d_relu2(self.d_bn2(self.d_conv2(output_d1_with_attn)))
        
        # Apply channel re-calibration for final adjustment
        calibration_weights = self.channel_calibration(final_output)
        final_output = final_output * calibration_weights
        
        return final_output
    