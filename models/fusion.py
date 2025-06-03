import torch
import torch.nn as nn

class Conv2D_pxp(nn.Module):
    """
    Basic convolutional block with batch normalization and PReLU activation.
    Used throughout the model for feature extraction and refinement.
    """
    def __init__(self, in_ch, out_ch, k, s, p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))
    