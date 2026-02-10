"""
Feature Pyramid Network (FPN) Neck.
Produces multi-scale feature maps with uniform channel dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network that takes multi-scale features and produces
    feature maps with a uniform channel dimension at each scale.
    
    Input: [F1(/4), F2(/8), F3(/16), F4(/32)] with varying channels
    Output: [P1(/4), P2(/8), P3(/16), P4(/32)] all with `out_channels`
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        """
        Args:
            in_channels_list: Channel dims for each input level, e.g. [64, 128, 256, 512]
            out_channels: Output channel dimension (uniform across all levels)
        """
        super().__init__()
        
        self.num_levels = len(in_channels_list)
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # Top-down pathway smooth convs (3x3 conv after merging)
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Build feature pyramid.
        
        Args:
            features: List of feature maps from backbone/fusion [F1, F2, F3, F4]
                     Each F_i has shape (B, C_i, H_i, W_i)
                     
        Returns:
            List of FPN outputs [P1, P2, P3, P4]
            Each P_i has shape (B, out_channels, H_i, W_i)
        """
        assert len(features) == self.num_levels
        
        # Lateral connections
        laterals = [
            lat_conv(feat) 
            for feat, lat_conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway (from coarsest to finest)
        for i in range(self.num_levels - 1, 0, -1):
            # Upsample coarser level and add to finer level
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[2:],
                mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Smooth convolutions
        outputs = [
            smooth_conv(lat)
            for lat, smooth_conv in zip(laterals, self.smooth_convs)
        ]
        
        return outputs
