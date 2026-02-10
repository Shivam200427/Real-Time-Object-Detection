"""
Feature-Level Fusion Module — Mid-Level Fusion.

This module fuses RGB and Thermal features at the intermediate feature map level
using attention mechanisms. This is NOT early fusion (raw pixel concat) or 
late fusion (prediction-level merge).

Components:
1. Channel Attention (SE-style) — Learns modality importance per channel
2. Spatial Attention — Learns where to focus across spatial dimensions
3. Cross-Modal Attention — Enables information exchange between modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Learns which feature channels are most informative for each modality.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Channel-attended features (B, C, H, W)
        """
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial attention module that learns where to focus.
    Uses channel-wise pooling followed by convolution.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Spatially-attended features (B, C, H, W)
        """
        # Channel-wise pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # Concatenate and convolve
        pool_cat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attention = self.conv(pool_cat)  # (B, 1, H, W)
        
        return x * attention


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for exchanging information between RGB and Thermal features.
    Uses a simplified multi-head attention mechanism with spatial features.
    
    The RGB features generate Queries, while Thermal features generate Keys and Values
    (and vice versa), enabling cross-modal information flow.
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        # Projections for RGB -> Query, Thermal -> Key/Value
        self.query_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.key_thermal = nn.Conv2d(in_channels, in_channels, 1)
        self.value_thermal = nn.Conv2d(in_channels, in_channels, 1)
        
        # Projections for Thermal -> Query, RGB -> Key/Value
        self.query_thermal = nn.Conv2d(in_channels, in_channels, 1)
        self.key_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_proj_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.out_proj_thermal = nn.Conv2d(in_channels, in_channels, 1)
        
        # Layer norms
        self.norm_rgb = nn.GroupNorm(num_groups=num_heads, num_channels=in_channels)
        self.norm_thermal = nn.GroupNorm(num_groups=num_heads, num_channels=in_channels)
        
        self.scale = self.head_dim ** -0.5
    
    def _attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """Compute scaled dot-product attention on spatial features.
        
        Uses spatial reduction (pooling) when feature maps are large
        to keep GPU memory usage manageable on limited VRAM GPUs.
        """
        B, C, H, W = query.shape
        
        # Spatial reduction for large feature maps to avoid OOM
        # If H*W > max_spatial, pool key/value down to reduce attention matrix size
        max_spatial = 256  # 16x16 max for key/value (very conservative for 4GB VRAM)
        spatial_size = H * W
        
        if spatial_size > max_spatial:
            # Pool key and value to reduced spatial size
            pool_size = int(max_spatial ** 0.5)  # 16
            k_pooled = F.adaptive_avg_pool2d(key, (pool_size, pool_size))
            v_pooled = F.adaptive_avg_pool2d(value, (pool_size, pool_size))
            S = pool_size * pool_size
            
            q = query.reshape(B, self.num_heads, self.head_dim, spatial_size)
            k = k_pooled.reshape(B, self.num_heads, self.head_dim, S)
            v = v_pooled.reshape(B, self.num_heads, self.head_dim, S)
            
            # Attention: (B, num_heads, H*W, S) — much smaller matrix
            attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            # Apply: (B, num_heads, head_dim, H*W)
            out = torch.matmul(v, attn.transpose(-2, -1))
        else:
            q = query.reshape(B, self.num_heads, self.head_dim, spatial_size)
            k = key.reshape(B, self.num_heads, self.head_dim, spatial_size)
            v = value.reshape(B, self.num_heads, self.head_dim, spatial_size)
            
            attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(v, attn.transpose(-2, -1))
        
        # Reshape back: (B, C, H, W)
        out = out.reshape(B, C, H, W)
        
        return out
    
    def forward(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-modal attention.
        
        Args:
            rgb_feat: RGB features (B, C, H, W)
            thermal_feat: Thermal features (B, C, H, W)
            
        Returns:
            Enhanced (rgb_feat, thermal_feat) with cross-modal information
        """
        # RGB attends to Thermal
        q_rgb = self.query_rgb(rgb_feat)
        k_thermal = self.key_thermal(thermal_feat)
        v_thermal = self.value_thermal(thermal_feat)
        rgb_cross = self.out_proj_rgb(self._attention(q_rgb, k_thermal, v_thermal))
        rgb_enhanced = self.norm_rgb(rgb_feat + rgb_cross)
        
        # Thermal attends to RGB
        q_thermal = self.query_thermal(thermal_feat)
        k_rgb = self.key_rgb(rgb_feat)
        v_rgb = self.value_rgb(rgb_feat)
        thermal_cross = self.out_proj_thermal(self._attention(q_thermal, k_rgb, v_rgb))
        thermal_enhanced = self.norm_thermal(thermal_feat + thermal_cross)
        
        return rgb_enhanced, thermal_enhanced


class FeatureFusionBlock(nn.Module):
    """
    Single-scale feature fusion block.
    Combines RGB and Thermal features at one resolution level using:
    1. Channel Attention on each modality
    2. Spatial Attention on each modality
    3. Cross-Modal Attention between modalities
    4. Learned combination weights
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        
        # Channel attention for each modality
        self.rgb_channel_attn = ChannelAttention(in_channels)
        self.thermal_channel_attn = ChannelAttention(in_channels)
        
        # Spatial attention for each modality
        self.rgb_spatial_attn = SpatialAttention()
        self.thermal_spatial_attn = SpatialAttention()
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(in_channels, num_heads)
        
        # Learnable fusion weights
        self.rgb_weight = nn.Parameter(torch.tensor(0.5))
        self.thermal_weight = nn.Parameter(torch.tensor(0.5))
        
        # Final 1x1 conv to refine fused features
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse RGB and Thermal features at one scale.
        
        Args:
            rgb_feat: (B, C, H, W)
            thermal_feat: (B, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        # Step 1: Channel Attention
        rgb_ca = self.rgb_channel_attn(rgb_feat)
        thermal_ca = self.thermal_channel_attn(thermal_feat)
        
        # Step 2: Spatial Attention
        rgb_sa = self.rgb_spatial_attn(rgb_ca)
        thermal_sa = self.thermal_spatial_attn(thermal_ca)
        
        # Step 3: Cross-Modal Attention
        rgb_cross, thermal_cross = self.cross_attn(rgb_sa, thermal_sa)
        
        # Step 4: Weighted combination
        w_rgb = torch.sigmoid(self.rgb_weight)
        w_thermal = torch.sigmoid(self.thermal_weight)
        fused = w_rgb * rgb_cross + w_thermal * thermal_cross
        
        # Step 5: Refinement
        fused = self.refine(fused)
        
        return fused


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Applies FeatureFusionBlock at each scale level from the dual-stream backbone.
    """
    
    def __init__(self, channel_list: List[int], num_heads: int = 4):
        """
        Args:
            channel_list: List of channel dimensions for each scale
                         e.g., [64, 128, 256, 512] for ResNet-34
            num_heads: Number of attention heads for cross-modal attention
        """
        super().__init__()
        
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(ch, num_heads=max(1, min(num_heads, ch // 8)))
            for ch in channel_list
        ])
    
    def forward(
        self,
        rgb_features: List[torch.Tensor],
        thermal_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Fuse features at all scales.
        
        Args:
            rgb_features: List of RGB feature maps [C1, C2, C3, C4]
            thermal_features: List of thermal feature maps [T1, T2, T3, T4]
            
        Returns:
            List of fused feature maps [F1, F2, F3, F4]
        """
        fused = []
        for rgb_f, thermal_f, fusion_block in zip(
            rgb_features, thermal_features, self.fusion_blocks
        ):
            fused.append(fusion_block(rgb_f, thermal_f))
        
        return fused
