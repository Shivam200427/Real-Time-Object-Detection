"""
Dual-Stream Backbone for RGB and Thermal feature extraction.
Each modality gets its own ResNet-based feature extractor.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Dict


class ResNetBackbone(nn.Module):
    """
    ResNet backbone that extracts multi-scale feature maps.
    Returns features from 4 stages (layer1 through layer4).
    """
    
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = True,
        freeze_stages: int = 0
    ):
        """
        Args:
            model_name: One of 'resnet18', 'resnet34', 'resnet50'
            pretrained: Use ImageNet pretrained weights
            freeze_stages: Number of initial stages to freeze (0-4)
        """
        super().__init__()
        
        # Load pretrained ResNet
        weights = "DEFAULT" if pretrained else None
        if model_name == "resnet18":
            resnet = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            resnet = models.resnet34(weights=weights)
        elif model_name == "resnet50":
            resnet = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")
        
        # Split into stages
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = resnet.layer1  # /4,  64 channels (resnet18/34) or 256 (resnet50)
        self.stage2 = resnet.layer2  # /8,  128 channels or 512
        self.stage3 = resnet.layer3  # /16, 256 channels or 1024
        self.stage4 = resnet.layer4  # /32, 512 channels or 2048
        
        # Channel dimensions for each stage
        if model_name in ["resnet18", "resnet34"]:
            self.out_channels = [64, 128, 256, 512]
        else:
            self.out_channels = [256, 512, 1024, 2048]
        
        # Freeze stages
        if freeze_stages >= 1:
            self._freeze_module(self.stage0)
            self._freeze_module(self.stage1)
        if freeze_stages >= 2:
            self._freeze_module(self.stage2)
        if freeze_stages >= 3:
            self._freeze_module(self.stage3)
        if freeze_stages >= 4:
            self._freeze_module(self.stage4)
    
    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            List of 4 feature maps at different scales
        """
        x = self.stage0(x)
        c1 = self.stage1(x)   # /4
        c2 = self.stage2(c1)  # /8
        c3 = self.stage3(c2)  # /16
        c4 = self.stage4(c3)  # /32
        
        return [c1, c2, c3, c4]


class DualStreamBackbone(nn.Module):
    """
    Dual-stream backbone with separate feature extractors for RGB and Thermal.
    
    This is the first stage of the mid-level fusion pipeline.
    Each modality is processed independently to extract modality-specific features.
    """
    
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = True,
        freeze_stages: int = 0,
        share_weights: bool = False
    ):
        """
        Args:
            model_name: Backbone architecture name
            pretrained: Use pretrained weights
            freeze_stages: Number of stages to freeze
            share_weights: If True, both streams share the same backbone
        """
        super().__init__()
        
        self.rgb_backbone = ResNetBackbone(model_name, pretrained, freeze_stages)
        
        if share_weights:
            self.thermal_backbone = self.rgb_backbone
        else:
            self.thermal_backbone = ResNetBackbone(model_name, pretrained, freeze_stages)
        
        self.out_channels = self.rgb_backbone.out_channels
    
    def forward(
        self, 
        rgb: torch.Tensor, 
        thermal: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract features from both modalities.
        
        Args:
            rgb: RGB input (B, 3, H, W)
            thermal: Thermal input (B, 3, H, W) — single channel repeated to 3
            
        Returns:
            Tuple of (rgb_features, thermal_features), each a list of 4 feature maps
        """
        rgb_features = self.rgb_backbone(rgb)
        thermal_features = self.thermal_backbone(thermal)
        
        return rgb_features, thermal_features
    
    def get_feature_info(self) -> List[Dict]:
        """Get information about output feature maps."""
        strides = [4, 8, 16, 32]
        return [
            {"channels": c, "stride": s} 
            for c, s in zip(self.out_channels, strides)
        ]
