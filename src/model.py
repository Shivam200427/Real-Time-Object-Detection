"""
Full Multi-Sensor Detection Model.
Assembles backbone, fusion, neck, and head into one end-to-end model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from src.backbone import DualStreamBackbone
from src.fusion import MultiScaleFeatureFusion
from src.neck import FPNNeck
from src.head import FCOSHead, decode_predictions


class MultiSensorDetector(nn.Module):
    """
    Multi-Sensor Object Detector with Feature-Level Fusion.
    
    Pipeline:
    1. RGB + Thermal images → Dual-stream backbones → Multi-scale features
    2. Multi-scale features → Feature fusion (attention-based) → Fused features
    3. Fused features → FPN neck → Uniform-channel pyramids
    4. Pyramids → FCOS detection head → Predictions
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 5,
        fpn_channels: int = 256,
        num_head_convs: int = 4,
        freeze_backbone_stages: int = 0,
        fusion_num_heads: int = 4,
        use_centerness: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 1. Dual-stream backbone
        self.backbone = DualStreamBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_backbone_stages,
            share_weights=False
        )
        
        # 2. Feature fusion
        self.fusion = MultiScaleFeatureFusion(
            channel_list=self.backbone.out_channels,
            num_heads=fusion_num_heads
        )
        
        # 3. FPN Neck
        self.neck = FPNNeck(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels
        )
        
        # 4. Detection Head
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=num_head_convs,
            use_centerness=use_centerness
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            rgb: RGB images (B, 3, H, W)
            thermal: Thermal images (B, 3, H, W)
            
        Returns:
            Dict with 'cls_scores', 'bbox_preds', 'centerness' lists
        """
        # Extract features from both modalities
        rgb_features, thermal_features = self.backbone(rgb, thermal)
        
        # Fuse features at each scale
        fused_features = self.fusion(rgb_features, thermal_features)
        
        # FPN
        fpn_features = self.neck(fused_features)
        
        # Detection head
        predictions = self.head(fpn_features)
        
        return predictions
    
    def predict(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
        img_size: Tuple[int, int] = (640, 640)
    ) -> List[Dict]:
        """
        Run inference and decode predictions.
        
        Args:
            rgb: RGB images (B, 3, H, W)
            thermal: Thermal images (B, 3, H, W)
            score_thresh: Minimum score threshold
            nms_thresh: NMS IoU threshold
            max_detections: Maximum detections per image
            img_size: Image size for coordinate clipping
            
        Returns:
            List of per-image detection dicts with 'boxes', 'scores', 'labels'
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(rgb, thermal)
            
            results = decode_predictions(
                cls_scores=predictions['cls_scores'],
                bbox_preds=predictions['bbox_preds'],
                centerness=predictions.get('centerness', []),
                strides=[4, 8, 16, 32],
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                max_detections=max_detections,
                img_size=img_size
            )
        
        return results
    
    def get_feature_maps(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Get intermediate feature maps for visualization.
        
        Returns:
            Dict with 'rgb_features', 'thermal_features', 'fused_features', 'fpn_features'
        """
        self.eval()
        with torch.no_grad():
            rgb_features, thermal_features = self.backbone(rgb, thermal)
            fused_features = self.fusion(rgb_features, thermal_features)
            fpn_features = self.neck(fused_features)
        
        return {
            'rgb_features': rgb_features,
            'thermal_features': thermal_features,
            'fused_features': fused_features,
            'fpn_features': fpn_features
        }


def build_model(cfg: dict) -> MultiSensorDetector:
    """Build model from config dict."""
    model_cfg = cfg.get('model', {})
    head_cfg = cfg.get('head', {})
    
    model = MultiSensorDetector(
        backbone_name=model_cfg.get('backbone', 'resnet34'),
        pretrained=model_cfg.get('pretrained', True),
        num_classes=model_cfg.get('num_classes', 5),
        fpn_channels=model_cfg.get('fpn_channels', 256),
        num_head_convs=head_cfg.get('num_convs', 4),
        freeze_backbone_stages=model_cfg.get('freeze_backbone_stages', 0),
        fusion_num_heads=4,
        use_centerness=head_cfg.get('use_centerness', True)
    )
    
    return model
