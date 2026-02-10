"""
FCOS-style Anchor-Free Detection Head.
Performs per-pixel classification and bounding box regression on FPN features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math


class Scale(nn.Module):
    """Learnable scale factor for each FPN level."""
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHead(nn.Module):
    """
    FCOS-style anchor-free detection head.
    
    For each pixel in the FPN feature map, predicts:
    - Classification scores (num_classes)
    - Box regression (left, top, right, bottom distances)
    - Centerness score (how close to object center)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
        num_convs: int = 4,
        use_centerness: bool = True
    ):
        """
        Args:
            in_channels: Input channels from FPN
            num_classes: Number of object classes
            num_convs: Number of conv layers in each branch
            use_centerness: Whether to predict centerness
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_centerness = use_centerness
        
        # Classification branch
        cls_layers = []
        for i in range(num_convs):
            cls_layers.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        
        # Regression branch
        reg_layers = []
        for i in range(num_convs):
            reg_layers.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            reg_layers.append(nn.GroupNorm(32, in_channels))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        
        # Centerness branch (shares tower with classification)
        if use_centerness:
            self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # Per-level scale factors for regression
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(4)])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper initialization."""
        for modules in [self.cls_tower, self.reg_tower]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Bias init for classification (prior probability)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
    
    def forward_single(
        self, 
        x: torch.Tensor, 
        scale: Scale
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single FPN level.
        
        Args:
            x: Feature map (B, C, H, W)
            scale: Scale factor for this level
            
        Returns:
            cls_scores: (B, num_classes, H, W)
            bbox_pred: (B, 4, H, W) — left, top, right, bottom
            centerness: (B, 1, H, W) or None
        """
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        
        cls_scores = self.cls_logits(cls_feat)
        bbox_pred = F.relu(scale(self.reg_pred(reg_feat)))  # distances must be positive
        
        if self.use_centerness:
            centerness = self.centerness(cls_feat)
        else:
            centerness = None
        
        return cls_scores, bbox_pred, centerness
    
    def forward(
        self, 
        features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass for all FPN levels.
        
        Args:
            features: List of FPN feature maps [P1, P2, P3, P4]
            
        Returns:
            Dict with keys:
                'cls_scores': List of (B, num_classes, H_i, W_i)
                'bbox_preds': List of (B, 4, H_i, W_i)
                'centerness': List of (B, 1, H_i, W_i)
        """
        cls_scores = []
        bbox_preds = []
        centerness_preds = []
        
        for feat, scale in zip(features, self.scales):
            cls, bbox, center = self.forward_single(feat, scale)
            cls_scores.append(cls)
            bbox_preds.append(bbox)
            if center is not None:
                centerness_preds.append(center)
        
        result = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
        }
        
        if self.use_centerness:
            result['centerness'] = centerness_preds
        
        return result


def decode_predictions(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    centerness: List[torch.Tensor],
    strides: List[int] = [4, 8, 16, 32],
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    max_detections: int = 100,
    img_size: Tuple[int, int] = (640, 640)
) -> List[Dict]:
    """
    Decode model predictions into bounding boxes.
    
    Args:
        cls_scores: List of (B, num_classes, H_i, W_i) from each FPN level
        bbox_preds: List of (B, 4, H_i, W_i) — left, top, right, bottom distances
        centerness: List of (B, 1, H_i, W_i)
        strides: Stride for each FPN level
        score_thresh: Minimum score to keep
        nms_thresh: NMS IoU threshold
        max_detections: Max number of detections per image
        img_size: Original image size (H, W) for clipping
        
    Returns:
        List of dicts per image, each with 'boxes', 'scores', 'labels'
    """
    batch_size = cls_scores[0].shape[0]
    num_classes = cls_scores[0].shape[1]
    device = cls_scores[0].device
    
    all_results = []
    
    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for level_idx, stride in enumerate(strides):
            cls = cls_scores[level_idx][b]  # (num_classes, H, W)
            bbox = bbox_preds[level_idx][b]  # (4, H, W)
            center = centerness[level_idx][b] if centerness else None  # (1, H, W)
            
            H, W = cls.shape[1], cls.shape[2]
            
            # Generate grid points
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            # Center of each cell
            x_coords = (x_coords + 0.5) * stride
            y_coords = (y_coords + 0.5) * stride
            
            # Get classification scores
            cls_prob = cls.sigmoid()  # (num_classes, H, W)
            if center is not None:
                center_prob = center.sigmoid()  # (1, H, W)
                cls_prob = cls_prob * center_prob
            
            # For each class
            for c in range(num_classes):
                scores = cls_prob[c]  # (H, W)
                mask = scores > score_thresh
                
                if mask.sum() == 0:
                    continue
                
                # Get positions above threshold
                scores_f = scores[mask]
                x_f = x_coords[mask]
                y_f = y_coords[mask]
                
                l = bbox[0][mask]  # left
                t = bbox[1][mask]  # top
                r = bbox[2][mask]  # right
                bot = bbox[3][mask]  # bottom
                
                # Convert LTRB to XYXY
                x1 = (x_f - l).clamp(min=0, max=img_size[1])
                y1 = (y_f - t).clamp(min=0, max=img_size[0])
                x2 = (x_f + r).clamp(min=0, max=img_size[1])
                y2 = (y_f + bot).clamp(min=0, max=img_size[0])
                
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                labels = torch.full((len(scores_f),), c, dtype=torch.long, device=device)
                
                all_boxes.append(boxes)
                all_scores.append(scores_f)
                all_labels.append(labels)
        
        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes)
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            
            # NMS per class
            keep_mask = torch.zeros(len(all_scores), dtype=torch.bool, device=device)
            for c in range(num_classes):
                c_mask = all_labels == c
                if c_mask.sum() == 0:
                    continue
                c_boxes = all_boxes[c_mask]
                c_scores = all_scores[c_mask]
                
                # torchvision NMS
                from torchvision.ops import nms
                keep_idx = nms(c_boxes, c_scores, nms_thresh)
                c_indices = torch.where(c_mask)[0]
                keep_mask[c_indices[keep_idx]] = True
            
            all_boxes = all_boxes[keep_mask]
            all_scores = all_scores[keep_mask]
            all_labels = all_labels[keep_mask]
            
            # Top-K
            if len(all_scores) > max_detections:
                topk_idx = all_scores.argsort(descending=True)[:max_detections]
                all_boxes = all_boxes[topk_idx]
                all_scores = all_scores[topk_idx]
                all_labels = all_labels[topk_idx]
            
            all_results.append({
                'boxes': all_boxes,
                'scores': all_scores,
                'labels': all_labels
            })
        else:
            all_results.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            })
    
    return all_results
