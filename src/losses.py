"""
Loss functions for FCOS-style detection.
- Focal Loss for classification (handles class imbalance)
- GIoU Loss for box regression
- BCE for centerness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for classification to handle class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, num_classes) raw logits
            target: (N,) class labels (0-indexed)
            
        Returns:
            Scalar focal loss
        """
        num_classes = pred.shape[1]
        
        # One-hot encode targets
        target_onehot = torch.zeros_like(pred)
        if target.numel() > 0:
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        pred_sigmoid = pred.sigmoid()
        
        pt = target_onehot * pred_sigmoid + (1 - target_onehot) * (1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = target_onehot * self.alpha + (1 - target_onehot) * (1 - self.alpha)
        
        bce = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
        
        loss = alpha_weight * focal_weight * bce
        
        return loss.sum() / max(target.numel(), 1)


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression."""
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, 4) predicted boxes [x1, y1, x2, y2]
            target: (N, 4) target boxes [x1, y1, x2, y2]
            
        Returns:
            Scalar GIoU loss (1 - GIoU)
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0
        
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        # Union
        area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = area_pred + area_target - inter_area
        
        iou = inter_area / (union + 1e-6)
        
        # Enclosing box
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        
        loss = 1 - giou
        return loss.mean()


class DetectionLoss(nn.Module):
    """
    Combined detection loss for FCOS-style detector.
    Total = cls_weight * FocalLoss + reg_weight * GIoULoss + center_weight * BCELoss
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        centerness_weight: float = 1.0,
        strides: List[int] = [4, 8, 16, 32]
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.giou_loss = GIoULoss()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.centerness_weight = centerness_weight
        self.strides = strides
    
    def _generate_targets(
        self,
        targets: List[Dict],
        feature_sizes: List[Tuple[int, int]],
        img_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate per-pixel targets for all FPN levels.
        
        For each pixel location, assign it to a ground-truth box if the pixel
        falls inside the box (FCOS-style target assignment).
        """
        device = targets[0]['boxes'].device if targets[0]['boxes'].numel() > 0 else 'cpu'
        batch_size = len(targets)
        
        all_cls_targets = []
        all_reg_targets = []
        all_centerness_targets = []
        
        # Size ranges for each FPN level
        size_ranges = [(0, 64), (64, 128), (128, 256), (256, 1e8)]
        
        for level_idx, (stride, (H, W)) in enumerate(zip(self.strides, feature_sizes)):
            min_size, max_size = size_ranges[level_idx]
            
            cls_target = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)
            reg_target = torch.zeros(batch_size, 4, H, W, device=device)
            centerness_target = torch.zeros(batch_size, 1, H, W, device=device)
            
            for b in range(batch_size):
                boxes = targets[b]['boxes']  # (N, 4) [x1, y1, x2, y2]
                labels = targets[b]['labels']  # (N,)
                
                if boxes.numel() == 0:
                    continue
                
                # Grid coordinates
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(H, device=device, dtype=torch.float32),
                    torch.arange(W, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                x_coords = (x_coords + 0.5) * stride  # center x
                y_coords = (y_coords + 0.5) * stride  # center y
                
                for obj_idx in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[obj_idx]
                    label = labels[obj_idx]
                    
                    # Size filtering
                    obj_size = max(x2 - x1, y2 - y1)
                    if obj_size < min_size or obj_size >= max_size:
                        continue
                    
                    # Find pixels inside this box
                    l = x_coords - x1
                    t = y_coords - y1
                    r = x2 - x_coords
                    bot = y2 - y_coords
                    
                    inside = (l > 0) & (t > 0) & (r > 0) & (bot > 0)
                    
                    if inside.sum() == 0:
                        continue
                    
                    # Assign target
                    cls_target[b][inside] = label + 1  # +1 because 0 is background
                    reg_target[b, 0][inside] = l[inside]
                    reg_target[b, 1][inside] = t[inside]
                    reg_target[b, 2][inside] = r[inside]
                    reg_target[b, 3][inside] = bot[inside]
                    
                    # Centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
                    lr = torch.min(l[inside], r[inside]) / (torch.max(l[inside], r[inside]) + 1e-6)
                    tb = torch.min(t[inside], bot[inside]) / (torch.max(t[inside], bot[inside]) + 1e-6)
                    centerness_target[b, 0][inside] = torch.sqrt(lr * tb)
            
            all_cls_targets.append(cls_target)
            all_reg_targets.append(reg_target)
            all_centerness_targets.append(centerness_target)
        
        return all_cls_targets, all_reg_targets, all_centerness_targets
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total detection loss.
        
        Args:
            predictions: Dict from detection head with 'cls_scores', 'bbox_preds', 'centerness'
            targets: List of per-image targets with 'boxes' and 'labels'
            
        Returns:
            Dict with 'total_loss', 'cls_loss', 'reg_loss', 'centerness_loss'
        """
        cls_scores = predictions['cls_scores']
        bbox_preds = predictions['bbox_preds']
        centerness_preds = predictions.get('centerness', None)
        
        # Get feature map sizes
        feature_sizes = [(cs.shape[2], cs.shape[3]) for cs in cls_scores]
        
        # Generate targets
        cls_targets, reg_targets, centerness_targets = self._generate_targets(
            targets, feature_sizes
        )
        
        total_cls_loss = 0
        total_reg_loss = 0
        total_center_loss = 0
        num_pos = 0
        
        for level_idx in range(len(cls_scores)):
            B, C, H, W = cls_scores[level_idx].shape
            
            # Reshape predictions
            cls_pred = cls_scores[level_idx].permute(0, 2, 3, 1).reshape(-1, C)
            bbox_pred = bbox_preds[level_idx]
            
            cls_target = cls_targets[level_idx].reshape(-1)
            reg_target = reg_targets[level_idx]
            center_target = centerness_targets[level_idx]
            
            # Positive mask (non-background)
            pos_mask = cls_target > 0
            num_pos += pos_mask.sum()
            
            # Classification loss (all pixels)
            # Convert targets: 0 is background, 1..num_classes are object classes
            cls_target_adj = cls_target.clone()
            cls_target_adj[pos_mask] = cls_target_adj[pos_mask] - 1
            cls_target_adj[~pos_mask] = 0
            
            total_cls_loss += self.focal_loss(cls_pred, cls_target_adj)
            
            # Regression loss (only positive pixels)
            if pos_mask.sum() > 0:
                # Convert LTRB predictions to XYXY for each positive pixel
                stride = self.strides[level_idx]
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(H, device=cls_pred.device, dtype=torch.float32),
                    torch.arange(W, device=cls_pred.device, dtype=torch.float32),
                    indexing='ij'
                )
                x_coords = (x_coords + 0.5) * stride
                y_coords = (y_coords + 0.5) * stride
                
                for b_idx in range(B):
                    b_pos_mask = cls_targets[level_idx][b_idx] > 0
                    if b_pos_mask.sum() == 0:
                        continue
                    
                    pred_l = bbox_pred[b_idx, 0][b_pos_mask]
                    pred_t = bbox_pred[b_idx, 1][b_pos_mask]
                    pred_r = bbox_pred[b_idx, 2][b_pos_mask]
                    pred_b = bbox_pred[b_idx, 3][b_pos_mask]
                    
                    x_pos = x_coords[b_pos_mask]
                    y_pos = y_coords[b_pos_mask]
                    
                    pred_boxes = torch.stack([
                        x_pos - pred_l, y_pos - pred_t,
                        x_pos + pred_r, y_pos + pred_b
                    ], dim=1)
                    
                    tgt_l = reg_target[b_idx, 0][b_pos_mask]
                    tgt_t = reg_target[b_idx, 1][b_pos_mask]
                    tgt_r = reg_target[b_idx, 2][b_pos_mask]
                    tgt_b = reg_target[b_idx, 3][b_pos_mask]
                    
                    tgt_boxes = torch.stack([
                        x_pos - tgt_l, y_pos - tgt_t,
                        x_pos + tgt_r, y_pos + tgt_b
                    ], dim=1)
                    
                    total_reg_loss += self.giou_loss(pred_boxes, tgt_boxes)
                
                # Centerness loss
                if centerness_preds is not None:
                    center_pred = centerness_preds[level_idx].reshape(-1)
                    center_tgt = center_target.reshape(-1)
                    pos_flat = cls_target > 0
                    
                    if pos_flat.sum() > 0:
                        total_center_loss += F.binary_cross_entropy_with_logits(
                            center_pred[pos_flat], center_tgt[pos_flat]
                        )
        
        num_pos = max(num_pos.float(), 1.0)
        
        total_loss = (
            self.cls_weight * total_cls_loss +
            self.reg_weight * total_reg_loss +
            self.centerness_weight * total_center_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'centerness_loss': total_center_loss,
            'num_pos': num_pos
        }
