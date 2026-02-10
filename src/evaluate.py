"""
Evaluation & Metrics for Multi-Sensor Detector.
Computes mAP, precision, recall, and per-class metrics.
"""

import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import compute_iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.
    
    Args:
        recall: Sorted recall values
        precision: Corresponding precision values
        
    Returns:
        Average Precision value
    """
    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def compute_ap_all_points(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using all-point interpolation (COCO-style)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum areas under curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_detections(
    all_predictions: List[Dict],
    all_targets: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.5],
    class_names: List[str] = None
) -> Dict:
    """
    Evaluate detection results against ground truth.
    
    Args:
        all_predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        all_targets: List of target dicts with 'boxes', 'labels'
        num_classes: Number of object classes
        iou_thresholds: IoU thresholds for evaluation
        class_names: Optional class names for reporting
        
    Returns:
        Dict with mAP, per-class AP, precision, recall, F1
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        per_class_ap = []
        per_class_metrics = {}
        
        for cls_id in range(num_classes):
            # Gather all predictions and GTs for this class
            cls_preds = []  # (score, is_tp_or_fp for each pred)
            total_gt = 0
            
            for img_idx, (pred, gt) in enumerate(zip(all_predictions, all_targets)):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                gt_boxes = gt['boxes']
                gt_labels = gt['labels']
                
                # Handle tensor/numpy conversion
                if isinstance(pred_boxes, torch.Tensor):
                    pred_boxes = pred_boxes.cpu().numpy()
                    pred_scores = pred_scores.cpu().numpy()
                    pred_labels = pred_labels.cpu().numpy()
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                    gt_labels = gt_labels.cpu().numpy()
                
                # Filter by class
                cls_pred_mask = pred_labels == cls_id
                cls_gt_mask = gt_labels == cls_id
                
                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                cls_gt_boxes = gt_boxes[cls_gt_mask]
                
                total_gt += len(cls_gt_boxes)
                
                if len(cls_pred_boxes) == 0:
                    continue
                
                if len(cls_gt_boxes) == 0:
                    for score in cls_pred_scores:
                        cls_preds.append((score, 0))  # FP
                    continue
                
                # Compute IoU matrix
                ious = compute_iou(cls_pred_boxes, cls_gt_boxes)
                
                # Match predictions to GTs
                gt_matched = np.zeros(len(cls_gt_boxes), dtype=bool)
                
                # Sort by score (highest first)
                sort_idx = cls_pred_scores.argsort()[::-1]
                
                for pred_idx in sort_idx:
                    score = cls_pred_scores[pred_idx]
                    
                    if len(cls_gt_boxes) > 0:
                        max_iou_idx = ious[pred_idx].argmax()
                        max_iou = ious[pred_idx, max_iou_idx]
                        
                        if max_iou >= iou_thresh and not gt_matched[max_iou_idx]:
                            cls_preds.append((score, 1))  # TP
                            gt_matched[max_iou_idx] = True
                        else:
                            cls_preds.append((score, 0))  # FP
                    else:
                        cls_preds.append((score, 0))  # FP
            
            # Compute precision-recall curve
            if total_gt == 0:
                per_class_ap.append(0.0)
                per_class_metrics[cls_id] = {
                    'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'num_gt': 0, 'num_pred': len(cls_preds)
                }
                continue
            
            # Sort predictions by score
            cls_preds.sort(key=lambda x: x[0], reverse=True)
            
            tp_cumsum = np.cumsum([p[1] for p in cls_preds])
            fp_cumsum = np.cumsum([1 - p[1] for p in cls_preds])
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / total_gt
            
            ap = compute_ap_all_points(recall, precision)
            per_class_ap.append(ap)
            
            # Final precision/recall at best F1
            f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
            best_f1_idx = f1_scores.argmax() if len(f1_scores) > 0 else 0
            
            cls_name = class_names[cls_id] if class_names else f"class_{cls_id}"
            per_class_metrics[cls_name] = {
                'ap': float(ap),
                'precision': float(precision[best_f1_idx]) if len(precision) > 0 else 0.0,
                'recall': float(recall[best_f1_idx]) if len(recall) > 0 else 0.0,
                'f1': float(f1_scores[best_f1_idx]) if len(f1_scores) > 0 else 0.0,
                'num_gt': int(total_gt),
                'num_pred': len(cls_preds)
            }
        
        mAP = np.mean(per_class_ap) if per_class_ap else 0.0
        
        results[f'mAP@{iou_thresh}'] = float(mAP)
        results[f'per_class@{iou_thresh}'] = per_class_metrics
    
    # COCO-style mAP@0.5:0.95
    if len(iou_thresholds) == 1 and iou_thresholds[0] == 0.5:
        results['mAP@0.5'] = results.get('mAP@0.5', 0.0)
    
    return results


def run_evaluation(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    class_names: List[str] = None,
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5
) -> Dict:
    """
    Run full evaluation on a dataset.
    
    Args:
        model: The detection model
        data_loader: DataLoader for evaluation data
        device: Device to run on
        num_classes: Number of classes
        class_names: Class names
        score_thresh: Score threshold
        nms_thresh: NMS threshold
        
    Returns:
        Evaluation results dict
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            targets = batch['targets']
            
            # Run inference
            results = model.predict(
                rgb, thermal,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh
            )
            
            all_predictions.extend(results)
            all_targets.extend(targets)
    
    # Compute metrics
    metrics = evaluate_detections(
        all_predictions, all_targets,
        num_classes=num_classes,
        iou_thresholds=[0.5, 0.75],
        class_names=class_names
    )
    
    return metrics
