"""
Utility functions for multi-sensor object detection.
"""

import os
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def load_image(path: str, mode: str = "rgb") -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        mode: 'rgb' for 3-channel, 'thermal' for single-channel
        
    Returns:
        numpy array of the image
    """
    if mode == "rgb":
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "thermal":
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size (H, W)."""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Normalize image with mean and std."""
    img = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img - mean) / std
    return img


def denormalize_image(img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Denormalize an image."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    result = boxes.copy()
    result[..., 2] = boxes[..., 0] + boxes[..., 2]
    result[..., 3] = boxes[..., 1] + boxes[..., 3]
    return result


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    result = boxes.copy()
    result[..., 2] = boxes[..., 2] - boxes[..., 0]
    result[..., 3] = boxes[..., 3] - boxes[..., 1]
    return result


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: (N, 4) array
        box2: (M, 4) array
        
    Returns:
        (N, M) IoU matrix
    """
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        threshold: IoU threshold for suppression
        
    Returns:
        List of kept indices
    """
    if len(boxes) == 0:
        return []
    
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious < threshold
        order = order[1:][mask]
    
    return keep


def visualize_detections(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    detections: List[Dict],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize detection results on both RGB and thermal images.
    
    Args:
        rgb_img: RGB image array (H, W, 3)
        thermal_img: Thermal image array (H, W) or (H, W, 3)
        detections: List of dicts with keys: 'bbox', 'class_id', 'score', 'class_name'
        class_names: List of class names
        figsize: Figure size
        save_path: If provided, save image to this path
        
    Returns:
        matplotlib Figure
    """
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # RGB with detections
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB + Detections")
    axes[0].axis("off")
    
    # Thermal with detections
    if thermal_img.ndim == 2:
        axes[1].imshow(thermal_img, cmap="inferno")
    else:
        axes[1].imshow(thermal_img)
    axes[1].set_title("Thermal + Detections")
    axes[1].axis("off")
    
    # Side-by-side overlay
    if thermal_img.ndim == 2:
        thermal_colored = cv2.applyColorMap(
            (thermal_img * 255 / thermal_img.max()).astype(np.uint8) if thermal_img.max() > 0 else thermal_img.astype(np.uint8),
            cv2.COLORMAP_INFERNO
        )
        thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
    else:
        thermal_colored = thermal_img
    
    overlay = cv2.addWeighted(rgb_img, 0.6, 
                               cv2.resize(thermal_colored, (rgb_img.shape[1], rgb_img.shape[0])), 
                               0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Fused Overlay + Detections")
    axes[2].axis("off")
    
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        class_id = det.get('class_id', 0)
        score = det.get('score', 1.0)
        class_name = det.get('class_name', class_names[class_id] if class_names else f"cls_{class_id}")
        
        color = colors[class_id % len(colors)]
        
        for ax in axes:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                bbox[0], bbox[1] - 5,
                f"{class_name}: {score:.2f}",
                color='white', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
            )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_sample_data(output_dir: str, num_samples: int = 10, img_size: Tuple[int, int] = (640, 640)):
    """
    Create synthetic sample data for testing the pipeline.
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of sample pairs to create
        img_size: Image size (H, W)
    """
    import json
    
    rgb_dir = os.path.join(output_dir, "rgb")
    thermal_dir = os.path.join(output_dir, "thermal")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "dog"},
            {"id": 4, "name": "other"}
        ]
    }
    
    ann_id = 0
    
    for i in range(num_samples):
        # Create synthetic RGB image with colored rectangles
        rgb = np.random.randint(50, 200, (img_size[0], img_size[1], 3), dtype=np.uint8)
        # Create synthetic thermal image with heat patterns
        thermal = np.random.randint(20, 80, (img_size[0], img_size[1]), dtype=np.uint8)
        
        # Add synthetic objects
        num_objects = np.random.randint(1, 6)
        
        annotations["images"].append({
            "id": i,
            "file_name": f"img_{i:04d}.jpg",
            "width": img_size[1],
            "height": img_size[0]
        })
        
        for j in range(num_objects):
            x1 = np.random.randint(0, img_size[1] - 100)
            y1 = np.random.randint(0, img_size[0] - 100)
            w = np.random.randint(40, 150)
            h = np.random.randint(40, 200)
            x2 = min(x1 + w, img_size[1])
            y2 = min(y1 + h, img_size[0])
            cls_id = np.random.randint(0, 5)
            
            # Draw on RGB - colored rectangle
            color = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                     (255, 255, 100), (255, 100, 255)][cls_id]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, -1)
            rgb[y1:y2, x1:x2] = cv2.GaussianBlur(rgb[y1:y2, x1:x2], (15, 15), 5)
            
            # Draw on thermal - bright heat signature
            cv2.rectangle(thermal, (x1, y1), (x2, y2), 200 + np.random.randint(0, 55), -1)
            thermal[y1:y2, x1:x2] = cv2.GaussianBlur(
                thermal[y1:y2, x1:x2], (11, 11), 3
            )
            
            annotations["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": cls_id,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "area": int((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            ann_id += 1
        
        # Add noise and blur for realism
        rgb = cv2.GaussianBlur(rgb, (5, 5), 1)
        thermal = cv2.GaussianBlur(thermal, (7, 7), 2)
        
        cv2.imwrite(os.path.join(rgb_dir, f"img_{i:04d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(thermal_dir, f"img_{i:04d}.jpg"), thermal)
    
    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {num_samples} sample image pairs in {output_dir}")
    print(f"Annotations saved to {ann_path}")
    
    return ann_path


def collate_fn(batch):
    """Custom collate function for dataloader."""
    rgb_imgs = []
    thermal_imgs = []
    targets = []
    
    for sample in batch:
        rgb_imgs.append(sample['rgb'])
        thermal_imgs.append(sample['thermal'])
        targets.append(sample['target'])
    
    rgb_imgs = torch.stack(rgb_imgs, dim=0)
    thermal_imgs = torch.stack(thermal_imgs, dim=0)
    
    return {
        'rgb': rgb_imgs,
        'thermal': thermal_imgs,
        'targets': targets
    }
