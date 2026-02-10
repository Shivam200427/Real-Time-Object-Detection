"""
Multi-Modal Dataset for RGB + Thermal image pairs with COCO-format annotations.
"""

import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import albumentations as A


class MultiModalDataset(Dataset):
    """
    Dataset that loads aligned RGB + Thermal image pairs with bounding box annotations.
    
    Supports COCO-format annotations and FLIR ADAS / LLVIP style directory structures.
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        rgb_dir: str = "rgb",
        thermal_dir: str = "thermal",
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = False,
        rgb_mean: List[float] = [0.485, 0.456, 0.406],
        rgb_std: List[float] = [0.229, 0.224, 0.225],
        thermal_mean: List[float] = [0.5, 0.5, 0.5],
        thermal_std: List[float] = [0.25, 0.25, 0.25],
    ):
        """
        Args:
            root_dir: Root directory containing rgb/ and thermal/ subdirectories
            annotation_file: Path to COCO-format JSON annotation file
            rgb_dir: Subdirectory name for RGB images
            thermal_dir: Subdirectory name for thermal images
            img_size: Target image size (H, W)
            augment: Whether to apply data augmentation
            rgb_mean: RGB normalization mean
            rgb_std: RGB normalization std
            thermal_mean: Thermal normalization mean
            thermal_std: Thermal normalization std
        """
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, rgb_dir)
        self.thermal_dir = os.path.join(root_dir, thermal_dir)
        self.img_size = img_size
        self.augment = augment
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories)
        
        # Build image_id -> annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Setup augmentations
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    size=(img_size[0], img_size[1]),
                    scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5
                ),
            ], bbox_params=A.BboxParams(
                format='coco', label_fields=['class_labels'],
                min_visibility=0.3
            ),
            additional_targets={'thermal_image': 'image'})
        else:
            self.transform = None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - 'rgb': Tensor (3, H, W) normalized RGB image
                - 'thermal': Tensor (3, H, W) normalized thermal image
                - 'target': Dict with 'boxes' (N, 4) in [x1,y1,x2,y2] and 'labels' (N,)
                - 'image_id': int
                - 'file_name': str
        """
        img_info = self.images[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, file_name)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            # Try alternative extensions
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                alt_path = os.path.splitext(rgb_path)[0] + ext
                rgb_img = cv2.imread(alt_path, cv2.IMREAD_COLOR)
                if rgb_img is not None:
                    break
        if rgb_img is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load thermal image
        thermal_path = os.path.join(self.thermal_dir, file_name)
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_img is None:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                alt_path = os.path.splitext(thermal_path)[0] + ext
                thermal_img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
                if thermal_img is not None:
                    break
        if thermal_img is None:
            raise FileNotFoundError(f"Thermal image not found: {thermal_path}")
        
        # Convert thermal to 3-channel for backbone compatibility
        thermal_img_3ch = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h] in COCO format
            boxes.append(bbox)
            labels.append(ann['category_id'])
        
        # Apply augmentations
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=rgb_img,
                thermal_image=thermal_img_3ch,
                bboxes=boxes,
                class_labels=labels
            )
            rgb_img = transformed['image']
            thermal_img_3ch = transformed['thermal_image']
            boxes = list(transformed['bboxes'])
            labels = list(transformed['class_labels'])
        
        # Resize images
        orig_h, orig_w = rgb_img.shape[:2]
        rgb_img = cv2.resize(rgb_img, (self.img_size[1], self.img_size[0]))
        thermal_img_3ch = cv2.resize(thermal_img_3ch, (self.img_size[1], self.img_size[0]))
        
        # Scale boxes to new image size
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        scaled_boxes = []
        for bbox in boxes:
            x, y, w, h = bbox
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        
        # Normalize images
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img = (rgb_img - np.array(self.rgb_mean)) / np.array(self.rgb_std)
        
        thermal_img_3ch = thermal_img_3ch.astype(np.float32) / 255.0
        thermal_img_3ch = (thermal_img_3ch - np.array(self.thermal_mean)) / np.array(self.thermal_std)
        
        # Convert to tensors (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float()
        thermal_tensor = torch.from_numpy(thermal_img_3ch.transpose(2, 0, 1)).float()
        
        # Convert targets
        if len(scaled_boxes) > 0:
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([img_id])
        }
        
        return {
            'rgb': rgb_tensor,
            'thermal': thermal_tensor,
            'target': target,
            'image_id': img_id,
            'file_name': file_name
        }
    
    def get_raw_images(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw (unnormalized) images for visualization."""
        img_info = self.images[idx]
        file_name = img_info['file_name']
        
        rgb_path = os.path.join(self.rgb_dir, file_name)
        thermal_path = os.path.join(self.thermal_dir, file_name)
        
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is not None:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        return rgb_img, thermal_img


def create_dataloaders(
    root_dir: str,
    annotation_file: str,
    cfg: dict,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from src.utils import collate_fn
    
    # Full dataset
    full_dataset = MultiModalDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        img_size=tuple(cfg.get('dataset', {}).get('img_size', [640, 640])),
        augment=False
    )
    
    # Split dataset
    total = len(full_dataset)
    train_size = int(total * cfg.get('dataset', {}).get('train_split', 0.8))
    val_size = int(total * cfg.get('dataset', {}).get('val_split', 0.1))
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
