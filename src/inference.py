"""
Inference pipeline for Multi-Sensor Detector.
Handles loading models, preprocessing, and running detection.
"""

import os
import sys
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model, MultiSensorDetector
from src.utils import load_config, get_device, load_image, resize_image


class MultiSensorInference:
    """
    Inference wrapper for the multi-sensor detector.
    Handles image loading, preprocessing, model inference, and postprocessing.
    """
    
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5
    ):
        """
        Args:
            config_path: Path to model configuration
            weights_path: Path to model weights (if None, uses random init)
            device: Device string ('cuda', 'cpu', or 'auto')
            score_thresh: Minimum detection score
            nms_thresh: NMS IoU threshold
        """
        self.cfg = load_config(config_path)
        self.device = get_device(device or self.cfg.get('inference', {}).get('device', 'auto'))
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        
        # Image preprocessing params
        ds_cfg = self.cfg.get('dataset', {})
        aug_cfg = self.cfg.get('augmentation', {})
        self.img_size = tuple(ds_cfg.get('img_size', [640, 640]))
        self.rgb_mean = aug_cfg.get('normalize_rgb', {}).get('mean', [0.485, 0.456, 0.406])
        self.rgb_std = aug_cfg.get('normalize_rgb', {}).get('std', [0.229, 0.224, 0.225])
        self.thermal_mean = aug_cfg.get('normalize_thermal', {}).get('mean', [0.5])
        self.thermal_std = aug_cfg.get('normalize_thermal', {}).get('std', [0.25])
        
        # Extend thermal mean/std to 3 channels if needed
        if len(self.thermal_mean) == 1:
            self.thermal_mean = self.thermal_mean * 3
        if len(self.thermal_std) == 1:
            self.thermal_std = self.thermal_std * 3
        
        # Class names
        model_cfg = self.cfg.get('model', {})
        self.class_names = model_cfg.get('class_names', 
            ["person", "car", "bicycle", "dog", "other"])
        self.num_classes = model_cfg.get('num_classes', len(self.class_names))
        
        # Build and load model
        self.model = build_model(self.cfg).to(self.device)
        
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        else:
            print("No weights loaded — using random initialization")
        
        self.model.eval()
    
    def preprocess_rgb(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image to tensor."""
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(self.rgb_mean)) / np.array(self.rgb_std)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def preprocess_thermal(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess thermal image to tensor."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(self.thermal_mean)) / np.array(self.thermal_std)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def detect(
        self,
        rgb_img: np.ndarray,
        thermal_img: np.ndarray
    ) -> List[Dict]:
        """
        Run detection on a pair of RGB + Thermal images.
        
        Args:
            rgb_img: RGB image as numpy array (H, W, 3)
            thermal_img: Thermal image as numpy array (H, W) or (H, W, 3)
            
        Returns:
            List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2] in original image coordinates
                - 'class_id': int
                - 'class_name': str
                - 'score': float
        """
        orig_h, orig_w = rgb_img.shape[:2]
        
        # Preprocess
        rgb_tensor = self.preprocess_rgb(rgb_img)
        thermal_tensor = self.preprocess_thermal(thermal_img)
        
        # Inference
        results = self.model.predict(
            rgb_tensor, thermal_tensor,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            img_size=self.img_size
        )
        
        # Convert to list of dicts & scale back to original size
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            
            scale_x = orig_w / self.img_size[1]
            scale_y = orig_h / self.img_size[0]
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                det = {
                    'bbox': [
                        float(x1 * scale_x),
                        float(y1 * scale_y),
                        float(x2 * scale_x),
                        float(y2 * scale_y)
                    ],
                    'class_id': int(labels[i]),
                    'class_name': self.class_names[int(labels[i])] if int(labels[i]) < len(self.class_names) else f"cls_{labels[i]}",
                    'score': float(scores[i])
                }
                detections.append(det)
        
        return detections
    
    def detect_from_paths(
        self,
        rgb_path: str,
        thermal_path: str
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Run detection from image file paths.
        
        Returns:
            (detections, rgb_image, thermal_image)
        """
        rgb_img = load_image(rgb_path, mode="rgb")
        thermal_img = load_image(thermal_path, mode="thermal")
        
        detections = self.detect(rgb_img, thermal_img)
        
        return detections, rgb_img, thermal_img
    
    def get_feature_maps(
        self,
        rgb_img: np.ndarray,
        thermal_img: np.ndarray
    ) -> Dict:
        """Get intermediate feature maps for visualization."""
        rgb_tensor = self.preprocess_rgb(rgb_img)
        thermal_tensor = self.preprocess_thermal(thermal_img)
        
        return self.model.get_feature_maps(rgb_tensor, thermal_tensor)
    
    def batch_detect(
        self,
        rgb_images: List[np.ndarray],
        thermal_images: List[np.ndarray]
    ) -> List[List[Dict]]:
        """Run detection on a batch of image pairs."""
        all_detections = []
        
        for rgb_img, thermal_img in zip(rgb_images, thermal_images):
            dets = self.detect(rgb_img, thermal_img)
            all_detections.append(dets)
        
        return all_detections
