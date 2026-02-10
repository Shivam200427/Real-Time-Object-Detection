"""
Training pipeline for the Multi-Sensor Detector.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model
from src.losses import DetectionLoss
from src.dataset import MultiModalDataset, create_dataloaders
from src.utils import load_config, get_device, collate_fn


class Trainer:
    """Training loop for the multi-sensor detector."""
    
    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or get_device(cfg.get('inference', {}).get('device', 'auto'))
        
        # Build model
        self.model = build_model(cfg).to(self.device)
        
        # Loss function
        loss_cfg = cfg.get('loss', {})
        self.criterion = DetectionLoss(
            num_classes=cfg.get('model', {}).get('num_classes', 5),
            focal_alpha=loss_cfg.get('focal_alpha', 0.25),
            focal_gamma=loss_cfg.get('focal_gamma', 2.0),
            cls_weight=loss_cfg.get('cls_weight', 1.0),
            reg_weight=loss_cfg.get('reg_weight', 1.0),
            centerness_weight=loss_cfg.get('centerness_weight', 1.0)
        )
        
        # Optimizer
        train_cfg = cfg.get('training', {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get('learning_rate', 1e-3),
            weight_decay=train_cfg.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.epochs = train_cfg.get('epochs', 50)
        scheduler_type = train_cfg.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=train_cfg.get('step_milestones', [30, 40]),
                gamma=train_cfg.get('step_gamma', 0.1)
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5
            )
        
        # Logging
        log_cfg = cfg.get('logging', {})
        self.log_dir = log_cfg.get('log_dir', 'outputs/logs')
        self.checkpoint_dir = log_cfg.get('checkpoint_dir', 'outputs/checkpoints')
        self.log_interval = log_cfg.get('log_interval', 50)
        self.save_interval = log_cfg.get('save_interval', 5)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device)
            thermal = batch['thermal'].to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch['targets']]
            
            # Forward
            predictions = self.model(rgb, thermal)
            
            # Loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_dict['cls_loss'].item():.4f}",
                'reg': f"{loss_dict['reg_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            rgb = batch['rgb'].to(self.device)
            thermal = batch['thermal'].to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch['targets']]
            
            predictions = self.model(rgb, thermal)
            loss_dict = self.criterion(predictions, targets)
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.cfg
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.epochs} — "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining complete! Best val loss: {self.best_loss:.4f}")
        print(f"History saved to {history_path}")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multi-Sensor Detector")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Override annotation file path')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override data paths if provided
    data_root = args.data_root or cfg.get('dataset', {}).get('root', 'data/raw')
    ann_file = args.annotations or os.path.join(data_root, 'annotations.json')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=data_root,
        annotation_file=ann_file,
        cfg=cfg,
        batch_size=cfg.get('training', {}).get('batch_size', 8),
        num_workers=cfg.get('dataset', {}).get('num_workers', 4)
    )
    
    # Create trainer and train
    trainer = Trainer(cfg)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
