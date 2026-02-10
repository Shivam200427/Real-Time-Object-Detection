"""
Production Training Script for Multi-Sensor Detector on LLVIP Dataset.

Trains on real LLVIP pedestrian detection data (RGB + Thermal fusion).
Optimized for CPU training with a manageable subset.

Usage:
    python train_production.py
    python train_production.py --num-images 1000 --epochs 25
    python train_production.py --full  # Use ALL 15488 images (very slow on CPU)
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import build_model
from src.losses import DetectionLoss
from src.dataset import MultiModalDataset
from src.utils import load_config, get_device, collate_fn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloaders(dataset, train_ratio=0.85, batch_size=2, num_workers=0):
    """Split dataset into train/val and create loaders."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    indices = list(range(n))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader


def train():
    parser = argparse.ArgumentParser(description="Production Training for Multi-Sensor Detector")
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to use for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='Backbone model (resnet18/resnet34, default: resnet18)')
    parser.add_argument('--full', action='store_true',
                        help='Use ALL images (very slow on CPU)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    set_seed(42)
    device = get_device('auto')
    print(f"\n{'='*60}")
    print(f"  Multi-Sensor Detector — Production Training")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Backbone:    {args.backbone}")
    print(f"  Images:      {'ALL' if args.full else args.num_images}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Image size:  {args.img_size}x{args.img_size}")
    print(f"  LR:          {args.lr}")
    print(f"{'='*60}\n")

    # ── Config ──
    cfg = load_config('configs/default.yaml')
    cfg['model']['num_classes'] = 1
    cfg['model']['class_names'] = ['person']
    cfg['model']['backbone'] = args.backbone
    cfg['dataset']['img_size'] = [args.img_size, args.img_size]

    # ── Dataset ──
    data_dir = 'data/llvip_processed'
    ann_file = os.path.join(data_dir, 'annotations.json')

    if not os.path.exists(ann_file):
        print("ERROR: LLVIP dataset not found at data/llvip_processed/")
        print("Run 'python data/finish_setup.py' first.")
        sys.exit(1)

    full_dataset = MultiModalDataset(
        root_dir=data_dir,
        annotation_file=ann_file,
        img_size=(args.img_size, args.img_size),
        augment=True
    )

    # Subset if not --full
    if not args.full and args.num_images < len(full_dataset):
        indices = random.sample(range(len(full_dataset)), args.num_images)
        full_dataset = Subset(full_dataset, indices)
        # Wrap subset so it looks like a dataset
        print(f"Using {args.num_images}/{15488} images")
    else:
        print(f"Using ALL {len(full_dataset)} images")

    train_loader, val_loader = create_dataloaders(
        full_dataset,
        train_ratio=0.85,
        batch_size=args.batch_size,
        num_workers=0
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val\n")

    # ── Model ──
    model = build_model(cfg).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Loss + Optimizer ──
    criterion = DetectionLoss(
        num_classes=1,
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_weight=1.0,
        reg_weight=1.0,
        centerness_weight=1.0
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Warmup
    warmup_epochs = min(3, args.epochs // 5)

    # ── Training Loop ──
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('weights', exist_ok=True)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # ── Train ──
        model.train()
        epoch_loss = 0
        epoch_cls = 0
        epoch_reg = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                    ncols=100, leave=True)
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

            predictions = model(rgb, thermal)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls += loss_dict['cls_loss'].item()
            epoch_reg += loss_dict['reg_loss'].item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'cls': f"{loss_dict['cls_loss'].item():.3f}",
                'reg': f"{loss_dict['reg_loss'].item():.3f}",
            })

        avg_train = epoch_loss / max(n_batches, 1)
        avg_cls = epoch_cls / max(n_batches, 1)
        avg_reg = epoch_reg / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Validate ──
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                              ncols=100, leave=False):
                rgb = batch['rgb'].to(device)
                thermal = batch['thermal'].to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

                predictions = model(rgb, thermal)
                loss_dict = criterion(predictions, targets)
                val_loss += loss_dict['total_loss'].item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        val_losses.append(avg_val)

        # Update scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - total_start

        print(f"  → Train: {avg_train:.4f} (cls={avg_cls:.4f}, reg={avg_reg:.4f}) | "
              f"Val: {avg_val:.4f} | LR: {lr:.6f} | "
              f"Time: {elapsed/60:.1f}min")

        # ── Save ──
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'config': cfg
            }
            torch.save(checkpoint, 'outputs/checkpoints/best_model.pth')
            # Also save to weights/ for apps
            torch.save(checkpoint, 'weights/best_trained.pth')
            print(f"  ★ New best model saved! (val_loss={avg_val:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'config': cfg
            }, f'outputs/checkpoints/checkpoint_epoch_{epoch+1}.pth')

    # ── Done ──
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: weights/best_trained.pth")
    print(f"{'='*60}")

    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args)
    }
    with open('outputs/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()
