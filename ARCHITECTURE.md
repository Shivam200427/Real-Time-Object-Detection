# Architecture: Multi-Sensor Feature-Level Fusion Detector

## Overview
This document describes the detailed architecture of the multi-modal object detection system that uses **mid-level (feature-level) fusion** to combine RGB and Thermal image streams.

---

## Why Mid-Level Fusion?

| Fusion Type | What is Fused | Pros | Cons |
|------------|--------------|------|------|
| **Early** | Raw pixels | Simple | Noise amplification, loses modality info |
| **Mid-Level** ✅ | Feature maps | Best of both worlds | More complex |
| **Late** | Predictions | Modality independence | Misses cross-modal correlations |

**Mid-level fusion** extracts meaningful features from each sensor independently, then combines them at the feature representation level. This preserves modality-specific information while enabling cross-modal learning.

---

## Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                      │
│  RGB Image (3, H, W)          Thermal Image (1→3, H, W)            │
└──────┬──────────────────────────────┬────────────────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│  RGB Backbone │              │ Thermal      │
│  (ResNet-34)  │              │ Backbone     │
│              │              │ (ResNet-34)   │
│  Layer1→C1   │              │ Layer1→T1     │
│  Layer2→C2   │              │ Layer2→T2     │
│  Layer3→C3   │              │ Layer3→T3     │
│  Layer4→C4   │              │ Layer4→T4     │
└──┬──┬──┬──┬──┘              └──┬──┬──┬──┬──┘
   │  │  │  │                    │  │  │  │
   ▼  ▼  ▼  ▼                    ▼  ▼  ▼  ▼
┌─────────────────────────────────────────────┐
│         FEATURE FUSION MODULE                │
│                                              │
│  For each scale level (C_i, T_i):           │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  1. Channel Attention (SE Block)     │    │
│  │     - Squeeze: GAP → (B, C, 1, 1)  │    │
│  │     - Excitation: FC→ReLU→FC→Sigmoid│    │
│  │     - Scale: element-wise multiply  │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  2. Spatial Attention                │    │
│  │     - Concat [avg_pool, max_pool]   │    │
│  │     - Conv2d(2, 1, k=7) → Sigmoid   │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  3. Cross-Modal Attention            │    │
│  │     - Q from RGB, K,V from Thermal  │    │
│  │     - Attention(Q, K, V)            │    │
│  │     - + Reverse direction           │    │
│  │     - Weighted combination          │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  Output: Fused features F1, F2, F3, F4      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│     FPN NECK                  │
│                               │
│  F4 ──→ Upsample + Lateral   │
│          ↓                    │
│  F3 ──→ Merge + Conv → P3    │
│          ↓                    │
│  F2 ──→ Merge + Conv → P2    │
│          ↓                    │
│  F1 ──→ Merge + Conv → P1    │
│                               │
│  Output: P1, P2, P3, P4      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     DETECTION HEAD            │
│     (FCOS-style Anchor-Free)  │
│                               │
│  For each P_i:               │
│  ┌────────────────────────┐  │
│  │  Classification Branch  │  │
│  │  Conv→GN→ReLU (×4)    │  │
│  │  → Conv → num_classes  │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Regression Branch      │  │
│  │  Conv→GN→ReLU (×4)    │  │
│  │  → Conv → 4 (ltrb)    │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Centerness Branch      │  │
│  │  → Conv → 1            │  │
│  └────────────────────────┘  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     POST-PROCESSING           │
│                               │
│  - Score thresholding         │
│  - NMS (Non-Max Suppression)  │
│  - Top-K selection            │
│                               │
│  Output: [(class, score,      │
│           x1, y1, x2, y2)]   │
└──────────────────────────────┘
```

---

## Module Details

### 1. Dual-Stream Backbone (backbone.py)
- **Architecture**: Two separate ResNet-34 networks (pretrained on ImageNet)
- **RGB stream**: Takes 3-channel RGB input
- **Thermal stream**: Takes 1-channel thermal input (repeated to 3 channels for pretrained compatibility)
- **Output**: Multi-scale feature maps at 4 levels (1/4, 1/8, 1/16, 1/32 of input resolution)

### 2. Feature Fusion Module (fusion.py)
Three-stage attention mechanism:
- **Channel Attention**: Learns which feature channels are important for each modality
- **Spatial Attention**: Learns which spatial locations across modalities are important
- **Cross-Modal Attention**: Enables information exchange between RGB and Thermal features

### 3. FPN Neck (neck.py)
- Top-down pathway with lateral connections
- Produces multi-scale feature maps with uniform channel dimension (256)
- Enables detection at multiple scales

### 4. Detection Head (head.py)
- FCOS-inspired anchor-free design
- Per-pixel prediction: class, box offsets (left, top, right, bottom), centerness
- Multi-scale predictions from FPN levels

### 5. Loss Functions (losses.py)
- **Classification**: Focal Loss (handles class imbalance)
- **Regression**: GIoU Loss (better than L1 for box regression)
- **Centerness**: Binary Cross-Entropy

---

## Data Flow Summary
```
Input: RGB (B, 3, 640, 640) + Thermal (B, 1, 640, 640)
  ↓
Backbone Features: 4 scales each → [(B,64,160,160), (B,128,80,80), (B,256,40,40), (B,512,20,20)]
  ↓
Fused Features: 4 scales → [(B,64,160,160), (B,128,80,80), (B,256,40,40), (B,512,20,20)]
  ↓
FPN Features: 4 scales → all (B, 256, H_i, W_i)
  ↓
Predictions: [(class_scores, box_offsets, centerness)] per scale
  ↓
Output: List of (class_id, confidence, x1, y1, x2, y2) per image
```

---

## Training Strategy
1. **Pretrain backbones**: Use ImageNet-pretrained ResNet-34
2. **Freeze initial layers**: First 2 residual blocks frozen initially
3. **Warm-up**: Linear warm-up for 1000 iterations
4. **Optimizer**: AdamW with weight decay 1e-4
5. **LR Schedule**: Cosine annealing from 1e-3 to 1e-6
6. **Augmentation**: Random flip, color jitter (RGB only), random crop, mosaic
7. **Batch size**: 8-16 depending on GPU memory
8. **Epochs**: 50-100

---

## Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: COCO-style mAP
- **Per-class AP**: AP for each object class
- **Precision/Recall curves**
- **Inference speed** (FPS)
- **Ablation**: RGB-only vs Thermal-only vs Fused
