# Multi-Sensor Object Detection — Project Plan

## 📌 Objective
Build an object detection system that accepts **multi-modal inputs** (RGB + Thermal/IR) from different sensors, performs **feature-level (mid-level) fusion**, and returns detected objects with bounding box coordinates.

---

## 🔬 Fusion Strategy: **Mid-Level Feature Fusion**
> ⚠️ NOT early fusion (raw pixel concat) or late fusion (decision-level merge)

### How it works:
1. **Separate Backbones** — Two independent ResNet-based feature extractors  
   - RGB Branch → extracts spatial/color features  
   - Thermal Branch → extracts heat signature features  
2. **Feature Fusion Module** — Attention-based fusion at intermediate feature map level  
   - Channel Attention (SE-style) to weight modality importance  
   - Spatial Attention to focus on relevant regions  
   - Cross-modal attention to exchange information between modalities  
3. **Detection Head** — FPN-style neck + anchor-free detection head  
   - Multi-scale feature processing  
   - Classification + Bounding Box Regression  

---

## 📊 Dataset: FLIR ADAS (Teledyne FLIR)
- **Source**: https://www.flir.com/oem/adas/adas-dataset-form/
- **Contents**: ~10K aligned RGB-Thermal image pairs with COCO-format annotations
- **Classes**: Person, Car, Bicycle, Dog, and other objects
- **Format**: COCO JSON annotations + aligned image pairs

### Alternative Datasets:
- KAIST Multispectral Pedestrian Dataset
- LLVIP (Low-Light Vision Image Pair) Dataset
- M3FD (Multi-spectral Multi-scene Multi-modal Fusion Dataset)

---

## 🏗️ Architecture Overview

```
RGB Image ──→ [ResNet Backbone] ──→ RGB Features ──┐
                                                     ├──→ [Feature Fusion Module] ──→ [FPN Neck] ──→ [Detection Head] ──→ Predictions
Thermal Image ──→ [ResNet Backbone] ──→ Thermal Features ─┘
                                                     ↑
                                              Channel Attention
                                              Spatial Attention
                                              Cross-Modal Attention
```

---

## 📁 Project Structure

```
multi_sensor_detection/
├── PLAN.md                          # This file
├── ARCHITECTURE.md                  # Detailed architecture doc
├── PROGRESS.md                      # Progress tracking
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
│
├── configs/
│   ├── default.yaml                 # Default hyperparameters
│   └── flir_adas.yaml              # FLIR ADAS specific config
│
├── data/
│   ├── raw/                         # Raw downloaded dataset
│   │   ├── rgb/
│   │   └── thermal/
│   ├── processed/                   # Processed & aligned data
│   ├── annotations/                 # COCO format annotations
│   ├── sample/                      # Small sample for testing
│   └── download_dataset.py          # Dataset download helper
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # Multi-modal dataset class
│   ├── backbone.py                  # Dual-stream feature extractors
│   ├── fusion.py                    # Feature fusion modules
│   ├── neck.py                      # FPN neck
│   ├── head.py                      # Detection head
│   ├── model.py                     # Full model assembly
│   ├── train.py                     # Training pipeline
│   ├── evaluate.py                  # Evaluation & metrics
│   ├── inference.py                 # Inference pipeline
│   ├── losses.py                    # Loss functions
│   ├── anchors.py                   # Anchor generation (if needed)
│   └── utils.py                     # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Explore FLIR ADAS dataset
│   ├── 02_feature_extraction.ipynb  # Extract & visualize features
│   ├── 03_fusion_analysis.ipynb     # Analyze fusion strategies
│   ├── 04_model_training.ipynb      # Train the detector
│   ├── 05_evaluation.ipynb          # Evaluate performance
│   └── 06_inference_demo.ipynb      # Run inference demos
│
├── streamlit_apps/
│   ├── app_01_data_viewer.py            # Browse RGB/Thermal pairs
│   ├── app_02_annotation_viewer.py      # View COCO annotations
│   ├── app_03_feature_extractor.py      # Extract & view features
│   ├── app_04_fusion_visualizer.py      # Visualize feature fusion
│   ├── app_05_single_detection.py       # Detect objects in one pair
│   ├── app_06_batch_detection.py        # Batch inference
│   ├── app_07_model_comparison.py       # Compare fusion strategies
│   ├── app_08_metrics_dashboard.py      # Performance metrics
│   ├── app_09_heatmap_viewer.py         # Detection heatmaps
│   ├── app_10_sensor_comparison.py      # RGB vs Thermal comparison
│   ├── app_11_realtime_simulator.py     # Simulated real-time detection
│   └── app_12_full_pipeline.py          # End-to-end pipeline
│
├── weights/                         # Saved model weights
│   └── .gitkeep
│
└── outputs/
    ├── logs/                        # Training logs
    ├── checkpoints/                 # Model checkpoints
    └── results/                     # Evaluation results
```

---

## 🔧 Tech Stack
| Component | Technology |
|-----------|-----------|
| Framework | PyTorch + torchvision |
| Backbone | ResNet-34 (pretrained on ImageNet) |
| Fusion | Custom Attention-based Mid-Level Fusion |
| Detection | Anchor-free (FCOS-style) head |
| Dataset | FLIR ADAS (COCO format) |
| Metrics | mAP, Precision, Recall, F1 |
| UI | Streamlit (12 apps) |
| Visualization | matplotlib, seaborn, OpenCV |

---

## 📅 Development Phases
1. **Phase 1**: Project setup, dataset download & exploration
2. **Phase 2**: Feature extraction pipeline (RGB + Thermal backbones)
3. **Phase 3**: Feature fusion module implementation
4. **Phase 4**: Detection head + full model assembly
5. **Phase 5**: Training pipeline
6. **Phase 6**: Evaluation & metrics
7. **Phase 7**: Streamlit apps
8. **Phase 8**: Documentation & cleanup

---

## ⬇️ Downloads Required (User Action)
1. **FLIR ADAS Dataset** — Download from https://www.flir.com/oem/adas/adas-dataset-form/
   - Place contents in `data/raw/`
   - OR use the `download_dataset.py` script for LLVIP dataset (auto-download)
2. **Pre-trained weights** — Auto-downloaded by torchvision
