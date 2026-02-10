# Multi-Sensor Object Detection with Feature-Level Fusion

A PyTorch-based multi-modal object detection system that fuses RGB and Thermal/IR imagery using **mid-level (feature-level) attention-based fusion** for robust object detection.

## Key Features
- **Dual-stream architecture**: Separate backbones for RGB and Thermal modalities
- **Mid-level feature fusion**: Attention-based fusion (NOT early/late fusion)
- **Anchor-free detection**: FCOS-style detection head
- **Multi-scale detection**: FPN neck for detecting objects at various scales
- **12 Streamlit apps**: Interactive tools for visualization, testing, and analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python data/download_dataset.py --dataset llvip
```
Or manually download FLIR ADAS from https://www.flir.com/oem/adas/adas-dataset-form/

### 3. Train
```bash
python src/train.py --config configs/default.yaml
```

### 4. Run Streamlit Apps
```bash
streamlit run streamlit_apps/app_12_full_pipeline.py
```

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Project Plan
See [PLAN.md](PLAN.md) for the full development plan.

## Progress
See [PROGRESS.md](PROGRESS.md) for current progress tracking.
