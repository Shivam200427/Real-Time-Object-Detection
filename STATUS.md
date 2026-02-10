# Multi-Sensor Object Detection — Project Status

> **Last Updated:** June 2025  
> **Author:** Shiva (with AI assistance)  
> **Purpose:** This file documents the complete project state so any AI assistant (Claude Opus 4, etc.) can pick up exactly where we left off.

---

## 1. What This Project Is

A **multi-sensor object detection system** that fuses **RGB + Thermal/IR images** to detect objects in challenging conditions (night, fog, etc.).

- **Architecture:** Dual-stream ResNet backbone → Cross-Modal Attention Fusion → FPN Neck → FCOS Detection Head
- **Dataset:** Unified FLIR ADAS v2 + LLVIP — **5,500 images, 58,709 annotations, 10 classes**
- **10 Classes:** `person`, `bike`, `car`, `motor`, `bus`, `truck`, `light`, `sign`, `dog`, `other_vehicle`
- **Frontend:** 12 Streamlit apps for visualization, detection, evaluation

---

## 2. Current Status — What's Done & What's Pending

### ✅ COMPLETED

| Item | Details |
|------|---------|
| Source code (11 modules) | `src/` — backbone, fusion, neck, head, model, dataset, losses, train, evaluate, inference, utils |
| 6 analysis notebooks | `notebooks/01-06` — data exploration, feature extraction, fusion analysis, training, evaluation, inference |
| 12 Streamlit apps | `streamlit_apps/app_01-12` — all updated for 10 classes, unified dataset |
| LLVIP dataset | `data/LLVIP/` — 15,488 image pairs, person-only |
| FLIR ADAS v2 dataset | `data/FLIR_ADAS_v2/` — vehicles, pedestrians, cyclists |
| Unified dataset | `data/unified_dataset/` — 5,500 images, 58,709 annotations, 10 classes |
| Dataset builder | `data/build_unified_dataset.py` — merges FLIR + LLVIP |
| Portable training notebook | `colab_training.ipynb` — auto-detects CUDA/DirectML/CPU |
| DirectML patch | `torch_directml/__init__.py` patched for Python 3.9 `staticmethod` bug |
| Fusion memory fix | `src/fusion.py` — spatial reduction pooling (16×16 max) in `_attention()` |
| Loss function fix | Training loop uses `criterion(predictions, targets_gpu)` — correct API |
| CPU-trained model (LLVIP only) | `weights/best_trained.pth` — resnet18, 1 class, val_loss=3.07 |
| Config files | `configs/default.yaml` — num_classes=10, backbone=resnet34 |

### ❌ NOT YET DONE (Critical)

| Item | Details |
|------|---------|
| **10-class model training** | Never successfully ran Cell 7 on GPU — no trained 10-class model exists |
| **Streamlit app testing** | Apps reference 10-class weights but model is still 1-class (LLVIP person-only) |
| **Cell 7 (training loop)** | Needs to run on RTX 3050 (8GB VRAM) to produce `weights/best_trained.pth` with 10 classes |

### ⚠️ MINOR / OPTIONAL

| Item | Details |
|------|---------|
| mAP evaluation | Run after training — `notebooks/05_evaluation.ipynb` |
| TensorBoard logging | Not integrated into portable notebook (optional) |
| Model export (ONNX) | Not done — can add post-training |

---

## 3. Device Configurations

The portable notebook (`colab_training.ipynb`) auto-detects the GPU and configures training accordingly:

| GPU | Type | VRAM | Backbone | ImgSize | Batch | Epochs | Est. Time |
|-----|------|------|----------|---------|-------|--------|-----------|
| **RTX 3050** | CUDA | 8 GB | resnet34 | 416×416 | 8 | 30 | ~20 min |
| **RX 6500M** | DirectML | 4 GB | resnet18 | 320×320 | 2 | 30 | ~90 min |
| **CPU** | cpu | - | resnet18 | 256×256 | 2 | 15 | ~4 hrs |

### Detection Logic (Cell 1 of notebook):
```
if torch.cuda.is_available()     → CUDA (NVIDIA)
elif import torch_directml works → DirectML (AMD)
else                             → CPU fallback
```

### Known Device Issues:
- **AMD RX 6500M (4GB):** BatchNorm backward is broken on DirectML → notebook auto-converts to GroupNorm
- **DirectML + Python 3.9:** `torch_directml/__init__.py` needs patch (lines 74,77) — `staticmethod` not callable in default args
- **RTX 3050 (8GB CUDA):** No known issues — standard PyTorch CUDA works perfectly
- **Windows:** `num_workers=0` in DataLoader (multiprocessing issues), set automatically by notebook

---

## 4. How to Run (Zip → Unzip → Train)

### On RTX 3050 laptop (NVIDIA CUDA):

```bash
# 1. Install Python 3.9+ and pip
# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
cd multi_sensor_detection
pip install -r requirements.txt

# 4. Open colab_training.ipynb in VS Code or Jupyter
# 5. Run ALL cells top to bottom (Cell 1 to Cell 11)
# 6. Cell 7 is the training loop — takes ~20 min on RTX 3050
# 7. After training, test with:
streamlit run streamlit_apps/app_12_full_pipeline.py
```

### On AMD laptop (DirectML):

```bash
# Same as above, but install torch-directml instead:
pip install torch torchvision
pip install torch-directml

# If Python 3.9 and import fails, patch torch_directml/__init__.py:
# Line 74: Change default_device() in default arg to None
# Line 77: Same fix — see Section 7 below for exact patch
```

---

## 5. File Inventory

### Source Code (`src/`)
| File | Purpose |
|------|---------|
| `backbone.py` | Dual-stream ResNet feature extractor (resnet18/34/50) |
| `fusion.py` | Cross-Modal Attention fusion (PATCHED — spatial reduction pooling) |
| `neck.py` | Feature Pyramid Network (FPN) |
| `head.py` | FCOS detection head (classification + regression) |
| `model.py` | `MultiSensorDetector` — combines all components; `predict()` at line 99 |
| `losses.py` | `DetectionLoss` — focal loss + GIoU; `forward(predictions, targets)` |
| `dataset.py` | `MultiModalDataset` — COCO-format loader for RGB+thermal pairs |
| `train.py` | Standalone training script |
| `evaluate.py` | mAP evaluation |
| `inference.py` | Inference utilities |
| `utils.py` | Config loading, image resizing, NMS |

### Key Patches Applied
1. **`src/fusion.py` — `_attention()` method:**
   - Added spatial reduction: when H×W > 256, pools key/value to 16×16 before attention
   - Prevents OOM on GPUs with < 6GB VRAM
   - The attention matrix would otherwise be (B, heads, 6400, 6400) = 2.6 GB

2. **`src/dataset.py` — line 79-82:**
   - Fixed for albumentations 2.0: `RandomResizedCrop(size=(h, w))` instead of `(h, w)` positional

3. **`src/losses.py` — `DetectionLoss.forward()`:**
   - Signature: `forward(self, predictions: Dict, targets: List[Dict])`
   - Training loop must call: `criterion(predictions, targets_gpu)` — NOT with keyword args like `cls_scores=...`

### Training Notebook (`colab_training.ipynb`)
| Cell | Purpose | Status |
|------|---------|--------|
| Header | Markdown with GPU comparison table | ✅ Updated |
| Cell 1 | Auto-detect GPU (CUDA/DirectML/CPU) | ✅ Updated |
| Cell 2 | Auto-detect project path | ✅ Updated |
| Cell 3 | Verify unified dataset | ✅ Works |
| Cell 4 | Import modules, auto-configure backbone/batch/imgsize | ✅ Updated |
| Cell 5 | Build model, conditional BN→GN for DirectML | ✅ Updated |
| Cell 6 | DataLoaders with conditional workers/pin_memory | ✅ Updated |
| Cell 7 | Training loop — **THE MAIN CELL TO RUN** | ✅ Updated, never run to completion |
| Cell 8 | Training curves (matplotlib) | ✅ Works |
| Cell 9 | Detection visualization on samples | ✅ Updated |
| Cell 10 | Print training summary | ✅ Updated |
| Cell 11 | Verify saved model file | ✅ Works |

### Streamlit Apps (`streamlit_apps/`)
All 12 apps created. Apps 05, 06, 07, 11, 12 updated for:
- 10-class detection
- Loading weights from `weights/best_trained.pth`
- Unified dataset paths

### Weights (`weights/`)
| File | Size | Contents | Valid? |
|------|------|----------|--------|
| `best_trained.pth` | 384.8 MB | resnet18, **1 class (person only)**, LLVIP-trained | ⚠️ Outdated — needs retraining for 10 classes |
| `notebook_trained.pth` | ~384 MB | Same as above | ⚠️ Outdated |

**After running Cell 7 on RTX 3050:** `best_trained.pth` will be overwritten with resnet34, 10 classes, unified dataset.

### Configs (`configs/`)
| File | Key Settings |
|------|-------------|
| `default.yaml` | `num_classes: 10`, `backbone: resnet34`, `root: data/unified_dataset` |
| `flir_adas.yaml` | FLIR-specific config |
| `notebook_inference.yaml` | Inference-time config |

### Data (`data/`)
| Directory | Contents |
|-----------|----------|
| `unified_dataset/` | 5,500 images (rgb/ + thermal/) + annotations.json — **THIS IS THE TRAINING DATA** |
| `LLVIP/` | Raw LLVIP dataset (15,488 pairs, person-only) |
| `FLIR_ADAS_v2/` | Raw FLIR ADAS v2 dataset |
| `llvip_processed/` | Processed LLVIP in COCO format |
| `build_unified_dataset.py` | Script that built unified_dataset from FLIR + LLVIP |

---

## 6. Class Distribution (Unified Dataset)

```
[0] person              : 20,681
[1] bike                :  2,126
[2] car                 : 23,532
[3] motor               :    363
[4] bus                 :    746
[5] truck               :    259
[6] light               :  4,509
[7] sign                :  6,131
[8] dog                 :      0
[9] other_vehicle       :    362
Total: 58,709 annotations across 5,500 images
```

---

## 7. Known Bugs & Workarounds

### torch-directml Python 3.9 Patch
If using AMD GPU on Python 3.9, the `torch_directml` package crashes on import. Fix:

**File:** `<python_path>/Lib/site-packages/torch_directml/__init__.py`  
**Line 74:** Change `def has_float64_support(device=default_device()):` → `def has_float64_support(device=None):`  
**Line 77:** Change `def gpu_memory(device=default_device()):` → `def gpu_memory(device=None):`  
Then add `if device is None: device = default_device()` as first line in each method body.

### OOM on Small GPUs
The `CrossModalAttention` in `src/fusion.py` creates attention matrices proportional to spatial resolution squared. At stride-4 feature maps (80×80 for 320px input), this is 6400×6400 = ~2.6GB per head.

**Fix already applied:** `_attention()` pools key/value to max 16×16 (256 spatial positions) before computing attention. This reduces memory from GBs to MBs.

### Windows DataLoader
`num_workers > 0` causes multiprocessing errors on Windows. The notebook auto-sets `num_workers=0` when `os.name == 'nt'`.

---

## 8. What To Do Next

### Immediate (on RTX 3050 laptop):
1. Unzip the project folder
2. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
3. Install requirements: `pip install -r requirements.txt`
4. Open `colab_training.ipynb`
5. **Run all cells 1-11** — Cell 7 trains for 30 epochs (~20 min on RTX 3050)
6. After Cell 7 completes, `weights/best_trained.pth` will have a 10-class model
7. Test: `streamlit run streamlit_apps/app_12_full_pipeline.py`

### After Training:
- Run `notebooks/05_evaluation.ipynb` for mAP metrics
- Run other Streamlit apps (01-11) to visualize results
- Optionally export to ONNX for deployment

---

## 9. For AI Assistants (Claude Opus 4, etc.)

If you're an AI picking up this project, here's what you need to know:

1. **The model has NOT been trained on the 10-class unified dataset yet.** The existing `weights/best_trained.pth` is from an old 1-class (person-only) LLVIP training run with resnet18.

2. **The notebook is ready to train.** All 12 cells are correctly configured. The user just needs to run Cell 1 through Cell 11 sequentially on a machine with a GPU.

3. **The target machine is an RTX 3050 (8GB VRAM).** The notebook auto-detects CUDA and configures resnet34, 416×416, batch 8, 30 epochs.

4. **Loss function API:** `DetectionLoss.forward(predictions_dict, targets_list)` — do NOT use keyword args like `cls_scores=`, `bbox_preds=`. Pass the full predictions dict from `model(rgb, thermal)`.

5. **Fusion memory:** `src/fusion.py` has been patched with spatial reduction pooling. If you change image sizes significantly, check that the `max_spatial=256` constant in `_attention()` is appropriate.

6. **All Streamlit apps expect 10-class weights.** They will work correctly once `best_trained.pth` is produced by the training notebook.

7. **Dataset is self-contained.** `data/unified_dataset/` has rgb/, thermal/, and annotations.json — everything needed for training.

---

*This file was generated to provide full project context for handoff between AI assistants and developer sessions.*
