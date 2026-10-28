# Multi-Sensor Object Detection with Feature-Level Fusion

A PyTorch-based RGB–thermal object detection system that combines complementary visual and infrared information through **mid-level feature fusion**. The project is designed for robust detection in difficult conditions such as low light, fog, and nighttime scenes.

## Highlights

- **Dual-stream model:** independent RGB and thermal feature extractors
- **Feature-level fusion:** channel, spatial, and cross-modal attention
- **FPN neck:** multi-scale feature extraction for objects of different sizes
- **FCOS-style head:** anchor-free classification, regression, and centerness prediction
- **Multiple datasets:** support for LLVIP, FLIR ADAS v2, and the unified dataset
- **Interactive analysis:** Streamlit applications for inference, visualization, and evaluation
- **Notebook workflow:** portable training and experimentation through `colab_training.ipynb`

## Architecture

```text
RGB image ───────► RGB backbone ─────┐
                                     ├─► Attention-based fusion ─► FPN ─► FCOS head ─► Detections
Thermal image ───► Thermal backbone ─┘
```

The model preserves modality-specific features before exchanging information between RGB and thermal streams. See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design, data flow, attention modules, losses, and training strategy.

## Repository Structure

```text
Fusion/
├── configs/             # YAML configuration files
├── data/                # Dataset preparation and dataset files
├── notebooks/           # Exploration, training, evaluation, and inference notebooks
├── src/                 # Model, dataset, loss, training, evaluation, and utility modules
├── streamlit_apps/      # Interactive demo and analysis applications
├── weights/             # Model checkpoints
├── outputs/             # Training outputs, logs, and visualizations
├── colab_training.ipynb # Portable end-to-end training notebook
├── train_production.py  # Command-line LLVIP training script
├── analyze_flir.py      # FLIR ADAS dataset inspection utility
├── test_weights.py      # Checkpoint loading and inference smoke test
├── requirements.txt     # Python dependencies
├── ARCHITECTURE.md      # Detailed model documentation
├── PLAN.md              # Development plan
├── PROGRESS.md          # Progress tracking
└── STATUS.md            # Current project status and known issues
```

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Shivam200427/Fusion.git
cd Fusion
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For NVIDIA GPUs, install a PyTorch build compatible with the CUDA version installed on your system before installing the remaining dependencies. The portable notebook can also detect CUDA, DirectML, or CPU execution and adjust its configuration.

## Datasets

The project supports paired RGB and thermal imagery from:

- **LLVIP:** primarily pedestrian detection
- **FLIR ADAS v2:** pedestrian, vehicle, and traffic-related classes
- **Unified dataset:** combines the supported sources into a common COCO-style format

Dataset preparation utilities are located in `data/`. The expected processed dataset layout includes paired RGB and thermal images plus an `annotations.json` file.

If the dataset downloader is available in your checkout, use:

```bash
python data/download_dataset.py --dataset llvip
```

For dataset-specific instructions and the current dataset state, see [STATUS.md](STATUS.md).

## Training

### Notebook workflow

Open `colab_training.ipynb` in Google Colab, Jupyter, or VS Code and run the cells in order. The notebook detects the available device and configures the model for CUDA, DirectML, or CPU execution.

### Command-line LLVIP training

The production script defaults to a small subset for development and CPU-friendly testing:

```bash
python train_production.py
```

Useful options:

```bash
python train_production.py --num-images 1000 --epochs 25
python train_production.py --backbone resnet34 --img-size 416 --batch-size 8
python train_production.py --full
python train_production.py --resume weights/best_trained.pth
```

Checkpoints are written to `outputs/checkpoints/` and the best checkpoint is also saved as `weights/best_trained.pth`.

For configuration-driven training, use the project configuration and training modules:

```bash
python src/train.py --config configs/default.yaml
```

> **Note:** Confirm that your checkpoint, dataset, and configuration use the same number of classes. Some existing checkpoints may represent the earlier LLVIP person-only experiment, while the unified configuration targets multiple classes. See [STATUS.md](STATUS.md) before evaluating a checkpoint.

## Inference and Streamlit Apps

Run the full-pipeline application with:

```bash
streamlit run streamlit_apps/app_12_full_pipeline.py
```

The other applications in `streamlit_apps/` provide additional tools for dataset inspection, visualization, inference, and evaluation.

To validate a trained checkpoint from the command line:

```bash
python test_weights.py
```

## Evaluation

Evaluation utilities and notebooks include support for:

- mAP at IoU 0.5
- COCO-style mAP at IoU 0.5:0.95
- Per-class average precision
- Precision/recall analysis
- Inference speed measurements
- RGB-only, thermal-only, and fused ablations

See `src/evaluate.py` and the notebooks in `notebooks/` for the available workflows.

## Documentation

- [Architecture](ARCHITECTURE.md) — model design and feature-fusion details
- [Project plan](PLAN.md) — planned work and milestones
- [Progress](PROGRESS.md) — implementation progress
- [Project status](STATUS.md) — datasets, checkpoints, known issues, and next steps

## Troubleshooting

- **Out-of-memory errors:** reduce image size or batch size. The fusion module includes spatial reduction for attention on smaller GPUs.
- **Windows DataLoader errors:** use `num_workers=0`.
- **DirectML issues:** consult the DirectML workaround documented in [STATUS.md](STATUS.md).
- **Missing dataset files:** verify that the processed RGB, thermal, and annotation paths match the selected configuration.

## License

No license has been specified for this repository yet. Contact the repository owner before redistributing or using the project commercially.
