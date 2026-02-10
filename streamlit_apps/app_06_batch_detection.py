"""
Streamlit App 06: Batch Detection
Run object detection on multiple image pairs at once.
"""

import streamlit as st
import sys, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Batch Detection", layout="wide")
st.title("📦 Batch Object Detection")
st.markdown("Run inference on multiple RGB + Thermal pairs from the dataset.")

from src.model import build_model
from src.utils import get_device, create_sample_data, load_config, resize_image

device = get_device('auto')
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
cfg = load_config(cfg_path)

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_dataset')
if not os.path.isdir(os.path.join(data_dir, 'rgb')):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
    if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
        create_sample_data(data_dir, num_samples=10)

CLASS_NAMES = cfg['dataset'].get('classes', ['person', 'bike', 'car', 'motor', 'bus', 'truck', 'light', 'sign', 'dog', 'other_vehicle'])

st.sidebar.header("Batch Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.3, 0.05)
nms_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 0.9, 0.5, 0.05)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_trained.pth')

@st.cache_resource
def load_model():
    _cfg = cfg
    if os.path.exists(WEIGHTS_PATH):
        _ckpt = torch.load(WEIGHTS_PATH, map_location='cpu')
        if 'config' in _ckpt:
            _cfg = _ckpt['config']
    model = build_model(_cfg)
    model.to(device)
    model.eval()
    if os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    return model

model = load_model()

rgb_dir = os.path.join(data_dir, 'rgb')
thermal_dir = os.path.join(data_dir, 'thermal')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found. Run `python data/download_dataset.py --dataset sample` first.")
    st.stop()

selected = st.multiselect("Select images for batch detection", img_files, default=img_files[:5])

if st.button("🚀 Run Batch Detection"):
    all_results = []
    progress = st.progress(0)
    cols_per_row = 3

    for i, fname in enumerate(selected):
        rgb_np = np.array(Image.open(os.path.join(rgb_dir, fname)).convert('RGB'))
        thermal_np = np.array(Image.open(os.path.join(thermal_dir, fname)).convert('L'))
        img_size = (320, 320)

        rgb_r = resize_image(rgb_np, img_size)
        th_3ch = np.stack([thermal_np]*3, -1) if thermal_np.ndim == 2 else thermal_np
        th_r = resize_image(th_3ch, img_size)

        rgb_t = torch.from_numpy(rgb_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        th_t = torch.from_numpy(th_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            preds = model.predict(rgb_t.to(device), th_t.to(device),
                                 score_thresh=conf_thresh, nms_thresh=nms_thresh)

        pred = preds[0]
        n_det = len(pred.get('boxes', []))
        all_results.append({'Image': fname, 'Detections': n_det})

        # Draw
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(rgb_r)
        boxes = pred.get('boxes', [])
        scores = pred.get('scores', [])
        labels = pred.get('labels', [])
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            cls_idx = int(labels[j]) if not isinstance(labels[j], int) else labels[j]
            score = float(scores[j])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'cls_{cls_idx}'
            ax.text(x1, y1-3, f'{name} {score:.2f}', fontsize=7, color='white',
                    bbox=dict(facecolor='green', alpha=0.7, pad=1))
        ax.set_title(f'{fname} ({n_det} det)', fontsize=9)
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        all_results[-1]['fig_buf'] = buf
        plt.close(fig)
        progress.progress((i + 1) / len(selected))

    # Show grid
    st.markdown("### Results Grid")
    for row_start in range(0, len(all_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(all_results):
                with col:
                    st.image(all_results[idx]['fig_buf'])

    # Summary table
    st.markdown("### Summary")
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'fig_buf'} for r in all_results])
    st.dataframe(summary_df, width="stretch")
    st.metric("Total Detections", summary_df['Detections'].sum())
