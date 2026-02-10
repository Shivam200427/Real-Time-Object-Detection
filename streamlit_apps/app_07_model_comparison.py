"""
Streamlit App 07: Model / Modality Comparison
Compare detection with RGB-only, Thermal-only, and Fused inputs.
"""

import streamlit as st
import sys, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("⚖️ Modality Comparison")
st.markdown("Compare detections: **RGB-only** vs **Thermal-only** vs **RGB+Thermal Fused**.")

from src.model import build_model
from src.utils import get_device, create_sample_data, load_config, resize_image

device = get_device('auto')
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
cfg = load_config(cfg_path)
CLASS_NAMES = cfg['dataset'].get('classes', ['person', 'bike', 'car', 'motor', 'bus', 'truck', 'light', 'sign', 'dog', 'other_vehicle'])

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_dataset')
if not os.path.isdir(os.path.join(data_dir, 'rgb')):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
    if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
        create_sample_data(data_dir, num_samples=10)

conf_thresh = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_trained.pth')

@st.cache_resource
def load_model():
    _cfg = cfg
    if os.path.exists(WEIGHTS_PATH):
        _ckpt = torch.load(WEIGHTS_PATH, map_location='cpu')
        if 'config' in _ckpt:
            _cfg = _ckpt['config']
    model = build_model(_cfg)
    model.to(device).eval()
    if os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    return model

model = load_model()

rgb_dir = os.path.join(data_dir, 'rgb')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found.")
    st.stop()

selected = st.selectbox("Select Image", img_files)

rgb_np = np.array(Image.open(os.path.join(data_dir, 'rgb', selected)).convert('RGB'))
thermal_np = np.array(Image.open(os.path.join(data_dir, 'thermal', selected)).convert('L'))

rgb_r = resize_image(rgb_np, (320, 320))
th_3ch = np.stack([thermal_np]*3, -1) if thermal_np.ndim == 2 else thermal_np
th_r = resize_image(th_3ch, (320, 320))

rgb_t = torch.from_numpy(rgb_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0
th_t = torch.from_numpy(th_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0

def run_detection(rgb_tensor, thermal_tensor, label):
    with torch.no_grad():
        preds = model.predict(rgb_tensor.to(device), thermal_tensor.to(device),
                             score_thresh=conf_thresh)
    return preds[0]

# Three modes: normal fused, RGB-only (zero thermal), Thermal-only (zero RGB)
if st.button("🔍 Compare Modalities"):
    zero_t = torch.zeros_like(th_t)
    zero_r = torch.zeros_like(rgb_t)

    results = {}
    for name, r, t in [
        ("RGB Only", rgb_t, zero_t),
        ("Thermal Only", zero_r, th_t),
        ("Fused (RGB+Thermal)", rgb_t, th_t)
    ]:
        results[name] = run_detection(r, t, name)

    cols = st.columns(3)
    for (name, preds), col in zip(results.items(), cols):
        with col:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(rgb_r)
            boxes = preds.get('boxes', [])
            scores = preds.get('scores', [])
            labels = preds.get('labels', [])
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                cls_idx = int(labels[j]) if not isinstance(labels[j], int) else labels[j]
                score = float(scores[j])
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2,
                                        edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                cname = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'cls_{cls_idx}'
                ax.text(x1, y1-3, f'{cname} {score:.2f}', fontsize=7, color='white',
                        bbox=dict(facecolor='green', alpha=0.7, pad=1))
            ax.set_title(f'{name}\n{len(boxes)} detections', fontsize=10)
            ax.axis('off')
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, width="stretch")
            plt.close(fig)

    # Comparison table
    st.markdown("### Detection Count Comparison")
    comp_data = []
    for name, preds in results.items():
        n = len(preds.get('boxes', []))
        avg_score = float(np.mean(preds['scores'])) if n > 0 and len(preds.get('scores', [])) > 0 else 0.0
        comp_data.append({'Modality': name, 'Detections': n, 'Avg Confidence': f'{avg_score:.3f}'})
    st.dataframe(pd.DataFrame(comp_data).set_index('Modality'), width="stretch")
