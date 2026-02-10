"""
Streamlit App 05: Single-Pair Detection
Upload or select an RGB + Thermal pair and detect objects.
"""

import streamlit as st
import sys, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Single Detection", layout="wide")
st.title("🎯 Single-Pair Object Detection")
st.markdown("Detect objects using a fused RGB + Thermal image pair.")

from src.model import build_model
from src.utils import get_device, create_sample_data, load_config

device = get_device('auto')

cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
cfg = load_config(cfg_path)

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_dataset')
if not os.path.isdir(os.path.join(data_dir, 'rgb')):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
    if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
        create_sample_data(data_dir, num_samples=10)

# Sidebar
st.sidebar.header("Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.3, 0.05)
nms_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 0.9, 0.5, 0.05)
max_dets = st.sidebar.slider("Max Detections", 10, 500, 100, 10)

source = st.sidebar.radio("Image Source", ["Sample Data", "Upload"])

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

CLASS_NAMES = cfg['dataset'].get('classes', ['person', 'bike', 'car', 'motor', 'bus', 'truck', 'light', 'sign', 'dog', 'other_vehicle'])
COLORS = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))

def detect(rgb_img, thermal_img):
    from src.utils import resize_image, normalize_image
    rgb_np = np.array(rgb_img.convert('RGB'))
    thermal_np = np.array(thermal_img.convert('L'))
    img_size = (320, 320)

    rgb_resized = resize_image(rgb_np, img_size)
    thermal_resized = resize_image(thermal_np if thermal_np.ndim == 3 else np.stack([thermal_np]*3, -1), img_size)

    rgb_t = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0
    thermal_t = torch.from_numpy(thermal_resized).float().permute(2, 0, 1) / 255.0

    with torch.no_grad():
        predictions = model.predict(
            rgb_t.unsqueeze(0).to(device),
            thermal_t.unsqueeze(0).to(device),
            score_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            max_detections=max_dets
        )
    return predictions[0], rgb_resized

def show_results(preds, rgb_np):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(rgb_np)
    boxes = preds.get('boxes', [])
    scores = preds.get('scores', [])
    labels = preds.get('labels', [])

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_idx = labels[i] if isinstance(labels[i], int) else int(labels[i])
        color = COLORS[cls_idx % len(COLORS)]
        score = scores[i] if isinstance(scores[i], float) else float(scores[i])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'cls_{cls_idx}'
        ax.text(x1, y1 - 4, f'{name} {score:.2f}', fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    ax.set_title(f'Detections: {len(boxes)} objects')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

if source == "Upload":
    col1, col2 = st.columns(2)
    with col1:
        rgb_file = st.file_uploader("Upload RGB Image", type=['png', 'jpg', 'jpeg'])
    with col2:
        thermal_file = st.file_uploader("Upload Thermal Image", type=['png', 'jpg', 'jpeg'])

    if rgb_file and thermal_file:
        rgb_img = Image.open(rgb_file)
        thermal_img = Image.open(thermal_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_img, caption="RGB", width="stretch")
        with col2:
            st.image(thermal_img, caption="Thermal", width="stretch")
        
        if st.button("🔍 Detect Objects"):
            with st.spinner("Running detection..."):
                preds, rgb_np = detect(rgb_img, thermal_img)
                st.markdown("### Detection Results")
                show_results(preds, rgb_np)
else:
    rgb_dir = os.path.join(data_dir, 'rgb')
    img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(rgb_dir) else []
    
    if img_files:
        selected = st.selectbox("Select Image", img_files)
        rgb_img = Image.open(os.path.join(data_dir, 'rgb', selected))
        thermal_img = Image.open(os.path.join(data_dir, 'thermal', selected))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_img, caption="RGB", width="stretch")
        with col2:
            st.image(thermal_img, caption="Thermal", width="stretch")
        
        if st.button("🔍 Detect Objects"):
            with st.spinner("Running detection..."):
                preds, rgb_np = detect(rgb_img, thermal_img)
                st.markdown("### Detection Results")
                show_results(preds, rgb_np)
    else:
        st.warning("No sample images found. Run `python data/download_dataset.py --dataset sample` first.")
