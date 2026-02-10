"""
Streamlit App 11: Real-Time Detection Simulator
Simulates a real-time stream by iterating through dataset images with auto-advance.
"""

import streamlit as st
import sys, os, torch, numpy as np, time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Realtime Simulator", layout="wide")
st.title("🎬 Real-Time Detection Simulator")
st.markdown("Simulate streaming detection across sequential image pairs.")

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

conf_thresh = st.sidebar.slider("Confidence", 0.05, 0.95, 0.3, 0.05)
delay = st.sidebar.slider("Frame Delay (s)", 0.1, 3.0, 1.0, 0.1)

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
thermal_dir = os.path.join(data_dir, 'thermal')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png','.jpg'))]) if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found.")
    st.stop()

# Frame navigation
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0

col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
with col_ctrl1:
    if st.button("⏮ First"):
        st.session_state.frame_idx = 0
with col_ctrl2:
    if st.button("◀ Previous"):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
with col_ctrl3:
    if st.button("Next ▶"):
        st.session_state.frame_idx = min(len(img_files)-1, st.session_state.frame_idx + 1)
with col_ctrl4:
    if st.button("⏭ Last"):
        st.session_state.frame_idx = len(img_files) - 1

st.markdown(f"**Frame {st.session_state.frame_idx + 1} / {len(img_files)}**")

fname = img_files[st.session_state.frame_idx]
rgb_np = np.array(Image.open(os.path.join(rgb_dir, fname)).convert('RGB'))
thermal_np = np.array(Image.open(os.path.join(thermal_dir, fname)).convert('L'))

img_size = (320, 320)
rgb_r = resize_image(rgb_np, img_size)
th_3ch = np.stack([thermal_np]*3, -1) if thermal_np.ndim == 2 else thermal_np
th_r = resize_image(th_3ch, img_size)

rgb_t = torch.from_numpy(rgb_r).float().permute(2,0,1).unsqueeze(0) / 255.0
th_t = torch.from_numpy(th_r).float().permute(2,0,1).unsqueeze(0) / 255.0

t_start = time.time()
with torch.no_grad():
    preds = model.predict(rgb_t.to(device), th_t.to(device), score_thresh=conf_thresh)
infer_time = time.time() - t_start
pred = preds[0]

# Draw results
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Input Pair")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb_r)
    axes[0].set_title("RGB")
    axes[0].axis('off')
    axes[1].imshow(th_r[:,:,0] if th_r.ndim == 3 else th_r, cmap='hot')
    axes[1].set_title("Thermal")
    axes[1].axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

with col2:
    st.markdown("#### Detection Output")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_r)
    boxes = pred.get('boxes', [])
    scores = pred.get('scores', [])
    labels = pred.get('labels', [])
    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes[j]
        cls_idx = int(labels[j]) if not isinstance(labels[j], int) else labels[j]
        score = float(scores[j])
        color = plt.cm.tab10(cls_idx % 10)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'cls_{cls_idx}'
        ax.text(x1, y1-3, f'{name} {score:.2f}', fontsize=8, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=1))
    ax.set_title(f'Frame {st.session_state.frame_idx+1}: {len(boxes)} detections')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

# Stats
c1, c2, c3 = st.columns(3)
c1.metric("Detections", len(boxes))
c2.metric("Inference Time", f"{infer_time*1000:.1f} ms")
c3.metric("FPS (est.)", f"{1.0/max(infer_time, 0.001):.1f}")

# Auto-play
st.markdown("---")
if st.button("▶️ Auto-Play All Frames"):
    placeholder = st.empty()
    for i in range(len(img_files)):
        fn = img_files[i]
        r = np.array(Image.open(os.path.join(rgb_dir, fn)).convert('RGB'))
        r = resize_image(r, img_size)
        placeholder.image(r, caption=f"Frame {i+1}/{len(img_files)}: {fn}", width="stretch")
        time.sleep(delay)
    placeholder.success("Playback complete!")
