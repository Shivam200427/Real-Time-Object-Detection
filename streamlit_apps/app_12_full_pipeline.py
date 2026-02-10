"""
Streamlit App 12: Full Pipeline Demo
End-to-end pipeline — load images, extract features, fuse, detect, evaluate.
"""

import streamlit as st
import sys, os, torch, numpy as np, time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Full Pipeline", layout="wide")
st.title("🚀 Full Multi-Sensor Detection Pipeline")
st.markdown("""
Complete end-to-end demo:  
**Input (RGB + Thermal) → Dual Backbone → Feature Fusion → FPN → Detection Head → Results**
""")

from src.model import build_model
from src.backbone import DualStreamBackbone
from src.fusion import MultiScaleFeatureFusion
from src.neck import FPNNeck
from src.utils import get_device, create_sample_data, load_config, resize_image

device = get_device('auto')
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
cfg = load_config(cfg_path)

# Match config to the best-trained model weights
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_trained.pth')
if os.path.exists(WEIGHTS_PATH):
    _ckpt = torch.load(WEIGHTS_PATH, map_location='cpu')
    if 'config' in _ckpt:
        cfg = _ckpt['config']
CLASS_NAMES = cfg.get('model', {}).get('class_names',
    cfg['dataset'].get('classes', ['person', 'bike', 'car', 'motor', 'bus', 'truck', 'light', 'sign', 'dog', 'other_vehicle']))

# Use unified dataset (FLIR ADAS + LLVIP) for meaningful detections
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_dataset')
if not os.path.isdir(os.path.join(data_dir, 'rgb')):
    # Fallback to sample data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
    if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
        create_sample_data(data_dir, num_samples=10)

conf_thresh = st.sidebar.slider("Confidence", 0.05, 0.95, 0.15, 0.05)
show_intermediates = st.sidebar.checkbox("Show Intermediate Features", True)

@st.cache_resource
def load_all():
    model = build_model(cfg)
    # Load trained weights if available
    if os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[APP12] Loaded weights from {WEIGHTS_PATH} (epoch {ckpt.get('epoch','?')}, val_loss {ckpt.get('val_loss','?')})")
        else:
            model.load_state_dict(ckpt)
            print(f"[APP12] Loaded raw state_dict from {WEIGHTS_PATH}")
    else:
        print(f"[APP12] WARNING: No weights found at {WEIGHTS_PATH} — using random weights!")
    model.to(device).eval()
    backbone = model.backbone
    fusion = model.fusion
    neck = model.neck
    return model, backbone, fusion, neck

model, backbone, fusion_module, neck = load_all()

rgb_dir = os.path.join(data_dir, 'rgb')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png','.jpg'))])[:50] if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found.")
    st.stop()

# Source
source = st.sidebar.radio("Source", ["Sample Data", "Upload"])

if source == "Upload":
    c1, c2 = st.columns(2)
    rgb_file = c1.file_uploader("RGB", type=['png','jpg','jpeg'])
    thermal_file = c2.file_uploader("Thermal", type=['png','jpg','jpeg'])
    if not (rgb_file and thermal_file):
        st.info("Upload both images to proceed.")
        st.stop()
    rgb_np = np.array(Image.open(rgb_file).convert('RGB'))
    thermal_np = np.array(Image.open(thermal_file).convert('L'))
else:
    selected = st.selectbox("Select Image", img_files)
    rgb_np = np.array(Image.open(os.path.join(data_dir, 'rgb', selected)).convert('RGB'))
    thermal_np = np.array(Image.open(os.path.join(data_dir, 'thermal', selected)).convert('L'))

# Use the same image size the model was trained with
_trained_img_size = cfg.get('dataset', {}).get('img_size', [256, 256])

if st.button("🚀 Run Full Pipeline"):
    img_size = tuple(_trained_img_size)
    rgb_r = resize_image(rgb_np, img_size)
    th_3ch = np.stack([thermal_np]*3, -1) if thermal_np.ndim == 2 else thermal_np
    th_r = resize_image(th_3ch, img_size)

    rgb_t = torch.from_numpy(rgb_r).float().permute(2,0,1).unsqueeze(0) / 255.0
    th_t = torch.from_numpy(th_r).float().permute(2,0,1).unsqueeze(0) / 255.0

    # ───────────── Stage 1: Input ─────────────
    st.markdown("---\n### Stage 1: Input Images")
    c1, c2 = st.columns(2)
    c1.image(rgb_r, caption="RGB Input", width="stretch")
    c2.image(th_r[:,:,0] if th_r.ndim == 3 else th_r, caption="Thermal Input",
             width="stretch", clamp=True)

    timings = {}

    # ───────────── Stage 2: Feature Extraction ─────────────
    t0 = time.time()
    with torch.no_grad():
        rgb_feats, thermal_feats = backbone(rgb_t.to(device), th_t.to(device))
    timings['Backbone'] = time.time() - t0

    if show_intermediates:
        st.markdown("---\n### Stage 2: Dual Backbone Features")
        for lvl in range(min(4, len(rgb_feats))):
            cols = st.columns(2)
            for col, feat, label in [(cols[0], rgb_feats[lvl], 'RGB'), (cols[1], thermal_feats[lvl], 'Thermal')]:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(feat[0].mean(0).cpu().numpy(), cmap='viridis' if label=='RGB' else 'inferno')
                ax.set_title(f'{label} L{lvl+1} ({feat.shape[1]}ch, {feat.shape[2]}×{feat.shape[3]})', fontsize=9)
                ax.axis('off')
                buf = BytesIO(); fig.savefig(buf, format='png', dpi=80, bbox_inches='tight'); buf.seek(0)
                col.image(buf)
                plt.close(fig)

    # ───────────── Stage 3: Feature Fusion ─────────────
    t0 = time.time()
    with torch.no_grad():
        fused_feats = fusion_module(rgb_feats, thermal_feats)
    timings['Fusion'] = time.time() - t0

    if show_intermediates:
        st.markdown("---\n### Stage 3: Mid-Level Feature Fusion (Attention-Based)")
        cols = st.columns(4)
        for i, col in enumerate(cols):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(fused_feats[i][0].mean(0).cpu().numpy(), cmap='magma')
            ax.set_title(f'Fused L{i+1}', fontsize=9)
            ax.axis('off')
            buf = BytesIO(); fig.savefig(buf, format='png', dpi=80, bbox_inches='tight'); buf.seek(0)
            col.image(buf)
            plt.close(fig)

    # ───────────── Stage 4: FPN Neck ─────────────
    t0 = time.time()
    with torch.no_grad():
        fpn_feats = neck(fused_feats)
    timings['FPN'] = time.time() - t0

    if show_intermediates:
        st.markdown("---\n### Stage 4: FPN Feature Pyramid")
        cols = st.columns(len(fpn_feats))
        for i, col in enumerate(cols):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(fpn_feats[i][0].mean(0).cpu().numpy(), cmap='plasma')
            ax.set_title(f'FPN P{i+3}', fontsize=9)
            ax.axis('off')
            buf = BytesIO(); fig.savefig(buf, format='png', dpi=80, bbox_inches='tight'); buf.seek(0)
            col.image(buf)
            plt.close(fig)

    # ───────────── Stage 5: Detection ─────────────
    t0 = time.time()
    with torch.no_grad():
        preds = model.predict(rgb_t.to(device), th_t.to(device), score_thresh=conf_thresh, img_size=img_size)
    timings['Detection'] = time.time() - t0
    pred = preds[0]
    print(f"[APP12] Detection: {len(pred.get('boxes',[]))} boxes, thresh={conf_thresh}, img_size={img_size}")
    if len(pred.get('scores',[])) > 0:
        print(f"[APP12] Top scores: {[f'{float(s):.3f}' for s in pred['scores'][:5]]}")

    st.markdown("---\n### Stage 5: Detection Results")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_r)
    boxes = pred.get('boxes', [])
    scores = pred.get('scores', [])
    labels = pred.get('labels', [])
    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes[j]
        cls_idx = int(labels[j]) if not isinstance(labels[j], int) else labels[j]
        score = float(scores[j])
        color = plt.cm.Set1(cls_idx % 9)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'cls_{cls_idx}'
        ax.text(x1, y1-3, f'{name} {score:.2f}', fontsize=9, color='white',
                bbox=dict(facecolor=color, alpha=0.85, pad=1.5))
    ax.set_title(f'Final Detections: {len(boxes)} objects')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

    # Results table
    if len(boxes) > 0:
        det_df = pd.DataFrame({
            'Class': [CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else f'cls_{int(l)}' for l in labels],
            'Confidence': [f'{float(s):.3f}' for s in scores],
            'X1': [f'{b[0]:.1f}' for b in boxes],
            'Y1': [f'{b[1]:.1f}' for b in boxes],
            'X2': [f'{b[2]:.1f}' for b in boxes],
            'Y2': [f'{b[3]:.1f}' for b in boxes],
        })
        st.dataframe(det_df, width="stretch")

    # Pipeline timing
    st.markdown("---\n### Pipeline Timing")
    total = sum(timings.values())
    timing_df = pd.DataFrame([
        {'Stage': k, 'Time (ms)': f'{v*1000:.1f}', 'Fraction': f'{v/total*100:.1f}%'}
        for k, v in timings.items()
    ])
    timing_df.loc[len(timing_df)] = ['**TOTAL**', f'{total*1000:.1f}', '100%']
    st.dataframe(timing_df.set_index('Stage'), width="stretch")

    fig, ax = plt.subplots(figsize=(8, 3))
    stages = list(timings.keys())
    times_ms = [v*1000 for v in timings.values()]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax.barh(stages, times_ms, color=colors[:len(stages)])
    ax.set_xlabel('Time (ms)')
    ax.set_title('Pipeline Stage Latency')
    for i, v in enumerate(times_ms):
        ax.text(v + 0.5, i, f'{v:.1f}ms', va='center', fontsize=9)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

    st.success(f"Pipeline complete — {len(boxes)} objects detected in {total*1000:.1f}ms")
