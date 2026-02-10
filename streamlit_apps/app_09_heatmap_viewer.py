"""
Streamlit App 09: Detection Heatmap Viewer
Visualize class activation / detection confidence heatmaps on images.
"""

import streamlit as st
import sys, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Heatmap Viewer", layout="wide")
st.title("🔥 Detection Heatmaps")
st.markdown("Visualize where the model focuses for detection via activation heatmaps.")

from src.model import build_model
from src.backbone import DualStreamBackbone
from src.fusion import MultiScaleFeatureFusion
from src.utils import get_device, create_sample_data, load_config, resize_image

device = get_device('auto')
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
cfg = load_config(cfg_path)

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
    create_sample_data(data_dir, num_samples=10)

st.sidebar.header("Settings")
cmap = st.sidebar.selectbox("Colormap", ['jet', 'hot', 'inferno', 'viridis', 'magma', 'plasma'], index=0)
alpha = st.sidebar.slider("Overlay Alpha", 0.1, 0.9, 0.5, 0.05)
level = st.sidebar.selectbox("Feature Level", [0, 1, 2, 3], format_func=lambda x: f"Level {x+1}")

@st.cache_resource
def load_modules():
    backbone = DualStreamBackbone('resnet34', pretrained=True).to(device).eval()
    fusion = MultiScaleFeatureFusion(backbone.out_channels).to(device).eval()
    return backbone, fusion

backbone, fusion = load_modules()

rgb_dir = os.path.join(data_dir, 'rgb')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png','.jpg'))]) if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found.")
    st.stop()

selected = st.selectbox("Select Image", img_files)
rgb_np = np.array(Image.open(os.path.join(data_dir, 'rgb', selected)).convert('RGB'))
thermal_np = np.array(Image.open(os.path.join(data_dir, 'thermal', selected)).convert('L'))

img_size = (320, 320)
rgb_r = resize_image(rgb_np, img_size)
th_3ch = np.stack([thermal_np]*3, -1) if thermal_np.ndim == 2 else thermal_np
th_r = resize_image(th_3ch, img_size)

rgb_t = torch.from_numpy(rgb_r).float().permute(2,0,1).unsqueeze(0) / 255.0
th_t = torch.from_numpy(th_r).float().permute(2,0,1).unsqueeze(0) / 255.0

with torch.no_grad():
    rgb_feats, thermal_feats = backbone(rgb_t.to(device), th_t.to(device))
    fused_feats = fusion(rgb_feats, thermal_feats)

def make_heatmap(feat, img, title):
    """Create heatmap overlay from feature map."""
    heat = feat[0].mean(0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    from PIL import Image as PILImage
    heat_resized = np.array(PILImage.fromarray((heat * 255).astype(np.uint8)).resize(
        (img.shape[1], img.shape[0]), PILImage.BILINEAR)) / 255.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ax.imshow(heat_resized, cmap=cmap, alpha=alpha)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

st.markdown(f"### Heatmaps at Level {level+1}")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(make_heatmap(rgb_feats[level], rgb_r, "RGB Activations"))
with col2:
    st.image(make_heatmap(thermal_feats[level], rgb_r, "Thermal Activations"))
with col3:
    st.image(make_heatmap(fused_feats[level], rgb_r, "Fused Activations"))

# All-levels view
st.markdown("---\n### All Levels — Fused Heatmaps")
cols = st.columns(4)
for i, col in enumerate(cols):
    with col:
        st.image(make_heatmap(fused_feats[i], rgb_r, f"Fused Level {i+1}"))

# Difference heatmap
st.markdown("---\n### Difference: Fused vs RGB (what Thermal adds)")
diff_feat = fused_feats[level][0].mean(0).cpu().numpy() - rgb_feats[level][0].mean(0).cpu().numpy()
diff_feat = (diff_feat - diff_feat.min()) / (diff_feat.max() - diff_feat.min() + 1e-8)
from PIL import Image as PILImage
diff_resized = np.array(PILImage.fromarray((diff_feat * 255).astype(np.uint8)).resize(
    (rgb_r.shape[1], rgb_r.shape[0]), PILImage.BILINEAR)) / 255.0

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(rgb_r)
im = ax.imshow(diff_resized, cmap='RdBu_r', alpha=0.6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f'Thermal Contribution Heatmap (Level {level+1})')
ax.axis('off')
buf = BytesIO()
fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
st.image(buf, width="stretch")
plt.close(fig)
