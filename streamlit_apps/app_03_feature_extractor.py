"""
Streamlit App 03: Feature Extractor
Extract and visualize features from both modalities using the backbone.
"""

import streamlit as st
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Feature Extractor", layout="wide")
st.title("🔬 Feature Extractor")
st.markdown("Extract and visualize CNN features from RGB and Thermal images.")

from src.backbone import DualStreamBackbone
from src.dataset import MultiModalDataset
from src.utils import get_device, create_sample_data

device = get_device('auto')
st.sidebar.markdown(f"**Device**: {device}")

@st.cache_resource
def load_backbone():
    backbone = DualStreamBackbone('resnet34', pretrained=True).to(device)
    backbone.eval()
    return backbone

backbone = load_backbone()

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
    if st.sidebar.button("Generate Sample Data"):
        create_sample_data(data_dir, num_samples=10)
        st.rerun()
    st.warning("Generate sample data first.")
    st.stop()

dataset = MultiModalDataset(
    root_dir=data_dir,
    annotation_file=os.path.join(data_dir, 'annotations.json'),
    img_size=(320, 320)
)

idx = st.sidebar.slider("Image Index", 0, len(dataset)-1, 0)
sample = dataset[idx]
rgb_tensor = sample['rgb'].unsqueeze(0).to(device)
thermal_tensor = sample['thermal'].unsqueeze(0).to(device)

# Extract features
with torch.no_grad():
    rgb_feats, thermal_feats = backbone(rgb_tensor, thermal_tensor)

# Settings
level = st.sidebar.selectbox("Feature Level", [1, 2, 3, 4], index=2) - 1
num_channels = st.sidebar.slider("Channels to Show", 4, 32, 8)
cmap_choice = st.sidebar.selectbox("Colormap", ["viridis", "inferno", "magma", "plasma", "hot"])

st.markdown(f"### Level {level+1} Features — Shape: {rgb_feats[level].shape}")

# Display feature maps
tab1, tab2 = st.tabs(["RGB Features", "Thermal Features"])

for tab, feats, name in [(tab1, rgb_feats, "RGB"), (tab2, thermal_feats, "Thermal")]:
    with tab:
        feat_np = feats[level][0].cpu().numpy()
        cols = st.columns(min(num_channels, 8))
        for ch_idx in range(min(num_channels, feat_np.shape[0])):
            with cols[ch_idx % len(cols)]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(feat_np[ch_idx], cmap=cmap_choice)
                ax.set_title(f'Ch {ch_idx}', fontsize=9)
                ax.axis('off')
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                buf.seek(0)
                st.image(buf)
                plt.close(fig)

# Mean activation map
st.markdown("### Mean Activation Maps")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_feats[level][0].mean(0).cpu().numpy(), cmap='viridis')
    ax.set_title(f'RGB Mean Activation (Level {level+1})')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(thermal_feats[level][0].mean(0).cpu().numpy(), cmap='inferno')
    ax.set_title(f'Thermal Mean Activation (Level {level+1})')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    st.image(buf)
    plt.close(fig)
