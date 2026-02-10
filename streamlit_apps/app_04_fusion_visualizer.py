"""
Streamlit App 04: Fusion Visualizer
Visualize how features from RGB and Thermal are fused.
"""

import streamlit as st
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Fusion Visualizer", layout="wide")
st.title("🔗 Feature Fusion Visualizer")
st.markdown("See how mid-level feature fusion combines RGB and Thermal information.")

from src.backbone import DualStreamBackbone
from src.fusion import MultiScaleFeatureFusion
from src.dataset import MultiModalDataset
from src.utils import get_device, create_sample_data

device = get_device('auto')

@st.cache_resource
def load_modules():
    backbone = DualStreamBackbone('resnet34', pretrained=True).to(device)
    backbone.eval()
    channels = backbone.out_channels
    fusion = MultiScaleFeatureFusion(channels).to(device)
    fusion.eval()
    return backbone, fusion

backbone, fusion = load_modules()

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
    create_sample_data(data_dir, num_samples=10)

dataset = MultiModalDataset(
    root_dir=data_dir,
    annotation_file=os.path.join(data_dir, 'annotations.json'),
    img_size=(320, 320)
)

idx = st.sidebar.slider("Image Index", 0, len(dataset)-1, 0)
sample = dataset[idx]

with torch.no_grad():
    rgb_feats, thermal_feats = backbone(
        sample['rgb'].unsqueeze(0).to(device),
        sample['thermal'].unsqueeze(0).to(device)
    )
    fused_feats = fusion(rgb_feats, thermal_feats)

# Visualize all levels
st.markdown("### Multi-Scale Feature Comparison")
st.markdown("**Left**: RGB | **Center**: Thermal | **Right**: Fused")

for level in range(4):
    st.markdown(f"---\n#### Level {level+1} — {rgb_feats[level].shape[1]} channels, "
               f"{rgb_feats[level].shape[2]}×{rgb_feats[level].shape[3]} spatial")
    
    col1, col2, col3 = st.columns(3)
    
    for col, feat, title, cmap in [
        (col1, rgb_feats[level], "RGB", "viridis"),
        (col2, thermal_feats[level], "Thermal", "inferno"),
        (col3, fused_feats[level], "Fused", "magma")
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(feat[0].mean(0).cpu().numpy(), cmap=cmap)
            ax.set_title(f'{title} (Mean Activation)', fontsize=10)
            ax.axis('off')
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            plt.close(fig)

# Fusion weights
st.markdown("---\n### Learned Fusion Weights")
weights_data = []
for i, block in enumerate(fusion.fusion_blocks):
    rgb_w = torch.sigmoid(block.rgb_weight).item()
    thermal_w = torch.sigmoid(block.thermal_weight).item()
    weights_data.append({'Level': i+1, 'RGB': rgb_w, 'Thermal': thermal_w})

import pandas as pd
df = pd.DataFrame(weights_data)
st.dataframe(df.set_index('Level'), width="stretch")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(4)
ax.bar(x - 0.2, df['RGB'], 0.35, label='RGB', color='coral')
ax.bar(x + 0.2, df['Thermal'], 0.35, label='Thermal', color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels([f'Level {i+1}' for i in range(4)])
ax.set_ylabel('Weight (sigmoid)')
ax.set_title('Fusion Weights: RGB vs Thermal Importance')
ax.legend()
ax.set_ylim(0, 1)
buf = BytesIO()
fig.savefig(buf, format='png', dpi=100)
buf.seek(0)
st.image(buf)
plt.close(fig)
