"""
Streamlit App 02: Annotation Viewer
View COCO-format annotations overlaid on images.
"""

import streamlit as st
import sys, os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Annotation Viewer", layout="wide")
st.title("🏷️ Annotation Viewer")

data_dir = st.sidebar.text_input("Data Directory", value="data/sample")
data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)
ann_path = os.path.join(data_dir, "annotations.json")

if not os.path.exists(ann_path):
    if st.sidebar.button("Generate Sample Data"):
        from src.utils import create_sample_data
        os.makedirs(data_dir, exist_ok=True)
        create_sample_data(data_dir, num_samples=30)
        st.rerun()
    st.warning("No annotations found. Generate sample data first.")
    st.stop()

with open(ann_path) as f:
    coco = json.load(f)

cat_map = {c['id']: c['name'] for c in coco['categories']}
img_to_anns = {}
for ann in coco['annotations']:
    img_to_anns.setdefault(ann['image_id'], []).append(ann)

st.sidebar.markdown(f"**{len(coco['images'])} images, {len(coco['annotations'])} annotations**")
st.sidebar.markdown(f"**Categories**: {', '.join(cat_map.values())}")

idx = st.sidebar.slider("Image Index", 0, len(coco['images'])-1, 0)
img_info = coco['images'][idx]
anns = img_to_anns.get(img_info['id'], [])

# Load image
rgb_path = os.path.join(data_dir, "rgb", img_info['file_name'])
rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if rgb_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

# Draw annotations
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.imshow(rgb_img)

colors = plt.cm.Set2(np.linspace(0, 1, len(cat_map)))
for ann in anns:
    x, y, w, h = ann['bbox']
    cls_id = ann['category_id']
    color = colors[cls_id % len(colors)]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y-3, f"{cat_map.get(cls_id, '?')}", fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

ax.set_title(f"{img_info['file_name']} — {len(anns)} objects")
ax.axis('off')
plt.tight_layout()

buf = BytesIO()
fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
st.image(buf, width="stretch")
plt.close(fig)

# Annotation table
st.markdown("### Annotations")
for i, ann in enumerate(anns):
    x, y, w, h = ann['bbox']
    st.markdown(f"**{i+1}.** {cat_map.get(ann['category_id'], '?')} — "
               f"BBox: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}] — Area: {ann['area']}")
