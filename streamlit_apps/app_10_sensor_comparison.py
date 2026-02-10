"""
Streamlit App 10: Sensor Comparison
Side-by-side comparison of sensor modalities and their detection strengths.
"""

import streamlit as st
import sys, os, torch, numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Sensor Comparison", layout="wide")
st.title("📡 RGB vs Thermal Sensor Analysis")
st.markdown("Analyze and compare information captured by each sensor modality.")

from src.utils import create_sample_data, resize_image

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
    create_sample_data(data_dir, num_samples=10)

rgb_dir = os.path.join(data_dir, 'rgb')
img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(rgb_dir) else []

if not img_files:
    st.warning("No images found.")
    st.stop()

selected = st.selectbox("Select Image", img_files)
rgb_np = np.array(Image.open(os.path.join(data_dir, 'rgb', selected)).convert('RGB'))
thermal_np = np.array(Image.open(os.path.join(data_dir, 'thermal', selected)).convert('L'))

st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.multiselect("Analyses", [
    "Side-by-Side View",
    "Histogram Comparison",
    "Edge Detection",
    "Statistical Summary",
    "Frequency Analysis"
], default=["Side-by-Side View", "Histogram Comparison", "Statistical Summary"])

# Side-by-side
if "Side-by-Side View" in analysis_type:
    st.markdown("### Side-by-Side View")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(rgb_np, caption="RGB", width="stretch")
    with col2:
        st.image(thermal_np, caption="Thermal (Grayscale)", width="stretch")
    with col3:
        # Overlay
        th_colored = plt.cm.hot(thermal_np / 255.0)[..., :3]
        th_colored = (th_colored * 255).astype(np.uint8)
        th_resized = np.array(Image.fromarray(th_colored).resize(
            (rgb_np.shape[1], rgb_np.shape[0])))
        overlay = (0.6 * rgb_np + 0.4 * th_resized).astype(np.uint8)
        st.image(overlay, caption="RGB + Thermal Overlay", width="stretch")

# Histograms
if "Histogram Comparison" in analysis_type:
    st.markdown("### Intensity Histograms")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ch, color) in enumerate(zip(['Red', 'Green', 'Blue'], ['red', 'green', 'blue'])):
        axes[i].hist(rgb_np[:,:,i].ravel(), bins=64, color=color, alpha=0.7, density=True)
        axes[i].set_title(f'{ch} Channel')
        axes[i].set_xlabel('Intensity')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(thermal_np.ravel(), bins=64, color='orange', alpha=0.7, density=True)
    ax.set_title('Thermal Intensity Distribution')
    ax.set_xlabel('Intensity')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

# Edge detection
if "Edge Detection" in analysis_type:
    st.markdown("### Edge Detection Comparison")
    from scipy.ndimage import sobel

    rgb_gray = np.mean(rgb_np, axis=2)
    rgb_edges = np.hypot(sobel(rgb_gray, axis=0), sobel(rgb_gray, axis=1))
    th_resized = np.array(Image.fromarray(thermal_np).resize((rgb_np.shape[1], rgb_np.shape[0])))
    th_edges = np.hypot(sobel(th_resized.astype(float), axis=0), sobel(th_resized.astype(float), axis=1))

    col1, col2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_edges, cmap='gray')
    ax.set_title('RGB Edges')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    col1.image(buf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(th_edges, cmap='gray')
    ax.set_title('Thermal Edges')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    col2.image(buf)
    plt.close(fig)

# Stats
if "Statistical Summary" in analysis_type:
    st.markdown("### Statistical Summary")
    stats_data = []
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        ch = rgb_np[:,:,ch_idx].ravel()
        stats_data.append({
            'Channel': ch_name, 'Mean': f'{ch.mean():.1f}', 'Std': f'{ch.std():.1f}',
            'Min': int(ch.min()), 'Max': int(ch.max()), 'Median': f'{np.median(ch):.1f}'
        })
    th_flat = thermal_np.ravel()
    stats_data.append({
        'Channel': 'Thermal', 'Mean': f'{th_flat.mean():.1f}', 'Std': f'{th_flat.std():.1f}',
        'Min': int(th_flat.min()), 'Max': int(th_flat.max()), 'Median': f'{np.median(th_flat):.1f}'
    })
    st.dataframe(pd.DataFrame(stats_data).set_index('Channel'), width="stretch")

# Frequency
if "Frequency Analysis" in analysis_type:
    st.markdown("### Frequency Domain (FFT Magnitude)")
    rgb_gray = np.mean(rgb_np, axis=2)
    th_resized_f = np.array(Image.fromarray(thermal_np).resize((rgb_np.shape[1], rgb_np.shape[0]))).astype(float)

    col1, col2 = st.columns(2)
    for col, img, name in [(col1, rgb_gray, 'RGB'), (col2, th_resized_f, 'Thermal')]:
        fft = np.fft.fftshift(np.fft.fft2(img))
        mag = np.log1p(np.abs(fft))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(mag, cmap='viridis')
        ax.set_title(f'{name} FFT Magnitude')
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        col.image(buf)
        plt.close(fig)
