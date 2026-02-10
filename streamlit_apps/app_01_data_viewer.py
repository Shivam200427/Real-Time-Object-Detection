"""
Streamlit App 01: Data Viewer
Browse RGB + Thermal image pairs from the dataset.
"""

import streamlit as st
import sys, os
import cv2
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Multi-Sensor Data Viewer", layout="wide")
st.title("📷 Multi-Sensor Data Viewer")
st.markdown("Browse aligned RGB + Thermal image pairs from the dataset.")

# Data directory selection
data_dir = st.sidebar.text_input("Data Directory", value="data/sample")
data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)

# Generate sample data if needed
if st.sidebar.button("Generate Sample Data"):
    from src.utils import create_sample_data
    os.makedirs(data_dir, exist_ok=True)
    create_sample_data(data_dir, num_samples=30)
    st.success("Sample data generated!")
    st.rerun()

if not os.path.exists(data_dir):
    st.warning(f"Data directory not found: {data_dir}")
    st.info("Click 'Generate Sample Data' in the sidebar to create synthetic data.")
    st.stop()

rgb_dir = os.path.join(data_dir, "rgb")
thermal_dir = os.path.join(data_dir, "thermal")

if not os.path.exists(rgb_dir):
    st.error("No 'rgb' subdirectory found.")
    st.stop()

# List images
rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
st.sidebar.markdown(f"**{len(rgb_files)} image pairs found**")

# Image selector
idx = st.sidebar.slider("Image Index", 0, max(0, len(rgb_files)-1), 0)
fname = rgb_files[idx]

# Load images
rgb_img = cv2.imread(os.path.join(rgb_dir, fname))
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if rgb_img is not None else None

thermal_img = cv2.imread(os.path.join(thermal_dir, fname), cv2.IMREAD_GRAYSCALE)

# Display
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("RGB Image")
    if rgb_img is not None:
        st.image(rgb_img, width="stretch")
    else:
        st.error("RGB image not found")

with col2:
    st.subheader("Thermal Image")
    if thermal_img is not None:
        colormap = st.sidebar.selectbox("Thermal Colormap", 
            ["INFERNO", "JET", "HOT", "BONE", "RAINBOW"])
        cmap = getattr(cv2, f"COLORMAP_{colormap}")
        thermal_colored = cv2.applyColorMap(thermal_img, cmap)
        thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
        st.image(thermal_colored, width="stretch")
    else:
        st.error("Thermal image not found")

with col3:
    st.subheader("Overlay")
    if rgb_img is not None and thermal_img is not None:
        alpha = st.sidebar.slider("Overlay Alpha (RGB)", 0.0, 1.0, 0.6)
        thermal_resized = cv2.resize(
            cv2.applyColorMap(thermal_img, cv2.COLORMAP_INFERNO),
            (rgb_img.shape[1], rgb_img.shape[0])
        )
        thermal_resized = cv2.cvtColor(thermal_resized, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_img, alpha, thermal_resized, 1-alpha, 0)
        st.image(overlay, width="stretch")

# Image stats
st.markdown("---")
col_s1, col_s2 = st.columns(2)
with col_s1:
    if rgb_img is not None:
        st.markdown(f"**RGB**: {rgb_img.shape[1]}x{rgb_img.shape[0]}, "
                   f"Mean: {rgb_img.mean():.1f}, Std: {rgb_img.std():.1f}")
with col_s2:
    if thermal_img is not None:
        st.markdown(f"**Thermal**: {thermal_img.shape[1]}x{thermal_img.shape[0]}, "
                   f"Mean: {thermal_img.mean():.1f}, Std: {thermal_img.std():.1f}")
