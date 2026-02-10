"""
Streamlit App 08: Metrics Dashboard
Display evaluation metrics: mAP, precision, recall, F1, per-class AP.
"""

import streamlit as st
import sys, os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Metrics Dashboard", layout="wide")
st.title("📊 Model Performance Dashboard")
st.markdown("Visualize detection evaluation metrics.")

CLASS_NAMES = ['person']

# Try to load real evaluation results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'results')
log_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')

# Generate demo metrics if no real ones exist
def generate_demo_metrics():
    np.random.seed(42)
    per_class_ap = {c: round(np.random.uniform(0.3, 0.8), 3) for c in CLASS_NAMES}
    return {
        'mAP@0.5': round(np.mean(list(per_class_ap.values())), 3),
        'mAP@0.75': round(np.mean(list(per_class_ap.values())) * 0.7, 3),
        'precision': round(np.random.uniform(0.5, 0.85), 3),
        'recall': round(np.random.uniform(0.4, 0.8), 3),
        'f1': 0.0,
        'per_class_ap': per_class_ap,
        'training_loss': [2.5 - 0.3*i + np.random.normal(0, 0.05) for i in range(20)],
        'val_loss': [2.8 - 0.25*i + np.random.normal(0, 0.08) for i in range(20)],
    }

metrics = generate_demo_metrics()
metrics['f1'] = round(2 * metrics['precision'] * metrics['recall'] /
                     (metrics['precision'] + metrics['recall'] + 1e-8), 3)

# Also check for any saved results
result_files = glob.glob(os.path.join(results_dir, '*.json'))
if result_files:
    st.sidebar.info(f"Found {len(result_files)} saved result file(s)")
    selected_file = st.sidebar.selectbox("Load saved results", ['Demo'] + result_files)
    if selected_file != 'Demo':
        with open(selected_file) as f:
            metrics = json.load(f)

# Top-level KPIs
st.markdown("### Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("mAP@0.5", f"{metrics['mAP@0.5']:.1%}")
c2.metric("mAP@0.75", f"{metrics['mAP@0.75']:.1%}")
c3.metric("Precision", f"{metrics['precision']:.1%}")
c4.metric("Recall", f"{metrics['recall']:.1%}")

st.markdown("---")

# Per-class AP
st.markdown("### Per-Class Average Precision")
ap_df = pd.DataFrame([
    {'Class': k, 'AP@0.5': v} for k, v in metrics['per_class_ap'].items()
]).sort_values('AP@0.5', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ap_df)))
ax.barh(ap_df['Class'], ap_df['AP@0.5'], color=colors)
ax.set_xlabel('AP@0.5')
ax.set_title('Per-Class Average Precision')
ax.set_xlim(0, 1)
for i, v in enumerate(ap_df['AP@0.5']):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
buf = BytesIO()
fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
st.image(buf, width="stretch")
plt.close(fig)

# Training curves
if 'training_loss' in metrics:
    st.markdown("### Training Curves")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs = range(1, len(metrics['training_loss'])+1)
        ax.plot(epochs, metrics['training_loss'], 'b-', label='Train Loss')
        if 'val_loss' in metrics:
            ax.plot(epochs, metrics['val_loss'], 'r--', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, width="stretch")
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        p = metrics['precision']
        r = metrics['recall']
        f1 = metrics['f1']
        bars = ax.bar(['Precision', 'Recall', 'F1'], [p, r, f1],
                      color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylim(0, 1)
        ax.set_title('Precision / Recall / F1')
        for bar, val in zip(bars, [p, r, f1]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=11)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, width="stretch")
        plt.close(fig)

# Confusion-style matrix (simplified)
st.markdown("### Detection Summary")
summary_df = pd.DataFrame({
    'Metric': ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1 Score'],
    'Value': [metrics['mAP@0.5'], metrics['mAP@0.75'],
              metrics['precision'], metrics['recall'], metrics['f1']]
})
st.dataframe(summary_df.set_index('Metric'), width="stretch")
