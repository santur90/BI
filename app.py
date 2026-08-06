import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="BioInfo Analysis", layout="wide")

st.title("🧬 Bioinformatics Data Analysis Platform")
st.markdown("""
Run DNAme & RNA-seq data analysis directly in your browser.
**Privacy Note:** Your data is processed locally in your browser and is never uploaded to any server.
""")


def plot_workflow_schematic():
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        (0.7, "Upload\nMatrix"),
        (3.7, "Data\nPreview"),
        (6.7, "Basic\nStatistics"),
        (9.7, "Correlation\nHeatmap"),
    ]

    for x, label in steps:
        box = FancyBboxPatch(
            (x, 1),
            2.2,
            1.1,
            boxstyle="round,pad=0.2,rounding_size=0.15",
            linewidth=1.5,
            edgecolor="#1f4e79",
            facecolor="#dceaf7",
        )
        ax.add_patch(box)
        ax.text(x + 1.1, 1.55, label, ha="center", va="center", fontsize=11, color="#0f2f4f")

    # Draw arrows between steps to show the analysis flow.
    for i in range(len(steps) - 1):
        start = steps[i][0] + 2.2
        end = steps[i + 1][0]
        ax.annotate(
            "",
            xy=(end - 0.15, 1.55),
            xytext=(start + 0.15, 1.55),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f4e79"),
        )

    ax.set_title("Analysis Workflow Schematic", fontsize=13, pad=10)
    return fig


st.subheader("Workflow Schematic")
st.pyplot(plot_workflow_schematic())

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Count Matrix / Beta Values (CSV/TSV)", type=["csv", "tsv", "txt"])

if uploaded_file is not None:
    try:
        # Determine separator
        if uploaded_file.name.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
            
        df = pd.read_csv(uploaded_file, sep=sep, index_col=0)
        st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} samples.")
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Basic Stats
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        # Visualization Demo
        st.subheader("Sample Correlation Heatmap")
        if st.button("Generate Heatmap"):
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap='viridis', ax=ax)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a file to start analysis.")

st.sidebar.markdown("---")
st.sidebar.info("Framework: Stlite (Python in Browser)")
