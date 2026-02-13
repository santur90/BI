import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="BioInfo Analysis", layout="wide")

st.title("🧬 Bioinformatics Data Analysis Platform")
st.markdown("""
Run DNAme & RNA-seq data analysis directly in your browser.
**Privacy Note:** Your data is processed locally in your browser and is never uploaded to any server.
""")

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
