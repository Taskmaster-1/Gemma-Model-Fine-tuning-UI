import streamlit as st
from ui.components import dataset_uploader, model_selector
from models.train import train_model

st.set_page_config(page_title="Gemma Model Fine-tuning UI", layout="wide")

st.title("🚀 Fine-tune Gemma Models Easily")

# Dataset Upload Section
dataset_path = dataset_uploader()

# Model Selection Section
selected_model, hyperparams = model_selector()

# Train Button
if st.button("Start Fine-tuning"):
    with st.spinner("Training in progress..."):
        train_model(dataset_path, selected_model, hyperparams)
    st.success("Fine-tuning completed!")
