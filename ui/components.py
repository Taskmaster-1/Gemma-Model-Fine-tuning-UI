import streamlit as st

def dataset_uploader():
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV, JSONL, or text file", type=["csv", "jsonl", "txt"])
    if uploaded_file:
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def model_selector():
    st.subheader("🤖 Select Model & Hyperparameters")
    model_name = st.selectbox("Choose a Gemma model", ["gemma-2b", "gemma-7b", "gemma-13b"])
    learning_rate = st.slider("Learning Rate", min_value=1e-5, max_value=1e-2, value=3e-4, step=1e-5)
    epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])
    
    return model_name, {"lr": learning_rate, "epochs": epochs, "batch_size": batch_size}
