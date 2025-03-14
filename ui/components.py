import streamlit as st
import os
import uuid

def dataset_uploader():
    """
    Create a file uploader for datasets.
    
    Returns:
        str: Path to the uploaded file, or None if no file was uploaded.
    """
    uploaded_file = st.file_uploader("Upload a CSV, JSONL, or text file", type=["csv", "jsonl", "txt"])
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
        file_path = os.path.join("data", unique_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def model_selector():
    """
    Create UI elements for selecting a model and configuring basic hyperparameters.
    
    Returns:
        tuple: (selected_model, model_config)
    """
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "Choose a Gemma model", 
            ["gemma-2b", "gemma-7b", "gemma-13b"]
        )
    with col2:
        instruction_format = st.selectbox(
            "Model Format", 
            ["Chat", "Instruction"], 
            help="Chat models use a conversation format, Instruction models follow instructions directly"
        )
    
    st.write("#### Basic Parameters")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider(
            "Learning Rate", 
            min_value=1e-6, 
            max_value=1e-3, 
            value=3e-5, 
            format="%.6f",
            help="Lower values (e.g., 1e-6) for stability, higher values (e.g., 1e-3) for faster learning"
        )
    with col2:
        epochs = st.slider(
            "Epochs", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Number of times to iterate through the entire dataset"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.select_slider(
            "Batch Size", 
            options=[1, 2, 4, 8, 16, 32, 64],
            value=8,
            help="Larger batch sizes need more memory but train faster"
        )
    with col2:
        max_length = st.select_slider(
            "Sequence Length", 
            options=[128, 256, 512, 1024, 2048],
            value=512,
            help="Maximum token length for each example"
        )
    
    model_config = {
        "lr": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "format": instruction_format.lower()
    }
    return model_name, model_config

def training_config():
    """
    Create UI elements for advanced training configuration.
    
    Returns:
        dict: Training configuration parameters.
    """
    with st.expander("Advanced Training Options"):
        col1, col2 = st.columns(2)
        with col1:
            weight_decay = st.slider(
                "Weight Decay", 
                min_value=0.0, 
                max_value=0.2, 
                value=0.01,
                step=0.01,
                help="Regularization parameter to prevent overfitting"
            )
            warmup_ratio = st.slider(
                "Warmup Ratio", 
                min_value=0.0, 
                max_value=0.2, 
                value=0.03,
                step=0.01,
                help="Portion of training to use for learning rate warmup"
            )
            gradient_accumulation = st.slider(
                "Gradient Accumulation Steps", 
                min_value=1, 
                max_value=16, 
                value=1,
                help="Accumulate gradients over multiple steps (helps with limited memory)"
            )
        with col2:
            optimizer = st.selectbox(
                "Optimizer", 
                ["AdamW", "Adafactor", "SGD"],
                index=0,
                help="Optimization algorithm to use during training"
            )
            lr_scheduler = st.selectbox(
                "LR Scheduler", 
                ["linear", "cosine", "cosine_with_restarts", "constant"],
                index=0,
                help="Learning rate schedule during training"
            )
            mixed_precision = st.selectbox(
                "Mixed Precision", 
                ["no", "fp16", "bf16"],
                index=0,
                help="Use mixed precision to reduce memory usage (if supported by hardware)"
            )
    return {
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation": gradient_accumulation,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "mixed_precision": mixed_precision
    }

def export_options():
    """
    Create UI elements for model export configuration.
    
    Returns:
        dict: Export configuration parameters.
    """
    with st.expander("Model Export Options"):
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.multiselect(
                "Export Formats", 
                ["PyTorch", "TensorFlow", "GGUF", "ONNX"],
                default=["PyTorch"],
                help="Formats to export the model to"
            )
            quantization = st.selectbox(
                "Quantization", 
                ["None", "8-bit (int8)", "4-bit (int4)"],
                index=0,
                help="Reduce model size with quantization (may affect quality)"
            )
        with col2:
            hub_upload = st.checkbox(
                "Upload to Hugging Face Hub", 
                value=False,
                help="Share your model on Hugging Face Hub"
            )
            if hub_upload:
                hub_model_id = st.text_input(
                    "Model ID", 
                    value="username/gemma-fine-tuned",
                    help="Your Hugging Face username and model name (e.g., username/model-name)"
                )
                hub_private = st.checkbox(
                    "Private Repository", 
                    value=True,
                    help="Make the repository private (requires Pro account for large models)"
                )
            else:
                hub_model_id = None
                hub_private = True
    return {
        "export_format": export_format,
        "quantization": quantization,
        "hub_upload": hub_upload,
        "hub_model_id": hub_model_id,
        "hub_private": hub_private
    }

def visualization():
    """
    Create UI elements for visualization settings.
    
    Returns:
        dict: Visualization configuration parameters.
    """
    with st.expander("Visualization Settings"):
        col1, col2 = st.columns(2)
        with col1:
            plot_loss = st.checkbox("Plot Loss", value=True)
            plot_accuracy = st.checkbox("Plot Accuracy", value=True)
        with col2:
            update_freq = st.select_slider(
                "Update Frequency", 
                options=[1, 5, 10, 25, 50],
                value=10,
                help="How often to update charts (in steps)"
            )
            chart_height = st.slider(
                "Chart Height", 
                min_value=200, 
                max_value=600, 
                value=300,
                step=50,
                help="Height of the charts in pixels"
            )
    return {
        "plot_loss": plot_loss,
        "plot_accuracy": plot_accuracy,
        "update_freq": update_freq,
        "chart_height": chart_height
    }

def inference_panel(model_path):
    """
    Create UI elements for testing the trained model through inference.
    
    Args:
        model_path (str): Path to the trained model.
    """
    import streamlit as st
    import torch
    from models.infer import generate_text
    from datetime import datetime
    
    st.markdown("### Test Your Fine-tuned Model")
    test_prompt = st.text_area("Enter a test prompt:", height=100)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                  help="Higher values make output more random, lower values more deterministic")
    with col2:
        max_length = st.slider("Max Length", 50, 500, 100, 10,
                                help="Maximum length of generated text")
    with col3:
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1,
                         help="Nucleus sampling parameter")
    
    if st.button("Generate"):
        if test_prompt:
            with st.spinner("Generating response..."):
                try:
                    response = generate_text(
                        model_path=model_path,
                        prompt=test_prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    st.markdown("### Generated Response:")
                    st.markdown(response)
                    if st.button("Save this example"):
                        os.makedirs("examples", exist_ok=True)
                        example = {
                            "prompt": test_prompt,
                            "response": response,
                            "parameters": {
                                "temperature": temperature,
                                "max_length": max_length,
                                "top_p": top_p
                            },
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        filename = f"example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(os.path.join("examples", filename), "w") as f:
                            import json
                            json.dump(example, f, indent=2)
                        st.success(f"Example saved as {filename}")
                except Exception as e:
                    st.error(f"Error generating text: {str(e)}")
        else:
            st.warning("Please enter a prompt first.")
