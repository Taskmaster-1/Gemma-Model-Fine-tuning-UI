import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import components and utilities
from ui.components import (
    dataset_uploader, 
    model_selector, 
    training_config, 
    export_options,
    visualization
)
from models.train import train_model
from utils.data_utils import load_data, validate_dataset
from utils.visualization import plot_training_progress
from utils.logging_utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Gemma Model Fine-tuning UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🚀 Gemma Model Fine-tuning UI")
st.markdown("""
    Fine-tune Google's Gemma models on your custom datasets with an easy-to-use interface.
    Upload your data, configure parameters, and start training with just a few clicks!
""")

# Initialize session state variables if they don't exist
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'total_steps' not in st.session_state:
    st.session_state.total_steps = 0

# Create sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Dataset upload section
    st.subheader("1. Dataset")
    dataset_path = dataset_uploader()
    
    # Show dataset preview if available
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = load_data(dataset_path)
            is_valid, message = validate_dataset(df)
            
            if is_valid:
                st.success(f"Dataset loaded successfully: {len(df)} examples")
                with st.expander("Dataset Preview"):
                    st.dataframe(df.head(5))
            else:
                st.error(f"Invalid dataset: {message}")
                dataset_path = None
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            dataset_path = None
    
    # Model selection section
    st.subheader("2. Model Selection")
    selected_model, model_config = model_selector()
    
    # Training configuration
    st.subheader("3. Training Configuration")
    training_config_params = training_config()
    
    # Export options
    st.subheader("4. Export Options")
    export_options_config = export_options()
    
    # Combine all configurations
    config = {
        "model": model_config,
        "training": training_config_params,
        "export": export_options_config
    }
    
    # Start training button
    start_button = st.button("Start Fine-tuning", disabled=not dataset_path)

# Main content area
main_container = st.container()

# Training section
with main_container:
    if start_button and dataset_path:
        st.session_state.training_started = True
        st.session_state.current_step = 0
        st.session_state.loss_history = []
        
        # Create tabs for different views
        train_tab, eval_tab, logs_tab = st.tabs(["Training Progress", "Evaluation", "Logs"])
        
        with train_tab:
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            example_container = st.empty()
            
            try:
                # Start training with callback for progress updates
                def progress_callback(step, total_steps, loss, examples=None):
                    st.session_state.current_step = step
                    st.session_state.total_steps = total_steps
                    progress = step / total_steps
                    progress_bar.progress(progress)
                    
                    status_text.text(f"Step {step}/{total_steps} - Loss: {loss:.4f}")
                    
                    # Update loss history
                    st.session_state.loss_history.append({"step": step, "loss": loss})
                    
                    # Update loss chart periodically
                    if step % 10 == 0 or step == total_steps:
                        fig = plot_training_progress(st.session_state.loss_history)
                        loss_chart.pyplot(fig)
                    
                    # Display example generations periodically
                    if examples and (step % 50 == 0 or step == total_steps):
                        example_container.markdown("### Example Generations")
                        for i, (prompt, completion) in enumerate(examples):
                            example_container.markdown(f"**Prompt {i+1}:** {prompt}")
                            example_container.markdown(f"**Completion {i+1}:** {completion}")
                
                # Run training with progress callback
                logger.info(f"Starting training with model {selected_model} on dataset {dataset_path}")
                output_dir = os.path.join("models", f"{selected_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                
                model_path = train_model(
                    dataset_path=dataset_path,
                    model_name=selected_model,
                    hyperparams=config,
                    output_dir=output_dir,
                    progress_callback=progress_callback
                )
                
                st.session_state.model_path = model_path
                st.session_state.training_completed = True
                
                # Show completion message
                st.success(f"Training completed! Model saved to: {model_path}")
                
            except Exception as e:
                logger.error(f"Training error: {str(e)}", exc_info=True)
                st.error(f"Training failed: {str(e)}")
                st.session_state.training_started = False

    # Show training results if completed
    elif st.session_state.training_completed and st.session_state.model_path:
        st.success(f"Training completed! Model is ready for use.")
        
        # Create download buttons for the model
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Model (PyTorch)",
                data=open(os.path.join(st.session_state.model_path, "pytorch_model.bin"), "rb"),
                file_name="pytorch_model.bin",
                mime="application/octet-stream"
            )
        
        with col2:
            if os.path.exists(os.path.join(st.session_state.model_path, "tf_model.h5")):
                st.download_button(
                    label="Download Model (TensorFlow)",
                    data=open(os.path.join(st.session_state.model_path, "tf_model.h5"), "rb"),
                    file_name="tf_model.h5",
                    mime="application/octet-stream"
                )
        
        # Model testing interface
        st.subheader("Test Your Fine-tuned Model")
        test_prompt = st.text_area("Enter a test prompt:", height=100)
        
        if st.button("Generate") and test_prompt:
            from models.infer import generate_text
            
            with st.spinner("Generating..."):
                try:
                    generated_text = generate_text(
                        model_path=st.session_state.model_path,
                        prompt=test_prompt
                    )
                    st.markdown("### Generated Output")
                    st.markdown(generated_text)
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
    
    # Welcome screen for first-time users
    elif not st.session_state.training_started:
        st.info("👈 Start by uploading your dataset and configuring your model in the sidebar.")
        
        # Show example workflow
        with st.expander("How to use this tool"):
            st.markdown("""
                ### Quick Start Guide
                
                1. **Upload your dataset** - Supported formats: CSV, JSONL, TXT
                2. **Select a Gemma model** - Choose from available model sizes
                3. **Configure training parameters** - Set learning rate, batch size, etc.
                4. **Choose export options** - Select formats for the fine-tuned model
                5. **Start training** - Monitor progress in real-time
                6. **Test and download** - Try your model and download the weights
                
                For detailed instructions, see the [documentation](docs/user_guide.md).
            """)
            
            st.image("https://via.placeholder.com/800x400?text=Workflow+Diagram", caption="Fine-tuning workflow")

# Footer
st.markdown("---")
st.markdown(
    "Developed for Google Summer of Code | "
    "[GitHub Repository](https://github.com/Taskmaster-1/Gemma-Model-Fine-tuning-UI) | "
    "[Report Issues](https://github.com/Taskmaster-1/Gemma-Model-Fine-tuning-UI/issues)"
<<<<<<< HEAD
)
=======
)
>>>>>>> 7ccfbc3e8009f1a64510e629f86fea3f0e696407
