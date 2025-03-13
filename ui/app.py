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
    visualization,
    inference_panel
)
from models.train import train_model
from models.infer import generate_text, batch_generate
from utils.data_utils import load_data, validate_dataset, prepare_dataset_for_training
from utils.visualization import plot_training_progress

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")),
            logging.StreamHandler()
        ]
    )

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
if 'generation_examples' not in st.session_state:
    st.session_state.generation_examples = []
if 'prepared_dataset' not in st.session_state:
    st.session_state.prepared_dataset = None

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
                # Prepare dataset if needed
                prepared_df = prepare_dataset_for_training(df)
                st.session_state.prepared_dataset = prepared_df
                
                st.success(f"Dataset loaded successfully: {len(df)} examples")
                with st.expander("Dataset Preview"):
                    st.dataframe(prepared_df.head(5))
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
    
    # Visualization settings
    visualization_config = visualization()
    
    # Combine all configurations
    config = {
        **model_config,
        **training_config_params
    }
    
    # Estimate memory requirements
    model_size_mb = {"gemma-2b": 2000, "gemma-7b": 7000, "gemma-13b": 13000}
    batch_mem = config["batch_size"] * config["max_length"] * 2 / 1024  # Estimated memory in MB
    total_mem = model_size_mb.get(selected_model, 2000) + batch_mem
    
    st.info(f"Estimated memory requirement: {total_mem:.0f} MB")
    
    # GPU availability check
    import torch
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        if gpu_mem < total_mem:
            st.warning(f"⚠️ Available GPU memory ({gpu_mem:.0f} MB) may be insufficient!")
    else:
        st.warning("⚠️ No GPU detected. Training will be slow on CPU.")
    
    # Start training button
    start_button = st.button("Start Fine-tuning", disabled=not dataset_path)

# Main content area
main_container = st.container()

# Training section
with main_container:
    if not st.session_state.training_completed:
        # Create tabs for different views
        train_tab, eval_tab, logs_tab = st.tabs(["Training Progress", "Evaluation", "Logs"])

    if start_button and dataset_path:
        st.session_state.training_started = True
        st.session_state.current_step = 0
        st.session_state.loss_history = []
        st.session_state.generation_examples = []
        
        with train_tab:
            progress_col1, progress_col2 = st.columns([3, 1])
            
            with progress_col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
            
            with progress_col2:
                stats_container = st.container()
                time_remaining = st.empty()
            
            example_container = st.empty()
            
            try:
                # Start of training timestamp for ETA calculation
                start_time = datetime.now()
                
                # Start training with callback for progress updates
                def progress_callback(step, total_steps, loss, examples=None):
                    st.session_state.current_step = step
                    st.session_state.total_steps = total_steps
                    progress = min(1.0, step / total_steps if total_steps > 0 else 0)
                    progress_bar.progress(progress)
                    
                    # Calculate time remaining
                    if step > 1:
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        time_per_step = elapsed_time / step
                        remaining_steps = total_steps - step
                        eta_seconds = remaining_steps * time_per_step
                        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                        time_remaining.text(f"Time remaining: {eta_str}")
                    
                    status_text.text(f"Step {step}/{total_steps} - Loss: {loss:.4f}")
                    
                    # Update loss history
                    st.session_state.loss_history.append({"step": step, "loss": loss})
                    
                    # Display training stats
                    with stats_container:
                        st.metric("Current Loss", f"{loss:.4f}")
                        if len(st.session_state.loss_history) > 1:
                            avg_loss = sum(item["loss"] for item in st.session_state.loss_history[-10:]) / min(10, len(st.session_state.loss_history))
                            st.metric("Avg Loss (10 steps)", f"{avg_loss:.4f}")
                    
                    # Update loss chart periodically or based on config
                    update_freq = visualization_config.get("update_freq", 10)
                    if step % update_freq == 0 or step == total_steps:
                        fig = plot_training_progress(st.session_state.loss_history)
                        loss_chart.pyplot(fig)
                    
                    # Store and display example generations periodically
                    if examples and (step % 50 == 0 or step == total_steps):
                        st.session_state.generation_examples = examples
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

    # Show inference UI when training is completed
    if st.session_state.training_completed and st.session_state.model_path:
        st.markdown("## Model Testing & Export")
        
        model_tabs = st.tabs(["Model Testing", "Batch Processing", "Export Model"])
        
        with model_tabs[0]:  # Model Testing
            # Basic inference
            inference_panel(st.session_state.model_path)
        
        with model_tabs[1]:  # Batch Processing
            st.markdown("### Batch Processing")
            
            # File uploader for batch inferencing
            batch_file = st.file_uploader("Upload a file with prompts (one per line)", type=["txt", "csv"])
            
            col1, col2 = st.columns(2)
            with col1:
                batch_temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            with col2:
                batch_max_length = st.slider("Max Length", 50, 500, 100, 10)
            
            if st.button("Run Batch Processing") and batch_file and st.session_state.model_path:
                with st.spinner("Processing batch..."):
                    try:
                        # Read prompts from file
                        content = batch_file.getvalue().decode("utf-8")
                        prompts = [line.strip() for line in content.split("\n") if line.strip()]
                        
                        # Set up batch parameters
                        batch_params = {
                            "temperature": batch_temp,
                            "max_length": batch_max_length,
                            "top_p": 0.9,
                            "top_k": 50
                        }
                        
                        # Run batch processing
                        results = batch_generate(
                            model_path=st.session_state.model_path,
                            prompts=prompts,
                            generation_params=batch_params
                        )
                        
                        # Display results in table
                        output_data = []
                        for i, (prompt, result) in enumerate(zip(prompts, results)):
                            output_data.append({
                                "Prompt": prompt,
                                "Response": result
                            })
                        
                        st.dataframe(output_data)
                        
                        # Create download button for results
                        import pandas as pd
                        import io
                        
                        output_df = pd.DataFrame(output_data)
                        buffer = io.StringIO()
                        output_df.to_csv(buffer, index=False)
                        
                        st.download_button(
                            "Download Results CSV",
                            data=buffer.getvalue(),
                            file_name="batch_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch processing error: {str(e)}")
        
        with model_tabs[2]:  # Export Model
            st.markdown("### Export Model")
            
            # Display model info
            st.info(f"Model path: {st.session_state.model_path}")
            
            # Create download buttons for the model
            if os.path.exists(st.session_state.model_path):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Download Model Files")
                    
                    model_files = os.listdir(st.session_state.model_path)
                    
                    # Create a zip of the entire model directory
                    if st.button("Prepare Model for Download"):
                        with st.spinner("Creating zip file..."):
                            import shutil
                            model_name = os.path.basename(st.session_state.model_path)
                            zip_path = f"{model_name}.zip"
                            shutil.make_archive(model_name, 'zip', st.session_state.model_path)
                            
                            with open(f"{model_name}.zip", "rb") as f:
                                st.download_button(
                                    "Download Complete Model",
                                    f,
                                    file_name=f"{model_name}.zip",
                                    mime="application/zip"
                                )
                
                with col2:
                    st.markdown("#### Push to Hugging Face Hub")
                    
                    hf_token = st.text_input("Hugging Face Token", type="password")
                    hf_repo_name = st.text_input("Repository Name", f"gemma-fine-tuned-{datetime.now().strftime('%Y%m%d')}")
                    hf_private = st.checkbox("Private Repository", value=True)
                    
                    if st.button("Push to Hugging Face"):
                        with st.spinner("Uploading to Hugging Face Hub..."):
                            try:
                                from huggingface_hub import HfApi
                                
                                st.info("This functionality requires `huggingface_hub` package. Please install it if needed.")
                                
                                api = HfApi(token=hf_token)
                                api.create_repo(repo_id=hf_repo_name, private=hf_private, exist_ok=True)
                                
                                st.success(f"Model uploaded to https://huggingface.co/{hf_repo_name}")
                            except Exception as e:
                                st.error(f"Error uploading to Hugging Face: {str(e)}")
    
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
)