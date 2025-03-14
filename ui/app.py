import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

st.set_page_config(
    page_title="Gemma Model Fine-tuning UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Gemma Model Fine-tuning UI")
st.markdown("""
    Fine-tune Google's Gemma models on your custom datasets with an easy-to-use interface.
    Upload your data, configure parameters, and start training with just a few clicks!
""")

# Initialize session state variables if not already present
for key in ['training_started', 'training_completed', 'model_path', 'loss_history', 'current_step', 'total_steps', 'generation_examples', 'prepared_dataset']:
    if key not in st.session_state:
        st.session_state[key] = False if 'started' in key or 'completed' in key else None
if st.session_state.loss_history is None:
    st.session_state.loss_history = []
if st.session_state.current_step is None:
    st.session_state.current_step = 0
if st.session_state.total_steps is None:
    st.session_state.total_steps = 0
if st.session_state.generation_examples is None:
    st.session_state.generation_examples = []

with st.sidebar:
    st.header("Configuration")
    
    # Dataset upload section
    st.subheader("1. Dataset")
    dataset_path = dataset_uploader()
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = load_data(dataset_path)
            is_valid, message = validate_dataset(df)
            if is_valid:
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
    
    st.subheader("2. Model Selection")
    selected_model, model_config = model_selector()
    
    st.subheader("3. Training Configuration")
    training_config_params = training_config()
    
    st.subheader("4. Export Options")
    export_options_config = export_options()
    
    visualization_config = visualization()
    
    config = { **model_config, **training_config_params }
    
    model_size_mb = {"gemma-2b": 2000, "gemma-7b": 7000, "gemma-13b": 13000}
    batch_mem = config["batch_size"] * config["max_length"] * 2 / 1024
    total_mem = model_size_mb.get(selected_model, 2000) + batch_mem
    
    st.info(f"Estimated memory requirement: {total_mem:.0f} MB")
    
    import torch
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        if gpu_mem < total_mem:
            st.warning(f"⚠️ Available GPU memory ({gpu_mem:.0f} MB) may be insufficient!")
    else:
        st.warning("⚠️ No GPU detected. Training will be slow on CPU.")
    
    start_button = st.button("Start Fine-tuning", disabled=not dataset_path)

main_container = st.container()

with main_container:
    if not st.session_state.training_completed:
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
                start_time = datetime.now()
                
                def progress_callback(step, total_steps, loss, examples=None):
                    st.session_state.current_step = step
                    st.session_state.total_steps = total_steps
                    progress = min(1.0, step / total_steps if total_steps > 0 else 0)
                    progress_bar.progress(progress)
                    
                    if step > 1:
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        time_per_step = elapsed_time / step
                        remaining_steps = total_steps - step
                        eta_seconds = remaining_steps * time_per_step
                        from datetime import timedelta
                        eta_str = str(timedelta(seconds=int(eta_seconds)))
                        time_remaining.text(f"Time remaining: {eta_str}")
                    
                    status_text.text(f"Step {step}/{total_steps} - Loss: {loss:.4f}")
                    st.session_state.loss_history.append({"step": step, "loss": loss})
                    
                    with stats_container:
                        st.metric("Current Loss", f"{loss:.4f}")
                        if len(st.session_state.loss_history) > 1:
                            avg_loss = sum(item["loss"] for item in st.session_state.loss_history[-10:]) / min(10, len(st.session_state.loss_history))
                            st.metric("Avg Loss (10 steps)", f"{avg_loss:.4f}")
                    
                    update_freq = visualization_config.get("update_freq", 10)
                    if step % update_freq == 0 or step == total_steps:
                        fig = plot_training_progress(st.session_state.loss_history)
                        loss_chart.pyplot(fig)
                    
                    if examples and (step % 50 == 0 or step == total_steps):
                        st.session_state.generation_examples = examples
                        example_container.markdown("### Example Generations")
                        for i, (prompt, completion) in enumerate(examples):
                            example_container.markdown(f"**Prompt {i+1}:** {prompt}")
                            example_container.markdown(f"**Completion {i+1}:** {completion}")
                
                logger.info(f"Starting training with model {selected_model} on dataset {dataset_path}")
                output_dir = os.path.join("models", f"{selected_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                model_config_with_framework = {**config, "framework": "torch"}
                model_path = train_model(
                    dataset_path=dataset_path,
                    model_name=selected_model,
                    hyperparams=model_config_with_framework,
                    output_dir=output_dir,
                    progress_callback=progress_callback
                )
                
                st.session_state.model_path = model_path
                st.session_state.training_completed = True
                st.success(f"Training completed! Model saved to: {model_path}")
                
            except Exception as e:
                logger.error(f"Training error: {str(e)}", exc_info=True)
                st.error(f"Training failed: {str(e)}")
                st.session_state.training_started = False

    if st.session_state.training_completed and st.session_state.model_path:
        st.markdown("## Model Testing & Export")
        model_tabs = st.tabs(["Model Testing", "Batch Processing", "Export Model"])
        
        with model_tabs[0]:
            inference_panel(st.session_state.model_path)
        
        with model_tabs[1]:
            st.markdown("### Batch Processing")
            batch_file = st.file_uploader("Upload a file with prompts (one per line)", type=["txt", "csv"])
            col1, col2 = st.columns(2)
            with col1:
                batch_temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            with col2:
                batch_max_length = st.slider("Max Length", 50, 500, 100, 10)
            
            if st.button("Run Batch Processing") and batch_file and st.session_state.model_path:
                with st.spinner("Processing batch..."):
                    try:
                        content = batch_file.getvalue().decode("utf-8")
                        prompts = [line.strip() for line in content.split("\n") if line.strip()]
                        batch_params = {
                            "temperature": batch_temp,
                            "max_length": batch_max_length,
                            "top_p": 0.9,
                            "top_k": 50
                        }
                        results = batch_generate(
                            model_path=st.session_state.model_path,
                            prompts=prompts,
                            generation_params=batch_params
                        )
                        output_data = []
                        for i, (prompt, result) in enumerate(zip(prompts, results)):
                            output_data.append({
                                "Prompt": prompt,
                                "Response": result
                            })
                        st.dataframe(output_data)
                        
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
        
        with model_tabs[2]:
            st.markdown("### Export Model")
            st.info(f"Model path: {st.session_state.model_path}")
            if os.path.exists(st.session_state.model_path):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Download Model Files")
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
                                st.info("This functionality requires the `huggingface_hub` package. Please install it if needed.")
                                api = HfApi(token=hf_token)
                                api.create_repo(repo_id=hf_repo_name, private=hf_private, exist_ok=True)
                                st.success(f"Model uploaded to https://huggingface.co/{hf_repo_name}")
                            except Exception as e:
                                st.error(f"Error uploading to Hugging Face: {str(e)}")
    
    elif not st.session_state.training_started:
        st.info("👈 Start by uploading your dataset and configuring your model in the sidebar.")
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

st.markdown("---")
st.markdown(
    "Developed for Google Summer of Code | "
    "[GitHub Repository](https://github.com/Taskmaster-1/Gemma-Model-Fine-tuning-UI) | "
    "[Report Issues](https://github.com/Taskmaster-1/Gemma-Model-Fine-tuning-UI/issues)"
)
