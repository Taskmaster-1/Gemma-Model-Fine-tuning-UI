import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
from datetime import datetime

class CustomCallback(transformers.TrainerCallback):
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress_callback and logs:
            step = state.global_step
            total_steps = state.max_steps
            loss = logs.get("loss", 0)
            self.progress_callback(step, total_steps, loss)
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_callback:
            step = state.global_step
            total_steps = state.max_steps
            loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
            self.progress_callback(step, total_steps, loss)

def train_model(dataset_path, model_name, hyperparams, output_dir="./models", progress_callback=None):
    """
    Train a Gemma model on a custom dataset.
    
    Args:
        dataset_path (str): Path to the dataset file
        model_name (str): Name of the model (e.g., "gemma-2b")
        hyperparams (dict): Dictionary containing training hyperparameters
        output_dir (str): Directory to save the trained model
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        str: Path to the saved model
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    file_extension = os.path.splitext(dataset_path)[1].lower()
    if file_extension == '.csv':
        dataset = load_dataset("csv", data_files={"train": dataset_path})
    elif file_extension == '.jsonl':
        dataset = load_dataset("json", data_files={"train": dataset_path})
    elif file_extension == '.txt':
        # For text files, we need special handling
        with open(dataset_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Convert to a format dataset can handle
        import pandas as pd
        df = pd.DataFrame({"text": lines})
        df.to_csv(f"{dataset_path}.csv", index=False)
        dataset = load_dataset("csv", data_files={"train": f"{dataset_path}.csv"})
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Get correct model name for loading from Hugging Face
    full_model_name = f"google/{model_name}" if not model_name.startswith("google/") else model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess dataset
    def preprocess(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=hyperparams.get("max_length", 512)
        )

    tokenized_dataset = dataset.map(
        preprocess, 
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        device_map="auto"
    )
    
    # Extract hyperparameters with defaults
    batch_size = hyperparams.get("batch_size", 16)
    epochs = hyperparams.get("epochs", 3)
    learning_rate = hyperparams.get("lr", 3e-5)
    
    # Calculate total steps for progress tracking
    total_steps = len(tokenized_dataset["train"]) // batch_size * epochs
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",  # Disable wandb, etc.
    )
    
    # Initialize trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        callbacks=[CustomCallback(progress_callback)] if progress_callback else None,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f"{model_name}-fine-tuned")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    return final_model_path