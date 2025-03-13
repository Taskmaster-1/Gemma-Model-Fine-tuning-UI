import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List, Tuple

logger = logging.getLogger(__name__)

class CustomCallback(transformers.TrainerCallback):
    """
    Custom callback for training progress tracking and visualization.
    """
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the callback.
        
        Args:
            progress_callback: Function to call with progress updates
        """
        self.progress_callback = progress_callback
        self.example_prompts = [
            "What is machine learning?",
            "Explain how fine-tuning works.",
            "Describe the benefits of LLMs."
        ]
        self.generation_examples = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle logging events during training."""
        if self.progress_callback and logs:
            step = state.global_step
            total_steps = state.max_steps
            loss = logs.get("loss", 0)
            self.progress_callback(step, total_steps, loss, self.generation_examples)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Handle step end events during training."""
        if self.progress_callback and state.log_history:
            step = state.global_step
            total_steps = state.max_steps
            loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
            self.progress_callback(step, total_steps, loss, self.generation_examples)
            
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate examples when evaluation occurs."""
        if model and state.global_step % 50 == 0:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                self.generation_examples = []
                for prompt in self.example_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_length=100, temperature=0.7, 
                            top_p=0.9, do_sample=True
                        )
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    self.generation_examples.append((prompt, generated))
            except Exception as e:
                logger.error(f"Error generating examples: {str(e)}")
                # Don't crash the training process if example generation fails
                pass


def train_model(
    dataset_path: str, 
    model_name: str, 
    hyperparams: Dict[str, Any], 
    output_dir: str = "./models",
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Train a Gemma model on a custom dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the model (e.g., "gemma-2b")
        hyperparams: Dictionary containing training hyperparameters
        output_dir: Directory to save the trained model
        progress_callback: Function to call with progress updates
        
    Returns:
        Path to the saved model
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    try:
        # Log training start
        logger.info(f"Starting training for model {model_name} on dataset {dataset_path}")
        logger.info(f"Hyperparameters: {hyperparams}")
        
        # Load dataset based on file extension
        file_extension = os.path.splitext(dataset_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                dataset = load_dataset("csv", data_files={"train": dataset_path})
            elif file_extension == '.jsonl' or file_extension == '.json':
                dataset = load_dataset("json", data_files={"train": dataset_path})
            elif file_extension == '.txt':
                # For text files, we need special handling
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                # Convert to a format dataset can handle
                import pandas as pd
                df = pd.DataFrame({"text": lines})
                temp_csv = f"{dataset_path}.csv"
                df.to_csv(temp_csv, index=False)
                dataset = load_dataset("csv", data_files={"train": temp_csv})
                
                # Clean up temporary file
                os.remove(temp_csv)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            logger.info(f"Dataset loaded successfully with {len(dataset['train'])} examples")
            
            # Add a validation split if the dataset is large enough
            if len(dataset['train']) > 100:
                dataset = dataset['train'].train_test_split(test_size=0.1)
            else:
                # Create a dummy validation set
                dataset = {
                    'train': dataset['train'],
                    'validation': dataset['train'].select(range(min(10, len(dataset['train']))))
                }
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Get correct model name for loading from Hugging Face
        full_model_name = f"google/{model_name}" if not model_name.startswith("google/") else model_name
        
        # Load tokenizer with error handling
        try:
            tokenizer = AutoTokenizer.from_pretrained(full_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer loaded successfully from {full_model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
        # Preprocess dataset
        def preprocess(examples):
            # Detect columns based on their names
            column_names = list(examples.keys())
            
            # If 'text' column exists, use it directly
            if 'text' in column_names:
                text_column = 'text'
            else:
                # Try to identify input/output columns
                input_cols = [col for col in column_names if any(x in col.lower() for x in ['input', 'prompt', 'question'])]
                output_cols = [col for col in column_names if any(x in col.lower() for x in ['output', 'response', 'completion', 'answer'])]
                
                if input_cols and output_cols:
                    # Format based on the model type (chat or instruction)
                    is_chat = hyperparams.get("format", "instruction").lower() == "chat"
                    
                    # Create a new 'text' column with the combined input/output
                    input_col = input_cols[0]
                    output_col = output_cols[0]
                    
                    if is_chat:
                        examples['text'] = [f"USER: {ex[input_col]}\nASSISTANT: {ex[output_col]}" for ex in examples]
                    else:
                        examples['text'] = [f"INSTRUCTION: {ex[input_col]}\nRESPONSE: {ex[output_col]}" for ex in examples]
                    
                    text_column = 'text'
                else:
                    # If we can't identify specific columns, use the first column
                    text_column = column_names[0]
                    logger.warning(f"Could not identify input/output columns. Using '{text_column}' as text column.")
            
            # Tokenize the text
            tokenized = tokenizer(
                examples[text_column], 
                truncation=True, 
                padding="max_length", 
                max_length=hyperparams.get("max_length", 512)
            )
            
            return tokenized

        try:
            # Process train and validation sets separately
            tokenized_train = dataset['train'].map(
                preprocess, 
                batched=True,
                remove_columns=dataset['train'].column_names
            )
            
            tokenized_validation = dataset['validation'].map(
                preprocess, 
                batched=True,
                remove_columns=dataset['validation'].column_names
            )
            
            tokenized_dataset = {
                'train': tokenized_train,
                'validation': tokenized_validation
            }
            
            logger.info("Dataset tokenized successfully")
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise
        
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # We're doing causal language modeling, not masked LM
        )
        
        # Load model with memory optimization
        try:
            # Add gradient checkpointing for large models
            use_gradient_checkpointing = hyperparams.get("model_size", "small") != "small"
            
            # Use low_cpu_mem_usage for more efficient loading
            model = AutoModelForCausalLM.from_pretrained(
                full_model_name,
                device_map="auto",
                use_cache=not use_gradient_checkpointing,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16 if hyperparams.get("mixed_precision") == "bf16" else None
            )
            
            # Enable gradient checkpointing if needed
            if use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
                
            logger.info(f"Model loaded successfully from {full_model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Extract hyperparameters with defaults
        batch_size = hyperparams.get("batch_size", 16)
        epochs = hyperparams.get("epochs", 3)
        learning_rate = hyperparams.get("lr", 3e-5)
        weight_decay = hyperparams.get("weight_decay", 0.01)
        warmup_ratio = hyperparams.get("warmup_ratio", 0.03)
        gradient_accumulation = hyperparams.get("gradient_accumulation", 1)
        lr_scheduler = hyperparams.get("lr_scheduler", "linear")
        mixed_precision = hyperparams.get("mixed_precision", "no")
        
        # Select optimizer based on hyperparams
        optimizer = hyperparams.get("optimizer", "AdamW").lower()
        optim = "adamw_torch" if optimizer == "adamw" else "adafactor" if optimizer == "adafactor" else "sgd"
        
        # Calculate total steps for progress tracking
        total_steps = (len(tokenized_dataset["train"]) // batch_size) * epochs
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",  # Disable wandb, etc.
            lr_scheduler_type=lr_scheduler,
            fp16=(mixed_precision == "fp16"),
            bf16=(mixed_precision == "bf16"),
            evaluation_strategy="steps",
            eval_steps=50,  # Evaluate every 50 steps
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim=optim,
            # Add dataloader num workers for faster data loading
            dataloader_num_workers=2,
        )
        
        # Initialize trainer with custom callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=[CustomCallback(progress_callback)] if progress_callback else None,
        )
        
        # Start training
        try:
            logger.info("Starting training")
            trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        # Save the final model
        try:
            final_model_path = os.path.join(output_dir, f"{model_name}-fine-tuned")
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            logger.info(f"Model saved to {final_model_path}")
            
            # Create a model card with training details
            with open(os.path.join(final_model_path, "README.md"), "w") as f:
                f.write(f"# Fine-tuned {model_name} Model\n\n")
                f.write(f"This model was fine-tuned on a custom dataset with the following parameters:\n\n")
                f.write("```\n")
                for key, value in hyperparams.items():
                    f.write(f"{key}: {value}\n")
                f.write("```\n\n")
                f.write(f"Training completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
        
        return final_model_path
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Clean up any temporary files if needed
        raise