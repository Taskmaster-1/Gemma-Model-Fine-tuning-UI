import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def train_model(dataset_path, model_name, hyperparams):
    dataset = load_dataset("csv", data_files={"train": dataset_path})
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")
    
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    dataset = dataset.map(preprocess, batched=True)

    model = AutoModelForCausalLM.from_pretrained(f"google/{model_name}")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=hyperparams["batch_size"],
        num_train_epochs=hyperparams["epochs"],
        learning_rate=hyperparams["lr"],
        logging_dir="./logs",
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
    trainer.train()
    
    model.save_pretrained(f"models/{model_name}-fine-tuned")
    tokenizer.save_pretrained(f"models/{model_name}-fine-tuned")
