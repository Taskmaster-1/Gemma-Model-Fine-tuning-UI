import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_name, prompt):
    model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}-fine-tuned")
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}-fine-tuned")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
