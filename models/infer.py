import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def generate_text(model_path, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate text using a fine-tuned model.
    
    Args:
        model_path (str): Path to the fine-tuned model directory
        prompt (str): Input prompt for text generation
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: Generated text
    """
    # Load model and tokenizer from the specified path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and return text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text