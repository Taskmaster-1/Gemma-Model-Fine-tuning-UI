import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from typing import Dict, Optional, Union, List

# Set up logging
logger = logging.getLogger(__name__)

def generate_text(
    model_path: str, 
    prompt: str, 
    max_length: int = 100, 
    temperature: float = 0.7, 
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1,
    device: Optional[str] = None
) -> Union[str, List[str]]:
    """
    Generate text using a fine-tuned model.
    
    Args:
        model_path (str): Path to the fine-tuned model directory
        prompt (str): Input prompt for text generation
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Penalty for repeating tokens
        num_return_sequences (int): Number of sequences to return
        device (str, optional): Device to use ('cpu', 'cuda', 'auto')
        
    Returns:
        Union[str, List[str]]: Generated text or list of generated texts
    """
    try:
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer from the specified path
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto" if device == "auto" else None,
            torch_dtype=torch.float16 if device == "cuda" else None
        )
        
        # Move model to the specified device if not using device_map="auto"
        if device != "auto":
            model = model.to(device)
            
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
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0 else False,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        if num_return_sequences > 1:
            return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        else:
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def batch_generate(
    model_path: str,
    prompts: List[str],
    generation_params: Optional[Dict] = None,
    device: Optional[str] = None
) -> List[str]:
    """
    Generate text for multiple prompts using the same model.
    More efficient than loading the model multiple times.
    
    Args:
        model_path (str): Path to the fine-tuned model
        prompts (List[str]): List of prompts to generate from
        generation_params (Dict, optional): Parameters for generation
        device (str, optional): Device to use ('cpu', 'cuda', 'auto')
        
    Returns:
        List[str]: List of generated texts
    """
    if generation_params is None:
        generation_params = {}
    
    try:
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto" if device == "auto" else None,
            torch_dtype=torch.float16 if device == "cuda" else None
        )
        
        # Move model to device if not using device_map="auto"
        if device != "auto":
            model = model.to(device)
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        results = []
        # Process each prompt
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=generation_params.get("max_length", 100),
                    temperature=generation_params.get("temperature", 0.7),
                    top_p=generation_params.get("top_p", 0.9),
                    top_k=generation_params.get("top_k", 50),
                    repetition_penalty=generation_params.get("repetition_penalty", 1.0),
                    do_sample=generation_params.get("temperature", 0.7) > 0,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}")
        raise