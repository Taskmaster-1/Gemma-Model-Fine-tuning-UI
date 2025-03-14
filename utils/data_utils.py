import pandas as pd
import json
import os

def load_data(file_path):
    """
    Load data from various file formats.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return pd.DataFrame([json.loads(line) for line in f])
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": lines})
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")

def validate_dataset(df):
    """
    Validate that a dataset is suitable for fine-tuning.
    
    Args:
        df (pd.DataFrame): Dataset to validate.
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string.
    """
    if df.empty:
        return False, "Dataset is empty"
    
    if 'text' not in df.columns:
        potential_input_cols = [col for col in df.columns if any(x in col.lower() for x in ['input', 'prompt', 'question'])]
        potential_output_cols = [col for col in df.columns if any(x in col.lower() for x in ['output', 'response', 'completion', 'answer'])]
        if potential_input_cols and potential_output_cols:
            return True, f"Dataset has input/output columns: {potential_input_cols[0]}/{potential_output_cols[0]}"
        return False, "Dataset must have a 'text' column or input/output columns"
    
    if df['text'].isna().any() or (df['text'] == '').any():
        return False, "Dataset contains empty text values"
    
    if (df['text'].str.len() < 10).any():
        return True, "Warning: Some texts are very short (<10 chars)"
    
    return True, "Dataset is valid"

def prepare_dataset_for_training(df):
    """
    Prepare a dataset for training by ensuring it has a 'text' column.
    
    Args:
        df (pd.DataFrame): Dataset to prepare.
        
    Returns:
        pd.DataFrame: Prepared dataset with a 'text' column.
    """
    if 'text' in df.columns:
        return df
    
    potential_input_cols = [col for col in df.columns if any(x in col.lower() for x in ['input', 'prompt', 'question'])]
    potential_output_cols = [col for col in df.columns if any(x in col.lower() for x in ['output', 'response', 'completion', 'answer'])]
    
    if potential_input_cols and potential_output_cols:
        input_col = potential_input_cols[0]
        output_col = potential_output_cols[0]
        new_df = pd.DataFrame({
            'text': df.apply(lambda row: f"Input: {row[input_col]}\nOutput: {row[output_col]}", axis=1)
        })
        return new_df
    
    return pd.DataFrame({'text': df.iloc[:, 0]})
