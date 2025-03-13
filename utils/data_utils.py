import pandas as pd
import json
import os

def load_data(file_path):
    """
    Load data from various file formats.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        return pd.DataFrame([json.loads(line) for line in open(file_path)])
    elif file_path.endswith(".txt"):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": lines})
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")

def validate_dataset(df):
    """
    Validate that a dataset is suitable for fine-tuning.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string
    """
    # Check if the dataframe is empty
    if df.empty:
        return False, "Dataset is empty"
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        # Check if we can identify input/output columns
        potential_input_cols = [col for col in df.columns if any(x in col.lower() for x in ['input', 'prompt', 'question'])]
        potential_output_cols = [col for col in df.columns if any(x in col.lower() for x in ['output', 'response', 'completion', 'answer'])]
        
        if potential_input_cols and potential_output_cols:
            return True, f"Dataset has input/output columns: {potential_input_cols[0]}/{potential_output_cols[0]}"
        
        return False, "Dataset must have a 'text' column or input/output columns"
    
    # Check for empty text values
    if df['text'].isna().any() or (df['text'] == '').any():
        return False, "Dataset contains empty text values"
    
    # Check if texts are too short
    if (df['text'].str.len() < 10).any():
        return True, "Warning: Some texts are very short (<10 chars)"
    
    return True, "Dataset is valid"

def prepare_dataset_for_training(df):
    """
    Prepare a dataset for training by ensuring it has the right format.
    
    Args:
        df (pd.DataFrame): Dataset to prepare
        
    Returns:
        pd.DataFrame: Prepared dataset
    """
    # If dataset already has a 'text' column, use it
    if 'text' in df.columns:
        return df
    
    # Try to identify input/output columns
    potential_input_cols = [col for col in df.columns if any(x in col.lower() for x in ['input', 'prompt', 'question'])]
    potential_output_cols = [col for col in df.columns if any(x in col.lower() for x in ['output', 'response', 'completion', 'answer'])]
    
    if potential_input_cols and potential_output_cols:
        input_col = potential_input_cols[0]
        output_col = potential_output_cols[0]
        
        # Create a new dataframe with a 'text' column combining input and output
        new_df = pd.DataFrame({
            'text': df.apply(lambda row: f"Input: {row[input_col]}\nOutput: {row[output_col]}", axis=1)
        })
        return new_df
    
    # If we can't identify the right columns, use the first column as 'text'
    return pd.DataFrame({'text': df.iloc[:, 0]})