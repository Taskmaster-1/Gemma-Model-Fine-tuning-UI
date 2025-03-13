import pandas as pd
import json

def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        return pd.DataFrame([json.loads(line) for line in open(file_path)])
    elif file_path.endswith(".txt"):
        return pd.DataFrame({"text": open(file_path).read().split("\n")})
    else:
        raise ValueError("Unsupported file format")
