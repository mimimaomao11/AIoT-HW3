"""Utility functions for the Streamlit app."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

def find_csv_files(root_dir: Path) -> Dict[str, Path]:
    """Find all CSV files in the given directory and its subdirectories."""
    datasets = {}
    
    # Find all CSV files in the root directory
    for csv_file in root_dir.glob("*.csv"):
        datasets[f"{csv_file.parent.name}/{csv_file.name}"] = csv_file
    
    # Find all CSV files in the processed directory
    processed_dir = root_dir / "processed"
    if processed_dir.exists():
        for csv_file in processed_dir.glob("*.csv"):
            datasets[f"processed/{csv_file.name}"] = csv_file
    
    return datasets

def load_model_artifacts(
    models_dir: Path,
    model_name: str
) -> Tuple[Optional[object], Optional[object], Optional[dict]]:
    """Load model, vectorizer and label map from the models directory."""
    try:
        import joblib
        
        model_path = models_dir / f"{model_name}_model.joblib"
        vectorizer_path = models_dir / f"{model_name}_vectorizer.joblib"
        label_map_path = models_dir / f"{model_name}_label_map.json"
        
        if not all(p.exists() for p in [model_path, vectorizer_path, label_map_path]):
            return None, None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        with open(label_map_path) as f:
            label_map = json.load(f)
        
        return model, vectorizer, label_map
    
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def save_model_artifacts(
    models_dir: Path,
    model_name: str,
    model: object,
    vectorizer: object,
    label_map: dict
) -> bool:
    """Save model artifacts to the models directory."""
    try:
        import joblib
        
        # Create models directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(model, models_dir / f"{model_name}_model.joblib")
        joblib.dump(vectorizer, models_dir / f"{model_name}_vectorizer.joblib")
        
        # Save label map
        with open(models_dir / f"{model_name}_label_map.json", 'w') as f:
            json.dump(label_map, f)
        
        return True
    
    except Exception as e:
        print(f"Error saving model artifacts: {str(e)}")
        return False

def get_dataset_columns(dataset_path: Path) -> List[str]:
    """Get column names from a CSV file."""
    try:
        return list(pd.read_csv(dataset_path, nrows=0).columns)
    except Exception:
        return []

def analyze_threshold_performance(y_true, y_prob, thresholds=None):
    """Analyze model performance across different thresholds."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        results.append({
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        })
    
    return pd.DataFrame(results)

def save_threshold_analysis(df: pd.DataFrame, output_path: Path):
    """Save threshold analysis results to CSV."""
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving threshold analysis: {str(e)}")
        return False