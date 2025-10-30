import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Add project root to path for importing scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_spam_classifier import train
from .utils import (
    find_csv_files,
    load_model_artifacts,
    save_model_artifacts,
    get_dataset_columns
)

# Constants for model artifact directories
MODELS_DIR = project_root / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Session state initialization
if "model_name" not in st.session_state:
    st.session_state.model_name = "default"

@st.cache_data
def get_available_datasets() -> dict:
    """Get mapping of dataset names to file paths."""
    datasets_dir = project_root / "datasets"
    return find_csv_files(datasets_dir)

def load_or_train_model(dataset_path: Path, 
                       text_col: str,
                       label_col: str,
                       test_size: float,
                       random_seed: int,
                       force_retrain: bool = False) -> tuple:
    """Load existing model or train a new one."""
    model_name = st.session_state.model_name
    
    if not force_retrain:
        # Try to load existing model
        model, vectorizer, label_map = load_model_artifacts(MODELS_DIR, model_name)
        if all(x is not None for x in [model, vectorizer, label_map]):
            return model, vectorizer, label_map
    
    # Train new model if loading failed or force_retrain is True
    df = pd.read_csv(dataset_path)
    
    # Create label mapping
    unique_labels = df[label_col].unique()
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col].map(label_map),
        test_size=test_size,
        random_state=random_seed,
        stratify=df[label_col]
    )
    
    # Create and train model
    args = argparse.Namespace(
        input=str(dataset_path),
        text_col=text_col,
        label_col=label_col,
        test_size=test_size,
        random_state=random_seed,
        class_weight='balanced',
        max_features=10000,
        model_path=str(MODELS_DIR / f"{model_name}_model.joblib"),
        vector_path=str(MODELS_DIR / f"{model_name}_vectorizer.joblib")
    )
    
    model, vectorizer = train(args)
    
    # Save artifacts
    save_model_artifacts(MODELS_DIR, model_name, model, vectorizer, label_map)
    
    return model, vectorizer, label_map

def sidebar_settings():
    """Render sidebar settings."""
    st.sidebar.title("Settings")
    
    # Dataset selection
    datasets = get_available_datasets()
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        key="dataset_name"
    )
    
    dataset_path = datasets[dataset_name]
    
    # Column selection
    columns = get_dataset_columns(dataset_path)
    
    text_col = st.sidebar.selectbox(
        "Text Column",
        options=columns,
        key="text_col"
    )
    
    label_col = st.sidebar.selectbox(
        "Label Column",
        options=columns,
        key="label_col"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    
    force_retrain = st.sidebar.checkbox(
        "Force Retrain Model",
        value=False,
        help="Train a new model even if one exists"
    )
    
    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key="test_size"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=999999,
        value=42,
        key="random_seed"
    )
    
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="threshold"
    )
    
    return dataset_path, text_col, label_col, test_size, random_seed, threshold, force_retrain