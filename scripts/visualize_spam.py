"""Generate visualizations and analysis for spam classification model.

Creates:
- Class distribution plot
- Top tokens per class
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Threshold sweep analysis

Examples:
    # Generate all visualizations:
    python scripts/visualize_spam.py --input datasets/processed/sms_spam_processed.csv 
                                   --model models/logistic_model.joblib 
                                   --vector models/tfidf_vectorizer.joblib 
                                   --top-n 20

    # Only generate specific plots:
    python scripts/visualize_spam.py --input datasets/processed/sms_spam_processed.csv 
                                   --plots class-dist tokens roc
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, average_precision_score, f1_score
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Default paths
DEFAULT_INPUT = Path('datasets') / 'processed' / 'sms_spam_processed.csv'
DEFAULT_MODEL = Path('models') / 'logistic_model.joblib'
DEFAULT_VECTOR = Path('models') / 'tfidf_vectorizer.joblib'
OUTPUT_DIR = Path('reports') / 'visualizations'

# Plot styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
COLORS = {'spam': '#FF6B6B', 'ham': '#4ECDC4'}


def setup_output_dir(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def load_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    required_cols = {'text', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    
    return df


def load_artifacts(model_path: Path, vector_path: Path) -> tuple[Any, Any]:
    """Load model and vectorizer."""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vector_path)
        return model, vectorizer
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot class distribution as a bar chart."""
    plt.figure(figsize=(10, 6))
    counts = df['label'].value_counts()
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette=[COLORS[x] for x in counts.index])
    
    # Add count labels on bars
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()


def plot_top_tokens(df: pd.DataFrame, vectorizer: Any, top_n: int, output_dir: Path) -> None:
    """Plot top tokens for each class."""
    feature_names = vectorizer.get_feature_names_out()
    
    plt.figure(figsize=(15, 8))
    
    for label in ['spam', 'ham']:
        # Get texts for this class
        texts = df[df['label'] == label]['text']
        
        # Transform and get mean feature values
        X = vectorizer.transform(texts)
        mean_tfidf = X.mean(axis=0).A1
        
        # Get top tokens
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_values = mean_tfidf[top_indices]
        top_terms = feature_names[top_indices]
        
        # Plot
        plt.subplot(1, 2, 1 if label == 'spam' else 2)
        plt.barh(range(top_n), top_values, color=COLORS[label])
        plt.yticks(range(top_n), top_terms)
        plt.title(f'Top {top_n} Tokens in {label.upper()} Messages')
        
        if label == 'spam':
            plt.xlabel('Mean TF-IDF Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_tokens.png')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()


def analyze_thresholds(y_true: np.ndarray, y_prob: np.ndarray, 
                      output_dir: Path, n_thresholds: int = 100) -> None:
    """Analyze model performance across different thresholds."""
    thresholds = np.linspace(0, 1, n_thresholds)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = (y_pred & y_true).sum() / max(y_pred.sum(), 1)
        recall = (y_pred & y_true).sum() / max(y_true.sum(), 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    # Create DataFrame and save
    df_thresholds = pd.DataFrame(results)
    df_thresholds.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(df_thresholds['threshold'], df_thresholds['precision'],
             label='Precision', color='blue')
    plt.plot(df_thresholds['threshold'], df_thresholds['recall'],
             label='Recall', color='red')
    plt.plot(df_thresholds['threshold'], df_thresholds['f1_score'],
             label='F1 Score', color='green')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Model Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_sweep.png')
    plt.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Generate visualizations for spam classification analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                       help='Path to processed dataset CSV')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                       help='Path to trained model')
    parser.add_argument('--vector', type=Path, default=DEFAULT_VECTOR,
                       help='Path to fitted vectorizer')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                       help='Directory to save visualizations')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top tokens to show per class')
    parser.add_argument('--plots', nargs='+', default=['all'],
                       choices=['all', 'class-dist', 'tokens', 'confusion',
                              'roc', 'pr', 'threshold'],
                       help='Which plots to generate')

    args = parser.parse_args(argv)
    
    try:
        # Create output directory
        setup_output_dir(args.output_dir)
        
        # Load dataset
        print(f"Loading dataset from {args.input}...")
        df = load_data(args.input)
        
        # Load model artifacts
        print(f"Loading model artifacts...")
        model, vectorizer = load_artifacts(args.model, args.vector)
        
        # Determine which plots to generate
        plots = set(args.plots)
        if 'all' in plots:
            plots = {'class-dist', 'tokens', 'confusion', 'roc', 'pr', 'threshold'}
        
        # Generate visualizations
        if 'class-dist' in plots:
            print("Generating class distribution plot...")
            plot_class_distribution(df, args.output_dir)
        
        if 'tokens' in plots:
            print("Generating top tokens plot...")
            plot_top_tokens(df, vectorizer, args.top_n, args.output_dir)
        
        # For model evaluation plots, we need predictions
        if {'confusion', 'roc', 'pr', 'threshold'} & plots:
            print("Generating model predictions...")
            X = vectorizer.transform(df['text'])
            y_true = (df['label'] == 'spam').astype(int)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            if 'confusion' in plots:
                print("Generating confusion matrix...")
                plot_confusion_matrix(y_true, y_pred, args.output_dir)
            
            if 'roc' in plots:
                print("Generating ROC curve...")
                plot_roc_curve(y_true, y_prob, args.output_dir)
            
            if 'pr' in plots:
                print("Generating Precision-Recall curve...")
                plot_pr_curve(y_true, y_prob, args.output_dir)
            
            if 'threshold' in plots:
                print("Generating threshold analysis...")
                analyze_thresholds(y_true, y_prob, args.output_dir)
        
        print(f"\nVisualizations saved to {args.output_dir}")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())