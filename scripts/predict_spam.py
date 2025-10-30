"""Predict spam/ham for single messages or batch CSVs.

Supports:
- Single text prediction via --text
- Batch prediction via --input (CSV with a `text` column) and --output

Examples:
    python scripts\predict_spam.py --model models\logistic_model.joblib --vector models\tfidf_vectorizer.joblib --text "Free money!!!"

    python scripts\predict_spam.py --model models\logistic_model.joblib --vector models\tfidf_vectorizer.joblib --input datasets\processed\sms_spam_processed.csv --output predictions.csv

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# Default paths (using Path for cross-platform compatibility)
DEFAULT_MODEL = Path('models') / 'logistic_model.joblib'
DEFAULT_VECTOR = Path('models') / 'tfidf_vectorizer.joblib'
SAMPLE_DATA = Path('datasets') / 'processed' / 'sms_spam_processed.csv'


def ensure_model_exists(model_path: Path, vector_path: Path) -> Tuple[Path, Path]:
    """Ensure model and vectorizer exist, training if needed."""
    if model_path.exists() and (vector_path is None or vector_path.exists()):
        return model_path, vector_path

    print("\nModel or vectorizer not found. Training new model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run training script
    train_script = Path('scripts') / 'train_spam_classifier.py'
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    cmd = [
        sys.executable,
        str(train_script),
        '--input', str(SAMPLE_DATA),
        '--model-path', str(model_path),
        '--vector-path', str(vector_path) if vector_path else str(DEFAULT_VECTOR)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during training:")
        print(result.stderr)
        raise RuntimeError("Model training failed")

    print("Model training completed.")
    return model_path, vector_path


def load_artifacts(model_path: Path, vector_path: Optional[Path] = None) -> Tuple[object, Optional[object]]:
    """Load model and optional vectorizer, ensuring they exist first."""
    model_path, vector_path = ensure_model_exists(model_path, vector_path)
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vector_path) if vector_path else None
        return model, vectorizer
    except Exception as e:
        raise RuntimeError(f"Error loading artifacts: {e}")


def predict_single(model, vectorizer, text: str) -> Tuple[str, float]:
    """Predict single text, returning (label, probability)."""
    t = str(text)
    if vectorizer is not None:
        X = vectorizer.transform([t])
    else:
        X = [t]  # model expected to handle raw text

    try:
        pred = model.predict(X if vectorizer else [t])[0]
        prob = model.predict_proba(X if vectorizer else [t])[0][1] if hasattr(model, 'predict_proba') else None
        label = 'spam' if int(pred) == 1 else 'ham'
        return label, prob
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


def predict_batch(model, vectorizer, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a DataFrame with a text column."""
    if 'text' not in df.columns:
        raise KeyError("Input CSV must contain a 'text' column")

    texts = df['text'].astype(str).tolist()
    
    try:
        if vectorizer is not None:
            X = vectorizer.transform(texts)
        else:
            X = texts

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

        df_out = df.copy()
        df_out['pred_label'] = ['spam' if int(p) == 1 else 'ham' for p in preds]
        if probs is not None:
            df_out['pred_prob'] = probs.astype(float)
        return df_out

    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {e}")


def format_prediction(label: str, prob: Optional[float], threshold: float = 0.5) -> str:
    """Format prediction results for display."""
    result = f"Prediction: {label}"
    if prob is not None:
        result += f" | spam-prob = {prob:.4f} (threshold = {threshold:.2f})"
    return result


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Predict spam/ham for single text or CSV file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', '-m', type=Path, default=DEFAULT_MODEL,
                        help='Path to saved model (will auto-train if not found)')
    parser.add_argument('--vector', '-v', type=Path, default=DEFAULT_VECTOR,
                        help='Path to saved TF-IDF vectorizer (optional if model is a pipeline)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='Single text to classify')
    group.add_argument('--input', type=Path, help='Input CSV file with a text column')
    
    parser.add_argument('--output', '-o', type=Path, help='Output CSV for batch predictions')
    parser.add_argument('--sample', type=int, default=5,
                        help='Number of example rows to print')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for spam classification')

    args = parser.parse_args(argv)

    try:
        # Load or train model
        model, vectorizer = load_artifacts(args.model, args.vector)

        # Single text prediction
        if args.text:
            label, prob = predict_single(model, vectorizer, args.text)
            print('\nInput text:', args.text)
            print(format_prediction(label, prob, args.threshold))

        # Batch prediction
        else:
            if not args.input.exists():
                print(f"Input file not found: {args.input}")
                return 1

            print(f"\nProcessing input file: {args.input}")
            df = pd.read_csv(args.input)
            
            out = predict_batch(model, vectorizer, df)
            
            # Auto-generate output path if not specified
            out_path = args.output
            if not out_path:
                stem = args.input.stem
                out_path = Path(f"{stem}_predictions.csv")

            # Ensure output directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            print(f'Wrote predictions to: {out_path}')

            # Print sample predictions
            if args.sample > 0:
                n = min(args.sample, len(out))
                print(f'\nSample predictions (first {n} rows):')
                sample = out[['text', 'pred_label', 'pred_prob']].head(n)
                print(sample.to_string(index=False))

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
