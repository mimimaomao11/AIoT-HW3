"""Train a LogisticRegression spam classifier using TF-IDF features.

Usage example:
    python scripts/train_spam_classifier.py \
        --input datasets/processed/sms_spam_processed.csv \
        --model-path models/logistic_model.joblib \
        --vector-path models/tfidf_vectorizer.joblib

The script will print accuracy, precision, recall, f1 and a confusion matrix.
It will save the trained model pipeline and the fitted vectorizer.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def prepare_labels(df: pd.DataFrame, label_col: str = 'label') -> pd.Series:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in data")
    # Map ham/spam to 0/1 if needed
    labels = df[label_col].astype(str).map(lambda x: 1 if x.lower() in ('spam','1','true','t') else 0)
    return labels


def train(args: argparse.Namespace) -> int:
    try:
        df = load_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 2

    if 'text_clean' in df.columns:
        texts = df['text_clean'].astype(str)
    elif 'text' in df.columns:
        texts = df['text'].astype(str)
    else:
        print("No 'text' or 'text_clean' column found in input data")
        return 3

    try:
        y = prepare_labels(df, label_col=args.label_col)
    except Exception as e:
        print(f"Error preparing labels: {e}")
        return 4

    # Basic checks
    if len(texts) < 2:
        print("Not enough data to train (need at least 2 samples).")
        return 5

    n_classes = int(y.nunique())
    if n_classes < 2:
        print("Training requires at least two classes (spam and ham) in the input labels. Found only one class.")
        return 6

    # Split (attempt stratified split when possible)
    test_size = args.test_size
    random_state = args.random_state

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception:
        # Fall back to non-stratified split if stratify fails for small sample counts
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state
        )

    # If the split produced a set with only one class (possible with tiny datasets),
    # fall back to training on the full dataset and skip evaluation.
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Warning: train/test split resulted in a set with only one class.\n"
              "Falling back to train on full dataset and skipping evaluation.")
        X_train = texts
        y_train = y
        X_test = None
        y_test = None

    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=args.max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Class weight support
    class_weight = None if args.class_weight == 'none' else args.class_weight

    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
    try:
        clf.fit(X_train_tfidf, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        return 6

    preds = clf.predict(X_test_tfidf)
    probs = None
    if hasattr(clf, 'predict_proba'):
        try:
            probs = clf.predict_proba(X_test_tfidf)[:, 1]
        except Exception:
            probs = None

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print('\n=== Evaluation ===')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-score:  {f1:.4f}\n')

    print('Classification report:')
    print(classification_report(y_test, preds, target_names=['ham','spam'], zero_division=0))

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=['ham','spam'], columns=['pred_ham','pred_spam'])
    print('Confusion matrix:')
    print(cm_df.to_string())

    # Save artifacts
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.vector_path) or '.', exist_ok=True)

    try:
        # Save model (we'll save the classifier and vectorizer separately)
        joblib.dump(clf, args.model_path)
        joblib.dump(vectorizer, args.vector_path)
        print(f'\nSaved model to: {args.model_path}')
        print(f'Saved vectorizer to: {args.vector_path}')
    except Exception as e:
        print(f'Error saving artifacts: {e}')
        return 7

    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a LogisticRegression spam classifier')
    parser.add_argument('--input', '-i', required=True, help='Path to processed CSV (expects text_clean or text column)')
    parser.add_argument('--label-col', default='label', help='Column name for labels (default: label)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for train/test split')
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced',
                        help="Class weight passed to LogisticRegression. Use 'none' for no class weighting.")
    parser.add_argument('--max-features', type=int, default=10000, help='Max features for TF-IDF vectorizer')
    parser.add_argument('--model-path', default='models/logistic_model.joblib', help='Output path for trained model')
    parser.add_argument('--vector-path', default='models/tfidf_vectorizer.joblib', help='Output path for fitted TF-IDF vectorizer')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    rc = train(args)
    sys.exit(rc)
