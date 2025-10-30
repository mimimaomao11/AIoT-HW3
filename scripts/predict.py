"""predict.py

Load a saved model and run predictions on new texts.
"""

import joblib
import argparse
import pandas as pd


def predict_texts(model_path: str, texts: list):
    model = joblib.load(model_path)
    preds = model.predict(texts)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(texts)[:, 1]
    return preds, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, help='CSV with text column')
    parser.add_argument('--output', required=False, help='Write predictions to CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds, probs = predict_texts(args.model, df['text'].astype(str).tolist())
    df['pred'] = preds
    if probs is not None:
        df['prob_spam'] = probs
    if args.output:
        df.to_csv(args.output, index=False)
        print(f'Wrote predictions to {args.output}')
    else:
        print(df.head())