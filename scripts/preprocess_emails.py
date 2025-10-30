"""scripts/preprocess_emails.py

Preprocess SMS / email datasets into a cleaned CSV suitable for training.

Features:
- Load CSV with at least `label` and `text` columns
- Clean text to `text_clean` with rules:
  * lowercase
  * remove URLs
  * replace phone numbers with <PHONE>
  * replace numbers with <NUM>
  * remove punctuation / strange characters
  * collapse whitespace
- Optionally save a steps CSV containing before/after for inspection
- Writes cleaned CSV to the specified output path

The script is executable: `python scripts/preprocess_emails.py --input ... --output ...`
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
import pandas as pd
from typing import Optional


PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DIGIT_RE = re.compile(r"\b\d+\b")
CONTROL_RE = re.compile(r"[\t\r\x0b\x0c\x0e-\x1f]")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame. Raises FileNotFoundError on missing path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    # Read with explicit column names for SMS spam format
    df = pd.read_csv(path, names=['label', 'text'], encoding='utf-8')
    df['label'] = df['label'].str.strip('"').str.lower()  # Clean label column
    df['text'] = df['text'].str.strip('"')  # Clean text column
    return df


def clean_text(s: Optional[str]) -> str:
    """Apply deterministic cleaning rules to a single text string.

    Rules implemented:
    - convert to lowercase
    - replace URLs with a single space
    - replace phone-like sequences with the token <PHONE>
    - replace standalone numbers with the token <NUM>
    - remove control characters and weird whitespace
    - remove punctuation (preserving token markers like <NUM> and <PHONE>)
    - collapse multiple spaces
    """
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    # remove URLs first
    s = URL_RE.sub(" ", s)
    # normalize control characters (tabs, etc.)
    s = CONTROL_RE.sub(" ", s)
    # replace phone numbers with token
    s = PHONE_RE.sub(" <PHONE> ", s)
    # replace standalone numbers with token
    s = DIGIT_RE.sub(" <NUM> ", s)
    # remove characters that are not a-z, 0-9, space, or angle brackets used for tokens
    s = re.sub(r"[^a-z0-9\s<>]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_df(df: pd.DataFrame, keep_original: bool = True) -> pd.DataFrame:
    """Return a new DataFrame with a `text_clean` column.

    If keep_original is True, keeps the original `text` column and adds `text_clean`.
    Otherwise `text` will be replaced by `text_clean`.
    """
    if 'text' not in df.columns:
        raise KeyError("Input DataFrame must contain a 'text' column")

    out = df.copy()
    out['text_clean'] = out['text'].apply(clean_text)
    if not keep_original:
        out = out.drop(columns=['text']).rename(columns={'text_clean': 'text'})
    return out


def save_steps(original: pd.DataFrame, processed: pd.DataFrame, steps_dir: str) -> str:
    """Save a CSV comparing original text and cleaned text. Returns path written."""
    os.makedirs(steps_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_path = os.path.join(steps_dir, f'preprocess_steps_{ts}.csv')
    compare = original.copy()
    compare['text_clean'] = processed['text_clean'].astype(str)
    compare.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Preprocess SMS / email CSV into cleaned text')
    parser.add_argument('--input', '-i', required=True, help='Path to input raw CSV')
    parser.add_argument('--output', '-o', required=True, help='Path to output processed CSV')
    parser.add_argument('--keep-original', dest='keep_original', action='store_true',
                        help='Keep original `text` column and add `text_clean` (default)')
    parser.add_argument('--no-keep-original', dest='keep_original', action='store_false',
                        help='Replace `text` with cleaned text (drop original)')
    parser.set_defaults(keep_original=True)
    parser.add_argument('--save-steps', '-s', nargs='?', const='datasets/processed/steps',
                        help='If provided, save a before/after CSV to the given folder (default: datasets/processed/steps)')

    args = parser.parse_args()

    try:
        df = load_csv(args.input)
    except Exception as e:
        print(f'Error loading input: {e}')
        return 2

    try:
        processed = preprocess_df(df, keep_original=args.keep_original)
    except Exception as e:
        print(f'Error during preprocessing: {e}')
        return 3

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)

    try:
        processed.to_csv(args.output, index=False)
    except Exception as e:
        print(f'Error writing output file: {e}')
        return 4

    printed = min(5, len(processed))
    print(f'Processed {len(processed)} rows. Wrote cleaned CSV to: {args.output}')
    if printed:
        print('\nPreview (first {n} rows):'.format(n=printed))
        print(processed.head(printed).to_string(index=False))

    if args.save_steps:
        steps_dir = args.save_steps or 'datasets/processed/steps'
        try:
            saved = save_steps(df, processed, steps_dir)
            print(f'Saved before/after steps CSV to: {saved}')
        except Exception as e:
            print(f'Error saving steps CSV: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())