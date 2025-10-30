"""visualize.py

Basic visualization helpers for training results and data exploration.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_label_distribution(df: pd.DataFrame):
    counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title('Label distribution')
    ax.set_xlabel('label')
    ax.set_ylabel('count')
    return fig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    fig = plot_label_distribution(df)
    fig.savefig('reports/visualizations/label_distribution.png')
    print('Saved visualization to reports/visualizations/label_distribution.png')