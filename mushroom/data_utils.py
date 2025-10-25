import pandas as pd
import numpy as np


def get_clean_X_y(csv_path='mushrooms_clean.csv', source_path='mushrooms.csv'):
    """Return (X, y) where X is the hot-encoded feature matrix and y is the label series.
    """
    try:
        df = pd.read_csv(csv_path)
        # assume last column is 'class' or has a 'class' column
        if 'class' in df.columns:
            y = df['class']
            X = df.drop(columns=['class'])
        else:
            # fallback: assume last column is label
            # all rows only first column
            y = df.iloc[:, -1]
            # all rows, all columns except 1st which is y
            X = df.iloc[:, :-1]
        return X, y
    except FileNotFoundError:
        raise FileNotFoundError("Rerun data cleaning files to obtain clean split Dataset.")
