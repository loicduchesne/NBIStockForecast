import numpy as np
import pandas as pd


def add_label(df, horizon=100, up_threshold=0.001, down_threshold=-0.001):
    # Sort data by timestamp to ensure chronological order
    df = df.sort_values('timestamp')

    # Calculate future price based on horizon
    df['future_price'] = df['price'].shift(-horizon)

    # Calculate percentage change
    df['price_change'] = (df['future_price'] - df['price']) / df['price'] * 100

    # Assign labels
    df['label'] = np.where(
        df['price_change'] > up_threshold,
        2,  # Up
        np.where(
            df['price_change'] < down_threshold,
            0,  # Down
            1   # Hold
        )
    )

    # Drop rows with NaN (last `horizon` rows)
    df = df.dropna(subset=['future_price'])
    return df