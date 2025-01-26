import pandas as pd

def resample_data(df, time_interval_str="3s"):
    # Convert timestamp to datetime index
    df = df.set_index('timestamp')

    # Resample to 100ms intervals (adjust as needed)
    resampled_df = df.resample(time_interval_str).agg({
        'bidVolume': 'mean',
        'bidPrice': 'mean',
        'askVolume': 'mean',
        'askPrice': 'mean',
        'volume': 'sum',  # Total volume in the interval
        'price': 'last',  # Closing price of the interval
        'label': 'last'   # Use last label in the interval
    }).dropna().reset_index()
    return resampled_df
