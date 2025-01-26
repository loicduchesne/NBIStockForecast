
import os
import pandas as pd

def get_df_period():
    pass

def get_df_market_data(base_path_to_period):
    stocks = ["A", "B", "C", "D", "E"]  # List of stocks to process
    stock_dataframes = {}  # Dictionary to store combined DataFrames for each stock

    for stock in stocks:
        stock_folder = os.path.join(base_path_to_period, stock)
        combined_data = []

        if os.path.exists(stock_folder):
            # Look for all market_data files in the stock folder
            for file_name in os.listdir(stock_folder):
                if file_name.startswith("market_data"):
                    file_path = os.path.join(stock_folder, file_name)

                    # Attempt to load the file with default or custom headers
                    try:
                        if not "_0" in file_name:  # Example for files requiring custom headers
                            data = pd.read_csv(
                                file_path,
                                names=["bidVolume", "bidPrice", "askVolume", "askPrice", "timestamp"],
                            )
                        else:
                            data = pd.read_csv(file_path)  # Default behavior
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    # Append the data to the combined list
                    combined_data.append(data)

            # Combine all data into a single DataFrame for the current stock
            if combined_data:
                combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
                stock_dataframes[stock] = combined_df
            else:
                print(f"No market_data files found for stock {stock} in {stock_folder}.")
                stock_dataframes[stock] = pd.DataFrame()  # Empty DataFrame for missing data
        else:
            print(f"Stock folder does not exist: {stock_folder}")
            stock_dataframes[stock] = pd.DataFrame()  # Empty DataFrame for missing folder

    return stock_dataframes



def get_df_resized_trade_data(base_path_to_period, combined_market_data):
    stocks = ["A", "B", "C", "D", "E"]  # List of stocks to process
    stock_dataframes = {}  # Dictionary to store combined DataFrames for each stock

    for stock in stocks:
        stock_folder = os.path.join(base_path_to_period, stock)
        if os.path.exists(stock_folder):
            for file_name in os.listdir(stock_folder):
                if file_name.startswith("trade_data"):
                    trade_data_path = os.path.join(stock_folder, file_name)
                    trade_df = pd.read_csv(trade_data_path)
                    trade_df['timestamp'] = (
                    trade_df['timestamp']
                    .str.extract(r'(\d+:\d+:\d+\.\d{6})', expand=False)  # Keep first 6 decimal digits
                    .apply(pd.to_datetime, format='%H:%M:%S.%f')
                    )
                    market_df = combined_market_data[stock]
                    # truncate nanoseconds to microseconds
                    market_df['timestamp'] = (
                        market_df['timestamp']
                        .str.extract(r'(\d+:\d+:\d+\.\d{6})')
                        .apply(pd.to_datetime, format='%H:%M:%S.%f')
                    )
                    market_df = market_df.drop(0)

                    # combine the two dfs together horizontally
                    trade_df = trade_df.sort_values('timestamp')
                    market_df = market_df.sort_values('timestamp')

                    # Perform an asof merge to align trade data with market data timestamps
                    merged_df = pd.merge_asof(market_df, trade_df, on='timestamp', direction='nearest')

                    # Optional: Rename columns for clarity if needed
                    merged_df.rename(columns={
                        'timestamp_x': 'timestamp',  # This is already the case, as merge_asof keeps left timestamp
                        'timestamp_y': 'trade_timestamp'
                    }, inplace=True, errors='ignore')

                    stock_dataframes[stock] = merged_df
        else:
            print(f"Stock folder does not exist: {stock_folder}")
            stock_dataframes[stock] = pd.DataFrame()
    return stock_dataframes


if __name__ == "__main__":
    base_path = "raw/Period1"  # Path to the folder containing stock subfolders (A, B, C, D, E)
    combined_data = get_df_market_data(base_path)
    result = get_df_resized_trade_data(base_path, combined_data)
    print(combined_data["D"].head())
    print(len(combined_data["D"]))
    file_path1 = "raw/Period1/D/market_data_D_1.csv"
    #file_path2 = "market_data_A_1.csv"
    df = pd.read_csv(file_path1)

    # Get the number of rows
    num_rows = df.shape[0]

    print(f"Number of rows: {num_rows}")


