
import os
import shutil

# Configuration
data_folder_name = "raw/TestData"  # Specify the initial folder name here
destination_dir = "raw"  # Destination for sorted files

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

def process_dataset(source_dir):
    """
    Processes the dataset by handling Period folders and organizing files into data/raw/.
    """
    # Traverse the main data folder
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            if dir_name.startswith("Period"):  # Check for Period folders
                period_path = os.path.join(root, dir_name)
                process_period_folder(period_path)

def process_period_folder(period_path):
    """
    Handles each Period folder, checking for duplicate structures and organizing files.
    """
    # Get the period name (e.g., Period11)
    period_name = os.path.basename(period_path)

    for root, dirs, files in os.walk(period_path):
        for stock in ["A", "B", "C", "D", "E"]:  # Iterate over stocks
            stock_path = os.path.join(root, stock)
            if os.path.exists(stock_path):
                copy_stock_files(stock_path, period_name)

def copy_stock_files(stock_path, period_name):
    """
    Copies market_data and trade_data files to the destination folder.
    """
    stock_name = os.path.basename(stock_path)

    for file_name in os.listdir(stock_path):
        if file_name.startswith("market_data") or file_name.startswith("trade_data"):
            # Define destination folder structure
            dest_folder = os.path.join(destination_dir, period_name, stock_name)
            os.makedirs(dest_folder, exist_ok=True)

            # Copy the file
            source_file = os.path.join(stock_path, file_name)
            destination_file = os.path.join(dest_folder, file_name)
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} -> {destination_file}")

if __name__ == "__main__":
    # Specify the folder name containing the raw data
    source_folder = data_folder_name

    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
    else:
        print(f"Processing dataset from '{source_folder}'...")
        process_dataset(source_folder)
        print("Dataset processing completed.")