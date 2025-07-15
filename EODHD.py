import requests
from datetime import datetime, timezone
import pandas as pd
import os
import concurrent.futures
import shutil


def download_stock(stock_symbol: str, idx: int, total: int, initial_date_utc: int, final_date_utc: int, reverse: bool=True, max_missed_steps: int = 5):
    # Check available disk space before starting
    total_b, used_b, free_b = shutil.disk_usage(data_folder)
    free_gb = free_b / (1024 ** 3)
    if free_gb < 10:
        print(f"Skipping {stock_symbol}: Less than 10GB free ({free_gb:.2f}GB left).")
        return

    print(f"Downloading {stock_symbol} information, {idx+1}/{total}")

    all_data = []
    error_count = 0

    if reverse:
        to_date = final_date_utc
        while to_date > initial_date_utc:
            from_date = max(to_date - time_interval, initial_date_utc)
            print(f"{stock_symbol} {idx+1}/{total} From {datetime.fromtimestamp(from_date, tz=timezone.utc)} to {datetime.fromtimestamp(to_date, tz=timezone.utc)}. Timestamp: {datetime.now()}")
            url = f'https://eodhd.com/api/intraday/{stock_symbol}.US?interval=1m&api_token={API_KEY}&fmt=json&from={from_date}&to={to_date}'
            response = requests.get(url)
            if response.status_code == 200 and response.text.strip():
                try:
                    data = response.json()
                    if isinstance(data, list) and data:
                        all_data = data + all_data
                        error_count = 0
                    else:
                        print(f"No data received for {stock_symbol} from {from_date} to {to_date}.")
                        error_count += 1
                        if error_count > max_missed_steps:
                            print(f"Too many steps no data found for {stock_symbol}. Stopping download.")
                            break
                except Exception as e:
                    print(f"Error decoding JSON: {e}")
                    break
            else:
                print(f"Empty or bad response for {from_date} to {to_date}. Stopping loop.")
                break
            to_date = from_date
    else:
        from_date = initial_date_utc
        while from_date < final_date_utc:
            to_date = min(from_date + time_interval, final_date_utc)
            print(f"{stock_symbol} {idx+1}/{total} From {datetime.fromtimestamp(from_date, tz=timezone.utc)} to {datetime.fromtimestamp(to_date, tz=timezone.utc)}. Timestamp: {datetime.now()}")
            url = f'https://eodhd.com/api/intraday/{stock_symbol}.US?interval=1m&api_token={API_KEY}&fmt=json&from={from_date}&to={to_date}'
            response = requests.get(url)
            if response.status_code == 200 and response.text.strip():
                try:
                    data = response.json()
                    if isinstance(data, list) and data:
                        all_data += data
                        error_count = 0
                    else:
                        print(f"No data received for {stock_symbol} from {from_date} to {to_date}.")
                        error_count += 1
                        if error_count > max_missed_steps:
                            print(f"Too many steps no data found for {stock_symbol}. Stopping download.")
                            break
                except Exception as e:
                    print(f"Error decoding JSON: {e}")
                    break
            else:
                print(f"Empty or bad response for {from_date} to {to_date}. Stopping loop.")
                break
            from_date = to_date

    # Save data to CSV 
    if all_data and isinstance(all_data, list):
        keys = all_data[0].keys()
        filename = f'{data_folder}/{stock_symbol}_intraday_1m.csv'
        new_df = pd.DataFrame(all_data)
        if 'datetime' in new_df.columns:
            new_df['datetime'] = pd.to_datetime(new_df['datetime'], utc=True)
            new_df = new_df.sort_values('datetime')
        else:
            print(f"No 'datetime' column in new data for {stock_symbol}.")
            return

        if os.path.exists(filename):
            old_df = pd.read_csv(filename, parse_dates=['datetime'])
            if 'datetime' in old_df.columns:
                old_df['datetime'] = pd.to_datetime(old_df['datetime'], utc=True)
                # Concatenate new data on top (before old data)
                combined = pd.concat([new_df, old_df], ignore_index=True)
                # Drop duplicates, keeping the first (newest) occurrence
                combined = combined.drop_duplicates(subset='datetime', keep='first')
                combined = combined.sort_values('datetime')
            else:
                combined = new_df
        else:
            combined = new_df

        # Save back to CSV
        combined.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")



API_KEY = "68303e5f30cb46.64268753" #Insert your EODHD API Key
data_folder = './EODHD_Data' # Path to your data folder
tickers_to_download = ['AAPL','TSLA','MSFT'] #List of stocks to download
initial_date_utc = int(datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
final_date_utc = int(datetime.strptime("2025-05-23 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
time_interval = 120*24*60*60 # In seconds. This equivalents to 120 days, max allowed by EODHD
reverse = True # Set to true to start backwards from newest to oldest, useful since some stocks might be newer than your initial date
max_workers = 20 # Parallel execution,set 1 to prevent parallelization
max_missed_steps = 5 #set how many time intervals can return no data before stopping the stock download e.g: 5 means 5 intervals of 120 days. 

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(download_stock, stock_symbol, idx, len(tickers_to_download), initial_date_utc, final_date_utc, reverse=False, max_missed_steps = 5)
        for idx, (stock_symbol) in enumerate(tickers_to_download)
    ]
    concurrent.futures.wait(futures)
