import pandas as pd
import os
import glob
import datetime

def csv_to_hdf5(filepath, stock_name, idx=None, total=None):
    output_path = os.path.join(output_folder, f"{stock_name}.h5")
    skip = False

    if os.path.exists(output_path):
        # Get first row from CSV
        csv_first_row = pd.read_csv(filepath, nrows=1, parse_dates=['datetime'])
        csv_first_ts = pd.to_datetime(csv_first_row['datetime'].iloc[0], utc=True)
        csv_first_row.set_index('datetime', inplace=True)
        csv_first_row = csv_first_row[['open', 'high', 'low', 'close', 'volume']]

        try:
            # Read only the row at the CSV's first timestamp directly using .loc
            h5_row = pd.read_hdf(output_path, key='data', columns=['open', 'high', 'low', 'close', 'volume'])
            prev_ts = csv_first_ts - pd.Timedelta(minutes=1)
            if csv_first_ts in h5_row.index:
                row = h5_row.loc[csv_first_ts][['open', 'high', 'low', 'close', 'volume']]
                print(f"{stock_name}: Checking if first CSV row exists and matches in HDF5 at {csv_first_ts}")
                print(f"  HDF5 @ {csv_first_ts}: open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}, volume={row['volume']}")
                if prev_ts in h5_row.index:
                    prev_row = h5_row.loc[prev_ts][['open', 'high', 'low', 'close', 'volume']]
                    print(f"  HDF5 @ {prev_ts}: open={prev_row['open']}, high={prev_row['high']}, low={prev_row['low']}, close={prev_row['close']}, volume={prev_row['volume']}")
                else:
                    print(f"  HDF5 @ {prev_ts}: (not found)")
                print(f"  CSV  @ {csv_first_ts}: open={csv_first_row.iloc[0]['open']}, high={csv_first_row.iloc[0]['high']}, low={csv_first_row.iloc[0]['low']}, close={csv_first_row.iloc[0]['close']}, volume={csv_first_row.iloc[0]['volume']}")
                if row.equals(csv_first_row.iloc[0]):
                    print(f"Skipping {stock_name}: first CSV row matches HDF5 row at {csv_first_ts}.")
                    skip = True
                else:
                    print(f"Removing existing HDF5 file for {stock_name}: first CSV row does not match HDF5.")
                    os.remove(output_path)
            else:
                print(f"Removing existing HDF5 file for {stock_name}: timestamp {csv_first_ts} not found in HDF5.")
                os.remove(output_path)
        except Exception as e:
            print(f"Error reading {output_path}, removing file. Reason: {e}")
            os.remove(output_path)

    if skip:
        return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if idx is not None and total is not None:
        print(f"[{now}] Processing {stock_name}... ({idx+1}/{total})")
    else:
        print(f"[{now}] Processing {stock_name}...")

    # Read the CSV and save to HDF5
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index()
    # Create master minute-level datetime index
    master_index = pd.date_range(start=start_time, end=end_time, freq="min")
    output_path = os.path.join(output_folder, f"{stock_name}.h5")
    df = df[~df.index.duplicated(keep='first')]
    df_aligned = df.reindex(master_index)
    df_aligned.to_hdf(
        output_path,
        key='data',
        mode='w',
        format='table',
        data_columns=True,
        complevel=9,
        complib='blosc'
    )
    print(f"Saved: {output_path}")


# === Configuration ===
csv_folder = "EODHD_Data"        # <-- Set your CSV directory here
output_folder = "HDF5_output"    # Where to save HDF5 files
start_time = pd.Timestamp("2004-01-01 00:00:00", tz="UTC")
end_time = pd.Timestamp("2025-05-24 00:00:00", tz="UTC")  # Adjust if needed

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
csv_files = [
    f for f in glob.glob(os.path.join(csv_folder, "*.csv"))
]

for idx, f in enumerate(csv_files):
    stock_name = str(os.path.basename(f)).split("_")[0] #Our files are in format XXXX_intraday_1m.csv
    df = csv_to_hdf5(f, stock_name, idx, len(csv_files))
