import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

def get_market_datetime_index(target_datetime_utc, start_time=None, end_time=None, start_hour="09:30", end_hour="16:00", time_zone="US/Eastern"):
    """
    Get the index position for a given UTC datetime in the market minutes index.
    
    Args:
        target_datetime_utc: pandas.Timestamp or datetime-like object in UTC
        start_time: Start date for market calendar (default: 2004-01-01)
        end_time: End date for market calendar (default: 2025-05-24)
        start_hour: Market open time (default: "09:30")
        end_hour: Market close time (default: "16:00")
        time_zone: Market timezone (default: "US/Eastern")
    
    Returns:
        int: Index position of the datetime in the market minutes index
        
    Raises:
        ValueError: If the datetime is not found in market hours
    """
    # Set default date range to match the merge file generation
    if start_time is None:
        start_time = pd.Timestamp("2004-01-01 00:00:00+00:00")
    if end_time is None:
        end_time = pd.Timestamp("2025-05-24 00:00:00+00:00")
    
    # Ensure target datetime is in UTC
    if isinstance(target_datetime_utc, str):
        target_datetime_utc = pd.Timestamp(target_datetime_utc, tz='UTC')
    elif target_datetime_utc.tz is None:
        target_datetime_utc = target_datetime_utc.tz_localize('UTC')
    elif target_datetime_utc.tz != 'UTC':
        target_datetime_utc = target_datetime_utc.tz_convert('UTC')
    
    # Generate the same market minutes index as used in merge file creation
    nyse = mcal.get_calendar('XNYS')
    schedule = nyse.schedule(start_date=start_time.date(), end_date=end_time.date())
    open_minutes = []
    
    for day in schedule.index:
        date_str = day.strftime('%Y-%m-%d')
        start_dt = pd.Timestamp(f"{date_str} {start_hour}", tz=time_zone)
        end_dt = pd.Timestamp(f"{date_str} {end_hour}", tz=time_zone)
        minutes = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(minutes=1), freq='1min')
        minutes = minutes.tz_convert('UTC')
        open_minutes.append(minutes)
    
    full_index = pd.DatetimeIndex(np.concatenate(open_minutes))
    
    # Find the exact datetime in the index
    try:
        index_position = full_index.get_loc(target_datetime_utc)
        return index_position
    except KeyError:
        # Check if it's close to a market minute (within 1 minute tolerance)
        closest_idx = full_index.get_indexer([target_datetime_utc], method='nearest')[0]
        if closest_idx >= 0 and closest_idx < len(full_index):
            closest_time = full_index[closest_idx]
            time_diff = abs((target_datetime_utc - closest_time).total_seconds())
            if time_diff <= 60:  # Within 1 minute
                return closest_idx
        
        # If not found, provide helpful error message
        target_date = target_datetime_utc.date()
        target_time = target_datetime_utc.time()
        
        # Check if it's a market day
        market_dates = [d.date() for d in schedule.index]
        if target_date in market_dates:
            raise ValueError(f"Datetime {target_datetime_utc} is not during market hours. "
                           f"Market hours are {start_hour}-{end_hour} {time_zone} on trading days.")
        else:
            raise ValueError(f"Date {target_date} is not a trading day (weekend/holiday).")


def merge_hdf5_files(stock_names,h5_paths,merged_h5_path):
    # --- Generate open market minutes index (NYSE) ---
    start_time = pd.Timestamp("2004-01-01 00:00:00+00:00")
    end_time = pd.Timestamp("2025-05-24 00:00:00+00:00")
    start_hour = "09:30"
    end_hour = "16:00"
    time_zone = "US/Eastern"

    nyse = mcal.get_calendar('XNYS')
    schedule = nyse.schedule(start_date=start_time.date(), end_date=end_time.date())
    open_minutes = []
    for day in schedule.index:
        date_str = day.strftime('%Y-%m-%d')
        start_dt = pd.Timestamp(f"{date_str} {start_hour}", tz=time_zone)
        end_dt = pd.Timestamp(f"{date_str} {end_hour}", tz=time_zone)
        minutes = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(minutes=1), freq='1min')
        minutes = minutes.tz_convert('UTC')
        open_minutes.append(minutes)
    full_index = pd.DatetimeIndex(np.concatenate(open_minutes))

    # --- Merge stocks, aligning to open market minutes ---
    with pd.HDFStore(merged_h5_path, mode='a') as out_store:
        existing_keys = set(out_store.keys())
        total = len(h5_paths)
        remaining = sum(1 for h5_path, stock_name in zip(h5_paths, stock_names) if f'/{stock_name}/data' not in existing_keys)
        for i, (h5_path, stock_name) in enumerate(zip(h5_paths, stock_names), 1):
            key = f'/{stock_name}/data'
            if key in existing_keys:
                print(f"[{i}/{total}] Skipping {stock_name} (already merged)")
                continue
            print(f"[{i}/{total}] Merging {h5_path} as {stock_name}/data ... ({remaining} remaining)")
            with pd.HDFStore(h5_path, mode='r') as in_store:
                df = in_store['data']
                # Ensure index is DatetimeIndex in UTC
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                elif df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
                # Align to open market minutes
                df = df.reindex(full_index)
                # Reset index to integer
                df = df.reset_index(drop=True)
                out_store.put(
                    f'{stock_name}/data',
                    df,
                    format='table',
                    complib='blosc',
                    complevel=9
                )
            remaining -= 1
            print(f"  Done {stock_name} ({remaining} remaining)")
    print(f"✅ All stocks merged into {merged_h5_path}")


# df = pd.read_csv('valid_hdf5_files_2004-01-02_19-00.csv')

# h5_paths = []
# stock_names = []
# for filepath in df['filepath']:
#     h5_paths.append(filepath)
#     stock_names.append(os.path.basename(filepath).replace('.h5',''))

stock_names = ["AAPL","MSFT","TSLA"]
h5_paths = [f"HDF5_output/{stock_name}.h5" for stock_name in stock_names]
merged_h5_path = 'stocks_merged.h5'

merge_hdf5_files(stock_names,h5_paths,merged_h5_path)