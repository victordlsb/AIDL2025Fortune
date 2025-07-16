import pandas as pd
import numpy as np
import tables
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

MERGED_H5 = 'stocks_merged.h5'  # Path to merged HDF5 file

# --- Parameters (customize as needed) ---
SEQ_LEN = 390
STRIDE = 185
COLUMNS = ['open', 'high', 'low', 'close', 'volume']
MODE = 'relative'  # or 'absolute'
HORIZONS = {
    '1h':   {"volatility_threshold": 0.01},   # 1% daily volatility
    '1d':   {"volatility_threshold": 0.02}    # 2% daily volatility
}

BASE_UNITS = {'m': 1, 'h': 60, 'd': 390, 'w': 1950, 'mo': 7800}

def get_horizon_offsets():
    offsets = {}
    import re
    for h in HORIZONS.keys():
        match = re.match(r"(\d*)([a-z]+)", h)
        if match:
            num, unit = match.groups()
            num = int(num) if num else 1
            offsets[h] = num * BASE_UNITS[unit]
    return offsets

def process_single_stock_sequence(stock_df, seq_length, mode, columns):
    """
    Process a single stock's sequence data (shared between single and batch processing)
    
    Args:
        stock_df: DataFrame with stock data for the sequence
        seq_length: Length of sequence to extract
        mode: 'relative' or 'absolute' transformation mode
        columns: List of columns to extract
    
    Returns:
        numpy array of processed sequence data for this stock
    """
    data_seq = stock_df.values
    
    # Get absolute final close value before transformations
    abs_final_close = data_seq[-1, 3:4].copy()  # Shape: (1,) - close is index 3
    
    # Apply relative transformation if specified
    if mode == 'relative':
        base = data_seq[0:1, :]  # First timestep as base
        relative = (data_seq - base) / (base + 1e-8)
        data_seq = relative
    
    # Create nan mask (1 if any feature is nan, else 0)
    nan_mask = np.any(np.isnan(data_seq), axis=1, keepdims=True).astype(np.float32)
    
    # Create absolute final close feature across all timesteps
    abs_final_features = np.tile(abs_final_close, (seq_length, 1))
    
    # Combine: original data + nan_mask + absolute final close
    combined = np.concatenate([data_seq, nan_mask, abs_final_features], axis=1)
    combined = np.nan_to_num(combined)
    
    return combined

def apply_quantile_labels(all_labels, horizons):
    """
    Apply rank-based 5-class labeling across all stocks (shared function)

    Args:
        all_labels: numpy array of shape (num_stocks, num_labels)
        horizons: Dictionary of horizon configurations

    Returns:
        Updated labels array with 5-class rank labels applied (0 = top, 4 = bottom)
    """
    for h_i, (horizon_name, horizon) in enumerate(horizons.items()):
        base_idx = h_i * 3
        future_returns = all_labels[:, base_idx + 2]
        valid = ~np.isnan(future_returns)

        if valid.sum() >= 5:
            # Compute rank-based percentiles
            valid_returns = future_returns[valid]
            ranks = valid_returns.argsort().argsort()
            percentiles = ranks / (len(ranks) - 1 + 1e-8)

            # Assign class labels based on quantiles (0 = top 20%, 4 = bottom 20%)
            labels = np.full_like(future_returns, np.nan)
            valid_idx = np.where(valid)[0]
            labels[valid_idx] = 4  # default to bottom
            labels[valid_idx[percentiles >= 0.80]] = 0  # top 20%
            labels[valid_idx[(percentiles >= 0.60) & (percentiles < 0.80)]] = 1
            labels[valid_idx[(percentiles >= 0.40) & (percentiles < 0.60)]] = 2
            labels[valid_idx[(percentiles >= 0.20) & (percentiles < 0.40)]] = 3
            labels[valid_idx[percentiles < 0.20]] = 4

            all_labels[:, base_idx + 0] = labels
        else:
            all_labels[:, base_idx + 0] = np.nan

    return all_labels

def extract_single_sequence(sequence_index, merged_h5_path=MERGED_H5, seq_length=390, 
                          columns=COLUMNS, mode=MODE, horizons=HORIZONS, 
                          include_labels=True, stock_names=None):
    """
    Extract a single sequence with labels for a given index.
    
    Args:
        sequence_index: The starting index for the sequence
        merged_h5_path: Path to the merged HDF5 file
        seq_length: Length of the sequence to extract
        columns: List of columns to extract ['open', 'high', 'low', 'close', 'volume']
        mode: 'relative' or 'absolute' transformation mode
        horizons: Dictionary of horizons for label calculation
        include_labels: Whether to calculate and return labels
        stock_names: List of stock names to use (if None, will read from file)
    
    Returns:
        tuple: (sequence_data, labels, stock_names) or (sequence_data, None, stock_names) if include_labels=False
        - sequence_data: numpy array of shape (seq_length, num_stocks, num_features)
        - labels: numpy array of shape (num_stocks, num_labels) or None
        - stock_names: list of stock names used
    """
    horizon_offsets = get_horizon_offsets()
    num_horizons = len(horizons)
    num_labels = num_horizons * 3
    
    with pd.HDFStore(merged_h5_path, mode='r') as merged_store:
        # Get stock names if not provided
        if stock_names is None:
            stock_names = []
            for k in merged_store.keys():
                if k.count('/') == 2 and k.endswith('/data'):
                    stock_names.append(k.split('/')[1])
        
        num_stocks = len(stock_names)
        
        # Check if sequence is valid - get total rows by selecting a large range
        sample_df = merged_store.select(f'/{stock_names[0]}/data', start=0, stop=None)
        total_rows = len(sample_df)
        del sample_df  # Free memory
        
        if sequence_index + seq_length > total_rows:
            raise ValueError(f"Sequence index {sequence_index} + seq_length {seq_length} exceeds data length {total_rows}")
        
        # Calculate data window needed (including future lookups for labels)
        max_future_offset = max(horizon_offsets.values()) if include_labels else 0
        data_end = sequence_index + seq_length + max_future_offset
        data_end = min(data_end, total_rows)  # Don't exceed available data
        
        # Read data for all stocks
        stock_dfs = {}
        for stock in stock_names:
            df = merged_store.select(f'/{stock}/data', 
                                   start=sequence_index, 
                                   stop=data_end, 
                                   columns=columns)
            stock_dfs[stock] = df.reset_index(drop=True)
        
        # Extract sequence data using shared function
        seqs = []
        for stock in stock_names:
            df = stock_dfs[stock].iloc[:seq_length]  # Take only seq_length rows
            processed_seq = process_single_stock_sequence(df, seq_length, mode, columns)
            seqs.append(processed_seq)
        
        # Stack into final sequence format
        sequence_data = np.stack(seqs, axis=1).astype(np.float32)  # (seq_length, num_stocks, features)
        
        # Calculate labels if requested
        labels = None
        if include_labels:
            all_labels = np.full((num_stocks, num_labels), np.nan, dtype=np.float32)
            
            # Calculate labels for each stock using shared function
            for s, stock in enumerate(stock_names):
                stock_labels = calculate_single_stock_labels(
                    stock_dfs[stock], sequence_index, seq_length, 
                    horizon_offsets, horizons, columns, min_start_offset=0
                )
                all_labels[s, :] = stock_labels
            
            # Apply quantile-based top/bottom labeling using shared function
            labels = apply_quantile_labels(all_labels, horizons)
    
    return sequence_data, labels, stock_names

def process_sequence_batch(args):
    batch_indices, stock_names, columns, seq_len, full_index, merged_h5_path, mode, horizon_offsets, horizons, num_labels = args
    results_seq = []
    results_tgt = []
    with pd.HDFStore(merged_h5_path, mode='r') as merged_store:
        # For memory efficiency, prefetch only the required window for each stock for the whole batch
        # Find the min/max indices needed for this batch
        min_start = min(i for i in batch_indices)
        max_stop = max(i + seq_len - 1 for i in batch_indices)
        # For each stock, read only the window needed for all sequences in the batch
        stock_dfs = {}
        for stock in stock_names:
            df = merged_store.select(f'/{stock}/data', start=min_start, stop=max_stop+1, columns=columns)
            stock_dfs[stock] = df.reset_index(drop=True)
        for i in batch_indices:
            # Safeguard: skip if sequence would go out of bounds
            if (i - min_start) < 0 or (i - min_start + seq_len) > len(stock_dfs[stock_names[0]]):
                continue
            seqs = []
            seq_start = i - min_start
            seq_end = seq_start + seq_len
            for stock in stock_names:
                df = stock_dfs[stock].iloc[seq_start:seq_end]
                # Use shared function for sequence processing
                processed_seq = process_single_stock_sequence(df, seq_len, mode, columns)
                seqs.append(processed_seq)
            data = np.stack(seqs, axis=1)  # (SEQ_LEN, num_stocks, features+nanmask+abs_final_features)
            # Feature structure: [OHLCV, nan_mask, abs_final_close] = 5 + 1 + 1 = 7 features per stock
            num_stocks = len(stock_names)
            seq_targets = np.full((num_stocks, num_labels), np.nan, dtype=np.float32)
            # Label index convention per horizon:
            # [0] = label_ranking (integer 0–4, rank class)
            # [1] = label_risk (binary)
            # [2] = future_return (float)
            # Calculate labels for each stock using shared function
            for s, stock in enumerate(stock_names):
                stock_labels = calculate_single_stock_labels(
                    stock_dfs[stock], i, seq_len, 
                    horizon_offsets, horizons, columns, min_start_offset=min_start
                )
                seq_targets[s, :] = stock_labels
            # Apply quantile-based top/bottom labeling using shared function
            seq_targets = apply_quantile_labels(seq_targets, horizons)
            results_seq.append(data.astype(np.float32))
            results_tgt.append(seq_targets.astype(np.float32))
    if len(results_seq) == 0 or len(results_tgt) == 0:
        return None
    return np.stack(results_seq), np.stack(results_tgt)

def calculate_single_stock_labels(stock_df, sequence_index, seq_length, horizon_offsets, horizons, columns, min_start_offset=0):
    """
    Calculate labels for a single stock given its dataframe and sequence information.
    Handles both single sequence extraction and batch processing cases.
    
    Args:
        stock_df: DataFrame with stock data (including future data for labels)
        sequence_index: Original sequence index in the full dataset
        seq_length: Length of the sequence
        horizon_offsets: Dictionary of horizon offsets
        horizons: Dictionary of horizon configurations
        columns: List of columns to use
        min_start_offset: Offset to adjust for windowed data reads (default: 0 for single sequence)
    
    Returns:
        numpy array of labels for all horizons
    """
    # Label index convention per horizon:
    # [0] = label_ranking (integer class from 0 to 4)
    num_labels = len(horizons) * 3
    labels = np.full(num_labels, np.nan, dtype=np.float32)
    
    # Get current values from the last timestep of the sequence
    v_now = stock_df.iloc[seq_length - 1][columns].values.astype(np.float64)
    
    for h_i, (horizon_name, horizon) in enumerate(horizons.items()):
        offset = horizon_offsets[horizon_name]
        
        # Calculate future index - for batch processing, adjust for windowed reads
        if min_start_offset > 0:
            # Batch processing case: sequence_index is relative to full dataset
            future_idx = sequence_index + seq_length - 1 + offset
            rel_future_idx = future_idx - min_start_offset
            if rel_future_idx >= len(stock_df):
                rel_future_idx = len(stock_df) - 1
        else:
            # Single sequence case: index is relative to the data window
            rel_future_idx = seq_length - 1 + offset
            if rel_future_idx >= len(stock_df):
                rel_future_idx = len(stock_df) - 1
        
        v_future = stock_df.iloc[rel_future_idx][columns].values.astype(np.float64)
        
        # Calculate labels using ORIGINAL (non-transformed) data
        # Get original close prices for proper return calculation
        original_closes = stock_df.iloc[:seq_length][columns[3]].values
        
        # Calculate future return using original prices
        current_close = original_closes[-1]
        future_close = stock_df.iloc[rel_future_idx][columns[3]]
        future_return = (future_close - current_close) / (current_close + 1e-8) if not np.isnan(current_close) and not np.isnan(future_close) else np.nan
        
        # Calculate risk based on sequence volatility using ORIGINAL prices
        returns = np.diff(original_closes) / (original_closes[:-1] + 1e-8)
        volatility = np.std(returns) if len(returns) > 0 else np.nan
        label_risk = float(volatility > float(horizon['volatility_threshold'])) if not np.isnan(volatility) else np.nan
        
        # Store labels
        base_idx = h_i * 3
        labels[base_idx] = np.nan  # Will be set by quantile post-processing
        labels[base_idx + 1] = label_risk
        labels[base_idx + 2] = future_return
    
    return labels

def process_all_stocks(num_sequences=None, start_index=0, num_workers=1, seq_length=1200, stride=60, batch_size=64, num_stocks = 100):
    initial_sequence_idx = start_index
    final_sequence_idx = start_index + (num_sequences - 1) * stride if num_sequences is not None else None
    output_file = f"{initial_sequence_idx}_to_{final_sequence_idx if final_sequence_idx is not None else 'end'}_stride_{stride}_seqlen_{seq_length}_stocks_{num_stocks}_with_close.h5"
    with pd.HDFStore(MERGED_H5, mode='r') as merged_store:
        # Only include stock names that have a '/data' group at the root
        stock_names = []
        for k in merged_store.keys():
            if k.count('/') == 2 and k.endswith('/data'):
                stock_names.append(k.split('/')[1])
        # print(f"[INFO] Found {len(stock_names)} stocks in merged file.")
        
        # Get total number of rows
        sample_df = merged_store.select(f'/{stock_names[0]}/data', start=0, stop=None)
        total_rows = len(sample_df)
        del sample_df  # Free memory
        full_index = np.arange(total_rows)
        
        num_stocks = len(stock_names)
        horizon_offsets = get_horizon_offsets()
        num_horizons = len(HORIZONS)
        num_labels = num_horizons * 3
        max_sequences = (len(full_index) - seq_length) // stride
        if num_sequences is None or num_sequences > max_sequences:
            num_sequences = max_sequences
        # Early exit if no sequences can be generated
        if start_index > (len(full_index) - seq_length):
            # print(f"[INFO] No sequences to generate: start_index {start_index} > last valid index {len(full_index) - seq_length}")
            return
        # print(f"[INFO] Extracting {num_sequences} sequences (stride={stride}, seq_len={seq_length}) from index {start_index}.")
        with tables.open_file(output_file, mode='w') as out_h5:
            # Preallocate extendable arrays
            atom_seq = tables.Atom.from_dtype(np.dtype(np.float32))
            atom_tgt = tables.Atom.from_dtype(np.dtype(np.float32))
            seq_shape = (0, seq_length, num_stocks, len(COLUMNS)+2)  # (N, T, S, F+nanmask+abs_final_close)
            tgt_shape = (0, num_stocks, num_labels)
            filters = tables.Filters(complevel=9, complib='blosc')
            seq_array = out_h5.create_earray('/', 'sequences', atom=atom_seq, shape=seq_shape, expectedrows=num_sequences, filters=filters)
            tgt_array = out_h5.create_earray('/', 'targets', atom=atom_tgt, shape=tgt_shape, expectedrows=num_sequences, filters=filters)
            batch_indices = [list(range(start, min(start+batch_size, num_sequences))) for start in range(0, num_sequences, batch_size)]
            args_list = [
                ( [start_index + idx*stride for idx in batch], stock_names, COLUMNS, seq_length, full_index, MERGED_H5, MODE, horizon_offsets, HORIZONS, num_labels )
                for batch in batch_indices
            ]
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for batch_result in tqdm(executor.map(process_sequence_batch, args_list), total=len(args_list), desc='Processing batches'):
                    if batch_result is None:
                        continue
                    batch_seq, batch_tgt = batch_result
                    seq_array.append(batch_seq)
                    tgt_array.append(batch_tgt)
            # print(f"Saved {seq_array.nrows} sequences, {seq_length} timesteps, {num_stocks} stocks, {len(COLUMNS)+2} features (OHLCV+nanmask+abs_final_close).")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sequences', type=int, default=None, help='Number of sequences to extract (default: all)')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for sequence extraction')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--seq_length', type=int, default=900, help='Length of the sequence (default: 900)')
    parser.add_argument('--stride', type=int, default=60, help='Stride between sequences (default: 60)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing (default: 64)')
    args = parser.parse_args()
    args.num_workers=12
    args.stride = 185
    args.seq_length = 390
    for start_index in [0]:
        process_all_stocks(
            num_sequences=args.num_sequences,
            start_index=start_index,
            num_workers=args.num_workers,
            seq_length=args.seq_length,
            stride=args.stride,
            batch_size=args.batch_size
        )
        # print(f"Stocks processed. Start index{start_index}")

if __name__ == "__main__":
    main()
