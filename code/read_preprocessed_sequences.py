#!/usr/bin/env python3
"""
Script to read preprocessed HDF5 sequences and display close values and targets.
"""

import numpy as np
import tables
import glob
import argparse
import pandas as pd
import os

def read_preprocessed_file(file_path, max_sequences=None, show_details=False):
    """
    Read preprocessed HDF5 file and display close values and targets.
    
    Args:
        file_path: Path to the preprocessed HDF5 file
        max_sequences: Maximum number of sequences to display (None for all)
        show_details: Whether to show detailed information for each sequence
    """
    print(f"Reading preprocessed file: {file_path}")
    
    with tables.open_file(file_path, mode='r') as h5_file:
        # Get basic info
        sequences = h5_file.root.sequences
        targets = h5_file.root.targets
        
        num_sequences, seq_length, num_stocks, num_features = sequences.shape
        _, _, num_labels = targets.shape
        
        print(f"\nFile structure:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Number of sequences: {num_sequences}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Number of stocks: {num_stocks}")
        print(f"  Features per stock: {num_features}")
        print(f"  Labels per stock: {num_labels}")
        
        # Feature structure: [OHLCV, nan_mask, abs_final_close] = 7 features
        print(f"\nFeature structure (per stock):")
        print(f"  [0-4]: OHLCV (open, high, low, close, volume)")
        print(f"  [5]:   NaN mask (1 if any feature is NaN, 0 otherwise)")
        print(f"  [6]:   Absolute final close value")
        
        # Target structure: 2 horizons × 5 labels = 10 labels per stock
        print(f"\nTarget structure (per stock, per horizon):")
        print(f"  Horizons: 1h, 1d")
        print(f"  Labels per horizon: [up, best, worst, risk, future_return]")
        print(f"  Total labels: {num_labels}")
        targets_data = targets[:]
        # Get current prices (last close in each sequence)
        current_prices = sequences[:, -1, :, 6]  # Shape: (num_sequences, num_stocks)
        current_prices[current_prices < 1] = np.nan
        # Assume targets_data has shape (time, tickers, features)
        # Fill NaNs using interpolation column-wise
        df = pd.DataFrame(current_prices)
        df = df.interpolate(axis=0, method='linear').ffill().bfill()
        
        # Get the filled numpy array back
        current_prices = df.values
        # Add current price as new column to targets data
        targets_data = np.dstack([targets_data, current_prices[:, :, np.newaxis]])
        
        # Assume targets_data has shape (time, tickers, features)
        T, N, F = targets_data.shape
        
        # Fill NaNs forward and then backward along the time axis (axis=0)
        for i in range(N):  # loop over tickers
            for j in range(F):  # loop over features
                # Use pandas Series to apply ffill and bfill
                s = pd.Series(targets_data[:, i, j])
                filled = s.ffill().bfill().values
                targets_data[:, i, j] = filled
        
        print(f"\nModified targets shape with current price: {targets_data.shape}")

        # Limit sequences to display
        for seq_idx in range(max_sequences):
            seq_data = sequences[seq_idx]  # Shape: (seq_length, num_stocks, num_features)
            seq_targets = targets[seq_idx]  # Shape: (num_stocks, num_labels)
            
            print(f"\nSequence {seq_idx}:")
            print("-" * 40)
            
            # Extract close values and absolute final close for each stock
            for stock_idx in range(num_stocks):
                stock_data = seq_data[:, stock_idx, :]  # Shape: (seq_length, num_features)
                stock_targets = seq_targets[stock_idx]  # Shape: (num_labels,)
                
                # Get close values (feature index 3) and absolute final close (feature index 6)
                close_values = stock_data[:, 3]  # Relative close values
                abs_final_close = stock_data[0, 6]  # Absolute final close (same for all timesteps)
                
                print(f"  Stock {stock_idx}:")
                print(f"    Absolute final close: {abs_final_close:.4f}")
                
                if show_details:
                    print(f"    Close values (first 10 timesteps): {close_values[:10]}")
                    print(f"    Close values (last 10 timesteps): {close_values[-10:]}")
                else:
                    print(f"    Close range: [{close_values.min():.4f}, {close_values.max():.4f}]")
                
                # Display targets
                print(f"    Targets:")
                
                # 1h horizon targets (indices 0-2)
                h1_top_bottom = stock_targets[0]
                h1_risk = stock_targets[1]
                h1_return = stock_targets[2]
                
                print(f"      1h: top_bottom={h1_top_bottom:.3f}, risk={h1_risk:.3f}, return={h1_return:.6f}")
                
                # 1d horizon targets (indices 3-5)
                d1_top_bottom = stock_targets[3]
                d1_risk = stock_targets[4]
                d1_return = stock_targets[5]
                
                print(f"      1d: top_bottom={d1_top_bottom:.3f}, risk={d1_risk:.3f}, return={d1_return:.6f}")
                
                # Check for NaN values
                nan_count = np.isnan(stock_targets).sum()
                if nan_count > 0:
                    print(f"      NaN targets: {nan_count}/{num_labels}")
                
                print()
            
            if not show_details and seq_idx < max_sequences - 1:
                print("    ... (use --details for more info)")
    return targets_data


def main():
    parser = argparse.ArgumentParser(description='Read and display preprocessed HDF5 sequences')
    parser.add_argument('--file', '-f', type=str, help='Path to preprocessed HDF5 file')
    parser.add_argument('--max_sequences', '-n', type=int, default=3, help='Maximum number of sequences to display (default: 3)')
    parser.add_argument('--details', '-d', action='store_true', help='Show detailed close values for each sequence')
    parser.add_argument('--list', '-l', action='store_true', help='List all available preprocessed files')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available preprocessed files:")
        files = glob.glob("*_stride_*_seqlen_*_stocks_*.h5")
        if files:
            for f in sorted(files):
                print(f"  {f}")
        else:
            print("  No preprocessed files found in current directory")
        return
    
    if args.file:
        if not args.file.endswith('.h5'):
            print("Error: File must be an HDF5 file (.h5)")
            return
        
        try:
            targets = read_preprocessed_file(args.file, args.max_sequences, args.details)
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        # Auto-detect most recent file
        files = glob.glob("*_stride_*_seqlen_*_stocks_*.h5")
        if files:
            latest_file = max(files, key=lambda f: os.path.getctime(f))
            print(f"Auto-detected latest file: {latest_file}")
            targets = read_preprocessed_file(latest_file, args.max_sequences, args.details)
            check=1
        else:
            print("No preprocessed files found. Use --list to see available files or specify with --file")
    return targets[:817,:,:]


if __name__ == "__main__":
    import os
    main()
