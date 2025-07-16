#!/usr/bin/env python3
"""
Inference Script: Loop through sequences and predict using FORTUNE Transformer

This script:
1. Loops through a specified range of sequence indexes
2. Extracts sequences using the refactored preprocessing functions
3. Runs inference using the trained FORTUNE Transformer model
4. Saves predictions and analysis results
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# Import our modules
from preprocess_merged_hdf5 import extract_single_sequence, MERGED_H5, COLUMNS, MODE, HORIZONS
from FORTUNE_Transformer import FORTUNETransformer
from parameters import h_params

# Model parameters (these should match the saved model)
MODEL_PARAMS = {
    'num_features': 6,  # OHLCV + nan_mask (the saved model was trained before abs_final_close was added)
    'd_model': 128,
    'nhead': 8,
    'num_layers': 3,  # The saved model has 3 layers (layers.0, layers.1, layers.2)
    'seq_len': 390,
    'num_stocks': 100,  # The saved model was trained with 100 stocks
    'horizons': {
        '1h': {"volatility_threshold": 0.02, "top": 0.80, "bottom": 0.20},
        '1d': {"volatility_threshold": 0.03, "top": 0.90, "bottom": 0.10}
    },
    'chunk_size': 56,
    'outputs_per_horizon': 7  # 5 ranking logits + 1 risk + 1 regression
}

def load_model(model_path):
    """Load the trained FORTUNE Transformer model."""
    print(f"Loading model from {model_path}...")
    
    # Load model state
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model with same parameters used in training
    model = FORTUNETransformer(
        num_features=MODEL_PARAMS['num_features'],
        d_model=MODEL_PARAMS['d_model'],
        nhead=MODEL_PARAMS['nhead'],
        num_layers=MODEL_PARAMS['num_layers'],
        seq_len=MODEL_PARAMS['seq_len'],
        num_stocks=MODEL_PARAMS['num_stocks'],
        horizons=MODEL_PARAMS['horizons'],
        chunk_size=MODEL_PARAMS['chunk_size']
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'train_loss' in checkpoint:
            print(f"  - Training loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model state dict")
    
    model.eval()
    return model

def interpret_predictions(predictions, stock_names, horizons=None):
    """
    Interpret model predictions into human-readable format.
    
    Args:
        predictions: Model output tensor [num_stocks, num_labels] 
                    where num_labels = len(horizons) * 7 (5 ranking logits + 1 risk + 1 regression per horizon)
        stock_names: List of stock names
        horizons: Dictionary of horizon configurations
    
    Returns:
        List of dictionaries with interpreted predictions
    """
    if horizons is None:
        horizons = MODEL_PARAMS['horizons']
    
    results = []
    predictions_np = predictions.cpu().numpy()
    
    for s, stock in enumerate(stock_names):
        stock_result = {
            'stock': stock,
            'predictions': {}
        }
        for h_i, (horizon_name, horizon_config) in enumerate(horizons.items()):
            base_idx = h_i * 7  # 7 outputs per horizon: 5 ranking logits + 1 risk + 1 regression
            
            # Extract the 7 values for this horizon
            ranking_logits = predictions_np[s, base_idx:base_idx+5]  # First 5 are ranking logits
            risk_raw = predictions_np[s, base_idx + 5]               # 6th is risk logit
            future_return = predictions_np[s, base_idx + 6]          # 7th is regression value
            
            # Ranking: softmax for probabilities, argmax for class (0-4)
            ranking_probs = torch.softmax(torch.tensor(ranking_logits), dim=0).numpy()
            ranking_class = int(np.argmax(ranking_probs))
            ranking_confidence = float(np.max(ranking_probs))
            
            # Risk: sigmoid for probability (0/1)
            risk_prob = torch.sigmoid(torch.tensor(risk_raw)).item()
            horizon_pred = {
                'ranking': {
                    'class': ranking_class,  # 0-4
                    'probs': ranking_probs.tolist(),
                    'prediction': f'class_{ranking_class}',
                    'confidence': ranking_confidence
                },
                'risk': {
                    'low_prob': 1.0 - risk_prob,
                    'high_prob': risk_prob,
                    'prediction': 'high' if risk_prob > 0.5 else 'low',
                    'confidence': max(risk_prob, 1.0 - risk_prob)
                },
                'future_return': {
                    'predicted_return': float(future_return),
                    'direction': 'up' if future_return > 0 else 'down',
                    'magnitude': abs(float(future_return))
                }
            }
            stock_result['predictions'][horizon_name] = horizon_pred
        results.append(stock_result)
    return results

def run_inference_loop(model_path, start_index, end_index, step=1, seq_length=390, 
                      output_file=None, device='cuda', save_raw=False):
    """
    Run inference loop through a range of sequence indexes.
    
    Args:
        model_path: Path to the trained model file
        start_index: Starting sequence index
        end_index: Ending sequence index (exclusive)
        step: Step size between indexes
        seq_length: Length of sequences to extract
        output_file: Output file path (if None, auto-generate)
        device: Device for model inference ('cuda' or 'cpu')
        save_raw: Whether to save raw predictions along with interpreted results
    """
    print("🚀 FORTUNE Transformer Inference Loop")
    print("=" * 50)
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path)
    model = model.to(device)
    
    # Setup output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"inference_results_{start_index}_{end_index}_{timestamp}.json"
    
    print(f"Output file: {output_file}")
    # Get stock names (we'll use the same ones for all sequences)
    print("Getting stock names from data...")
    test_sequence, _, stock_names = extract_single_sequence(
        sequence_index=start_index,
        seq_length=seq_length,
        include_labels=False
    )
    print(f"Found {len(stock_names)} stocks")
    print(f"Raw sequence shape: {test_sequence.shape} (using first 6 features)")
    print(f"Model output structure: {len(HORIZONS)} horizons × 7 outputs per horizon (5 ranking logits + 1 risk + 1 regression)")
    
    # Verify we can extract 6 features for the model
    if test_sequence.shape[2] < 6:
        raise ValueError(f"Sequence has only {test_sequence.shape[2]} features, but model needs 6")
    
    del test_sequence  # Free memory
    
    # Validate model compatibility
    if len(stock_names) != MODEL_PARAMS['num_stocks']:
        print(f"⚠️  Warning: Data has {len(stock_names)} stocks but model expects {MODEL_PARAMS['num_stocks']}")
        if len(stock_names) > MODEL_PARAMS['num_stocks']:
            print(f"   Using first {MODEL_PARAMS['num_stocks']} stocks for inference")
            stock_names = stock_names[:MODEL_PARAMS['num_stocks']]
        else:
            raise ValueError(f"Not enough stocks in data ({len(stock_names)}) for model ({MODEL_PARAMS['num_stocks']})")
    
    # Prepare results storage
    all_results = {
        'metadata': {
            'model_path': model_path,
            'start_index': start_index,
            'end_index': end_index,
            'step': step,
            'seq_length': seq_length,
            'num_stocks': len(stock_names),
            'stock_names': stock_names,
            'horizons': HORIZONS,
            'timestamp': datetime.now().isoformat(),
            'device': str(device)
        },
        'predictions': []
    }
    
    # Run inference loop
    print(f"\n🔄 Running inference from index {start_index} to {end_index} (step={step})")
    
    sequence_indexes = list(range(start_index, end_index, step))
    
    with torch.no_grad():
        for seq_idx in tqdm(sequence_indexes, desc="Processing sequences"):
            try:
                # Extract sequence
                sequence_data, _, current_stock_names = extract_single_sequence(
                    sequence_index=seq_idx,
                    seq_length=seq_length,
                    include_labels=False,
                    stock_names=stock_names  # Use consistent stock names
                )
                
                # Remove the last feature (abs_final_close) to match the model's expected input
                # The model was trained with 6 features: [OHLCV, nan_mask] but current preprocessing adds abs_final_close as 7th
                sequence_data = sequence_data[:, :, :6]  # Keep only first 6 features
                
                # Prepare input tensor
                # sequence_data shape: (seq_length, num_stocks, 6_features)
                input_tensor = torch.tensor(sequence_data, dtype=torch.float32).to(device)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, seq_length, num_stocks, num_features)
                
                # Run inference
                predictions = model(input_tensor)  # Output: (1, num_stocks, num_labels)
                predictions = predictions.squeeze(0)  # Remove batch dimension: (num_stocks, num_labels)
                
                # Debug: Print shape on first iteration
                if seq_idx == sequence_indexes[0]:
                    expected_outputs = len(MODEL_PARAMS['horizons']) * MODEL_PARAMS['outputs_per_horizon']
                    print(f"   Model output shape: {predictions.shape} (expected: {MODEL_PARAMS['num_stocks']}, {expected_outputs})")
                    if predictions.shape[1] != expected_outputs:
                        print(f"   ⚠️  Warning: Expected {expected_outputs} outputs, got {predictions.shape[1]}")
                
                # Interpret predictions
                interpreted_results = interpret_predictions(predictions, stock_names)
                
                # Store results
                result_entry = {
                    'sequence_index': seq_idx,
                    'stocks': interpreted_results
                }
                
                # Optionally store raw predictions
                if save_raw:
                    result_entry['raw_predictions'] = predictions.cpu().numpy().tolist()
                
                all_results['predictions'].append(result_entry)
                
            except Exception as e:
                print(f"⚠️  Error processing sequence {seq_idx}: {e}")
                continue
    
    # Save results
    print(f"\n💾 Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Completed! Processed {len(all_results['predictions'])} sequences")
    print(f"   Results saved to: {output_file}")
    
    # Print sample results
    if all_results['predictions']:
        print("\n📊 Sample Prediction (first sequence):")
        sample = all_results['predictions'][0]
        print(f"   Sequence Index: {sample['sequence_index']}")
        
        # Show predictions for first few stocks
        for i, stock_data in enumerate(sample['stocks'][:3]):
            stock = stock_data['stock']
            print(f"   {stock}:")
            for horizon, pred in stock_data['predictions'].items():
                ranking = pred['ranking']
                risk = pred['risk']
                ret = pred['future_return']
                print(f"     {horizon}: class {ranking['class']} (conf {ranking['confidence']:.3f}), {risk['prediction']} risk ({risk['confidence']:.3f}), return {ret['predicted_return']:.4f}")
        if len(sample['stocks']) > 3:
            print(f"   ... and {len(sample['stocks']) - 3} more stocks")
    
    return all_results

def analyze_predictions(results_file):
    """
    Analyze prediction results and generate summary statistics.
    
    Args:
        results_file: Path to the JSON results file
    """
    print(f"\n📈 Analyzing predictions from {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    predictions = results['predictions']
    stock_names = results['metadata']['stock_names']
    horizons = list(results['metadata']['horizons'].keys())
    
    print(f"Loaded {len(predictions)} predictions for {len(stock_names)} stocks")
    
    # Aggregate statistics
    stats = {
        'total_predictions': len(predictions),
        'stocks_analyzed': len(stock_names),
        'horizons': horizons,
        'summary': {}
    }
    
    for horizon in horizons:
        horizon_stats = {
            'class_counts': [0, 0, 0, 0, 0],
            'high_risk_predictions': 0,
            'low_risk_predictions': 0,
            'avg_predicted_return': 0.0,
            'positive_returns': 0,
            'negative_returns': 0
        }
        total_return = 0
        total_count = 0
        for pred in predictions:
            for stock_data in pred['stocks']:
                stock_pred = stock_data['predictions'][horizon]
                # Count ranking class predictions
                ranking_class = stock_pred['ranking']['class']
                if 0 <= ranking_class < 5:
                    horizon_stats['class_counts'][ranking_class] += 1
                # Count risk predictions
                if stock_pred['risk']['prediction'] == 'high':
                    horizon_stats['high_risk_predictions'] += 1
                else:
                    horizon_stats['low_risk_predictions'] += 1
                # Aggregate returns
                pred_return = stock_pred['future_return']['predicted_return']
                total_return += pred_return
                total_count += 1
                if pred_return > 0:
                    horizon_stats['positive_returns'] += 1
                else:
                    horizon_stats['negative_returns'] += 1
        horizon_stats['avg_predicted_return'] = float(total_return / total_count if total_count > 0 else 0)
        stats['summary'][horizon] = horizon_stats
    # Print summary
    print("\n📊 Prediction Summary:")
    for horizon, horizon_stats in stats['summary'].items():
        total = sum(horizon_stats['class_counts'])
        print(f"\n{horizon.upper()} Horizon:")
        for c in range(5):
            pct = (horizon_stats['class_counts'][c] / total * 100) if total > 0 else 0
            print(f"  Class {c}: {horizon_stats['class_counts'][c]} ({pct:.1f}%)")
        print(f"  Risk: {horizon_stats['high_risk_predictions']}/{horizon_stats['low_risk_predictions']} high/low ({horizon_stats['high_risk_predictions']/total*100:.1f}%/{horizon_stats['low_risk_predictions']/total*100:.1f}%)")
        print(f"  Returns: {horizon_stats['positive_returns']}/{horizon_stats['negative_returns']} pos/neg ({horizon_stats['positive_returns']/total*100:.1f}%/{horizon_stats['negative_returns']/total*100:.1f}%)")
        print(f"  Avg Predicted Return: {horizon_stats['avg_predicted_return']:.4f}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Run FORTUNE Transformer inference on sequence ranges")
    parser.add_argument('--model_path', type=str, default='fortune_transformer_263_stocks_final.pt',
                       help='Path to the trained model file')
    parser.add_argument('--start_index', type=int, default=10000,
                       help='Starting sequence index')
    parser.add_argument('--end_index', type=int, default=10100,
                       help='Ending sequence index (exclusive)')
    parser.add_argument('--step', type=int, default=1,
                       help='Step size between sequence indexes')
    parser.add_argument('--seq_length', type=int, default=390,
                       help='Length of sequences to extract')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (auto-generated if not specified)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--save_raw', action='store_true',
                       help='Save raw predictions along with interpreted results')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze results after inference')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Run inference
    results = run_inference_loop(
        model_path=args.model_path,
        start_index=args.start_index,
        end_index=args.end_index,
        step=args.step,
        seq_length=args.seq_length,
        output_file=args.output,
        device=args.device,
        save_raw=args.save_raw
    )
    
    print(results)
    
    # Analyze results if requested
    if args.analyze:
        output_file = args.output or f"inference_results_{args.start_index}_{args.end_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyze_predictions(output_file)

if __name__ == "__main__":
    main()
