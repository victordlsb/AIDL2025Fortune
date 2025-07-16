import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from FORTUNE_Transformer import FORTUNETransformer, mixed_loss
from hdf5_pytorch_dataset import HDF5SequenceDataset
from parameters import h_params, stock_names
from plot_metrics import create_training_plot, create_loss_components_plot, create_metrics_plot, create_per_label_accuracy_plot, create_sign_accuracy_plot, create_per_label_f1_plot
import gc
import time
import os
import datetime

def clear_memory():
    """Clear Python and GPU memory - less aggressive version"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Remove the synchronize call to avoid blocking

def calculate_accuracy_and_f1(predictions, targets, threshold=0.5):
    """
    Calculate balanced accuracy and F1 score for binary classification tasks (global and per-label)
    New structure: For each horizon, indices 0-1 are binary (top_or_bottom, risk), index 2 is regression
    Global metrics are calculated as the average of per-label metrics
    Returns: (global_accuracy, global_f1, per_label_accuracy, per_label_f1)
    """
    with torch.no_grad():
        # Extract predictions and targets per label type
        label_names = ['ranking', 'risk']
        label_data = {name: {'preds': [], 'targets': []} for name in label_names}

        num_horizons = targets.shape[-1] // 3  # Each horizon has 3 outputs
        for horizon in range(num_horizons):
            base_idx = horizon * 3
            # Index 0: ranking (multiclass), Index 1: risk (binary)
            # RANKING (multiclass)
            ranking_idx = base_idx + 0
            if ranking_idx < targets.shape[-1]:
                pred_logits = predictions[:, :, ranking_idx:ranking_idx+5]  # (B, S, 5)
                target_ranking = targets[:, :, ranking_idx]  # (B, S)
                valid_mask = ~torch.isnan(target_ranking)
                if valid_mask.any():
                    pred_valid = pred_logits[valid_mask]
                    target_valid = target_ranking[valid_mask].long()
                    label_data['ranking']['preds'].append(pred_valid)
                    label_data['ranking']['targets'].append(target_valid)
            # RISK (binary)
            risk_idx = base_idx + 1
            if risk_idx < targets.shape[-1]:
                pred_risk = predictions[:, :, risk_idx]
                target_risk = targets[:, :, risk_idx]
                valid_mask = ~torch.isnan(target_risk)
                if valid_mask.any():
                    pred_valid = pred_risk[valid_mask]
                    target_valid = target_risk[valid_mask]
                    label_data['risk']['preds'].append(pred_valid)
                    label_data['risk']['targets'].append(target_valid)

        # Check if we have any valid data
        has_valid_data = any(data['preds'] for data in label_data.values())
        if not has_valid_data:
            return 0.0, 0.0, {}, {}

        def calculate_ranking_metrics(preds, targets):
            # preds: (N, 5) logits, targets: (N,) int
            pred_classes = torch.argmax(preds, dim=-1)
            correct = (pred_classes == targets)
            accuracy = correct.float().mean().item()
            # Macro F1
            f1s = []
            for c in range(5):
                tp = ((pred_classes == c) & (targets == c)).sum().item()
                fp = ((pred_classes == c) & (targets != c)).sum().item()
                fn = ((pred_classes != c) & (targets == c)).sum().item()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1s.append(f1)
            macro_f1 = sum(f1s) / len(f1s)
            return accuracy, macro_f1

        def calculate_risk_metrics(preds, targets):
            # preds: logits, targets: float (0/1)
            preds = torch.sigmoid(preds)
            preds = (preds > threshold).float()
            true_positives = ((preds == 1) & (targets == 1)).float().sum()
            true_negatives = ((preds == 0) & (targets == 0)).float().sum()
            false_positives = ((preds == 1) & (targets == 0)).float().sum()
            false_negatives = ((preds == 0) & (targets == 1)).float().sum()
            sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
            specificity = true_negatives / (true_negatives + false_positives + 1e-8)
            precision = true_positives / (true_positives + false_positives + 1e-8)
            balanced_accuracy = ((sensitivity + specificity) / 2).item()
            f1_score = (2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)).item()
            return balanced_accuracy, f1_score

        per_label_accuracy = {}
        per_label_f1 = {}
        valid_labels = []
        # Ranking
        if label_data['ranking']['preds']:
            label_preds = torch.cat(label_data['ranking']['preds'])
            label_targets = torch.cat(label_data['ranking']['targets'])
            per_label_accuracy['ranking'], per_label_f1['ranking'] = calculate_ranking_metrics(label_preds, label_targets)
            valid_labels.append('ranking')
        else:
            per_label_accuracy['ranking'] = 0.0
            per_label_f1['ranking'] = 0.0
        # Risk
        if label_data['risk']['preds']:
            label_preds = torch.cat(label_data['risk']['preds'])
            label_targets = torch.cat(label_data['risk']['targets'])
            per_label_accuracy['risk'], per_label_f1['risk'] = calculate_risk_metrics(label_preds, label_targets)
            valid_labels.append('risk')
        else:
            per_label_accuracy['risk'] = 0.0
            per_label_f1['risk'] = 0.0

        # Calculate global metrics as average of per-label metrics (only from valid labels)
        if valid_labels:
            global_balanced_accuracy = sum(per_label_accuracy[label] for label in valid_labels) / len(valid_labels)
            global_f1_score = sum(per_label_f1[label] for label in valid_labels) / len(valid_labels)
        else:
            global_balanced_accuracy = 0.0
            global_f1_score = 0.0

        calculate_accuracy_and_f1.per_label_accuracy = per_label_accuracy
        calculate_accuracy_and_f1.per_label_f1 = per_label_f1
        return global_balanced_accuracy, global_f1_score, per_label_accuracy, per_label_f1

def calculate_accuracy(predictions, targets, threshold=0.5):
    """
    Backward compatibility wrapper for calculate_accuracy_and_f1
    Returns only the global balanced accuracy
    """
    global_acc, _, per_label_acc, _ = calculate_accuracy_and_f1(predictions, targets, threshold)
    # Keep the per_label_accuracy attribute for backward compatibility
    calculate_accuracy.per_label_accuracy = per_label_acc
    return global_acc

def calculate_f1_score(predictions, targets, threshold=0.5):
    """
    Backward compatibility wrapper for calculate_accuracy_and_f1
    Returns only the global F1 score
    """
    _, global_f1, _, per_label_f1 = calculate_accuracy_and_f1(predictions, targets, threshold)
    # Store per_label_f1 as attribute for potential future use
    calculate_f1_score.per_label_f1 = per_label_f1
    return global_f1

def calculate_smape(predictions, targets):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) for regression tasks
    New structure: For each horizon, index 2 is regression (returns), indices 0-1 are binary
    """
    with torch.no_grad():
        # Extract only regression targets (index 2 for each horizon)
        regression_preds = []
        regression_targets = []
        
        num_horizons = targets.shape[-1] // 3  # Each horizon has 3 outputs
        
        for horizon in range(num_horizons):
            base_idx = horizon * 3
            regression_idx = base_idx + 2  # Index 2 is regression (returns)
            
            if regression_idx < targets.shape[-1]:
                pred_slice = predictions[:, :, regression_idx]
                target_slice = targets[:, :, regression_idx]
                
                # Only include non-NaN targets
                valid_mask = ~torch.isnan(target_slice)
                if valid_mask.any():
                    regression_preds.append(pred_slice[valid_mask])
                    regression_targets.append(target_slice[valid_mask])
        
        if not regression_preds:
            return 0.0  # No regression targets found
        
        # Concatenate all regression predictions and targets
        all_regression_preds = torch.cat(regression_preds)
        all_regression_targets = torch.cat(regression_targets)
        
        # Calculate Symmetric SMAPE (SMAPE)
        denominator = (torch.abs(all_regression_targets) + torch.abs(all_regression_preds)) / 2.0 + 1e-8
        smape = torch.mean(torch.abs(all_regression_targets - all_regression_preds) / denominator) * 100
        return smape.item()

def calculate_returns_sign_accuracy(predictions, targets, threshold=0.5):
    # Only compare regression channels (index 2 for each horizon)
    num_horizons = targets.shape[-1] // 3
    pred_regression = []
    target_regression = []
    for h in range(num_horizons):
        reg_idx = h * 3 + 2
        if reg_idx < predictions.shape[-1] and reg_idx < targets.shape[-1]:
            pred_regression.append(predictions[:, :, reg_idx])
            target_regression.append(targets[:, :, reg_idx])
    if not pred_regression or not target_regression:
        return torch.tensor(0.0)
    pred_reg = torch.cat([p.flatten() for p in pred_regression])
    target_reg = torch.cat([t.flatten() for t in target_regression])
    valid = (target_reg != 0) | (pred_reg != 0)
    correct = (torch.sign(pred_reg[valid]) == torch.sign(target_reg[valid]))
    if correct.numel() == 0:
        return torch.tensor(0.0)
    return correct.float().mean()

def train_epoch(model, dataloader, optimizer, device, print_every=10, accumulation_steps=4):  # Increased accumulation
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_f1_score = 0
    total_smape = 0
    total_steps = 0
    total_returns_sign_accuracy = 0
    start_time = time.time()
    
    # Track loss components
    epoch_binary_losses = []
    epoch_regression_losses = []
    
    # Track per-label accuracies and F1s
    label_names = ['ranking', 'risk']
    epoch_per_label_accuracies = {name: [] for name in label_names}
    epoch_per_label_f1s = {name: [] for name in label_names}

    optimizer.zero_grad()  # Initialize gradients

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    for i, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc="Training")):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast('cuda'):
            preds = model(batch_x)
            loss, binary_losses, regression_losses = mixed_loss(
                preds, batch_y,
                regression_loss_weight=h_params["regression_loss_weight"],
                classification_loss_weight=h_params["classification_loss_weight"],
                huber_delta=h_params["huber_delta"],
                huber_weight=h_params["huber_weight"],
                sign_penalty_weight=h_params["sign_penalty_weight"]
            )
        if binary_losses:
            epoch_binary_losses.append([bl.item() for bl in binary_losses])
        if regression_losses:
            epoch_regression_losses.append([rl.item() for rl in regression_losses])
        batch_accuracy, batch_f1_score, batch_per_label_acc, batch_per_label_f1 = calculate_accuracy_and_f1(preds, batch_y)
        batch_smape = calculate_smape(preds, batch_y)
        batch_returns_sign_accuracy = calculate_returns_sign_accuracy(preds, batch_y)
        for label_name in label_names:
            if label_name in batch_per_label_acc:
                epoch_per_label_accuracies[label_name].append(batch_per_label_acc[label_name])
            if label_name in batch_per_label_f1:
                epoch_per_label_f1s[label_name].append(batch_per_label_f1[label_name])
        loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % (accumulation_steps * 10) == 0:
                clear_memory()
        total_loss += loss.item() * accumulation_steps
        total_accuracy += batch_accuracy
        total_f1_score += batch_f1_score
        total_smape += batch_smape
        total_steps += 1
        total_returns_sign_accuracy += batch_returns_sign_accuracy
        # ...existing code...

    # Final step if remaining gradients
    if total_steps % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    duration = time.time() - start_time
    avg_loss = total_loss / total_steps
    avg_accuracy = total_accuracy / total_steps
    avg_f1_score = total_f1_score / total_steps
    avg_smape = total_smape / total_steps
    avg_returns_sign_accuracy = total_returns_sign_accuracy / total_steps

    # Average the loss components across batches
    avg_binary_losses = []
    avg_regression_losses = []
    
    if epoch_binary_losses:
        # Get the maximum number of components
        max_binary_components = max(len(bl) for bl in epoch_binary_losses)
        for i in range(max_binary_components):
            component_losses = [bl[i] for bl in epoch_binary_losses if i < len(bl)]
            if component_losses:
                avg_binary_losses.append(sum(component_losses) / len(component_losses))
    
    if epoch_regression_losses:
        # Get the maximum number of components
        max_regression_components = max(len(rl) for rl in epoch_regression_losses)
        for i in range(max_regression_components):
            component_losses = [rl[i] for rl in epoch_regression_losses if i < len(rl)]
            if component_losses:
                avg_regression_losses.append(sum(component_losses) / len(component_losses))
    
    # Average per-label accuracies and F1s
    avg_per_label_accuracies = {}
    avg_per_label_f1s = {}
    for label_name in label_names:
        if epoch_per_label_accuracies[label_name]:
            avg_per_label_accuracies[label_name] = sum(epoch_per_label_accuracies[label_name]) / len(epoch_per_label_accuracies[label_name])
        else:
            avg_per_label_accuracies[label_name] = 0.0
        if epoch_per_label_f1s[label_name]:
            avg_per_label_f1s[label_name] = sum(epoch_per_label_f1s[label_name]) / len(epoch_per_label_f1s[label_name])
        else:
            avg_per_label_f1s[label_name] = 0.0
    # ...existing code...
    # Clear memory after training epoch (less aggressive)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_loss, avg_accuracy, avg_f1_score, avg_smape, avg_binary_losses, avg_regression_losses, avg_per_label_accuracies, avg_per_label_f1s, avg_returns_sign_accuracy

@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_f1_score = 0
    total_smape = 0
    total_steps = 0
    total_returns_sign_accuracy = 0

    # Track loss components
    epoch_binary_losses = []
    epoch_regression_losses = []
    
    # Track per-label accuracies and F1s
    label_names = ['ranking', 'risk']
    epoch_per_label_accuracies = {name: [] for name in label_names}
    epoch_per_label_f1s = {name: [] for name in label_names}
    # ...existing code...

    for batch_x, batch_y in tqdm(dataloader, desc="Validation"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast('cuda'):
            preds = model(batch_x)
            loss, binary_losses, regression_losses = mixed_loss(
                preds, batch_y,
                regression_loss_weight=h_params["regression_loss_weight"],
                huber_delta=h_params["huber_delta"],
                huber_weight=h_params["huber_weight"],
                sign_penalty_weight=h_params["sign_penalty_weight"]
            )
        if binary_losses:
            epoch_binary_losses.append([bl.item() for bl in binary_losses])
        if regression_losses:
            epoch_regression_losses.append([rl.item() for rl in regression_losses])
        batch_accuracy, batch_f1_score, batch_per_label_acc, batch_per_label_f1 = calculate_accuracy_and_f1(preds, batch_y)
        batch_smape = calculate_smape(preds, batch_y)
        batch_returns_sign_accuracy = calculate_returns_sign_accuracy(preds, batch_y)
        for label_name in label_names:
            if label_name in batch_per_label_acc:
                epoch_per_label_accuracies[label_name].append(batch_per_label_acc[label_name])
            if label_name in batch_per_label_f1:
                epoch_per_label_f1s[label_name].append(batch_per_label_f1[label_name])
        total_loss += loss.item()
        total_accuracy += batch_accuracy
        total_f1_score += batch_f1_score
        total_smape += batch_smape
        total_steps += 1
        total_returns_sign_accuracy += batch_returns_sign_accuracy

    avg_loss = total_loss / total_steps
    avg_accuracy = total_accuracy / total_steps
    avg_f1_score = total_f1_score / total_steps
    avg_smape = total_smape / total_steps
    avg_returns_sign_accuracy = total_returns_sign_accuracy / total_steps

    # Average the loss components across batches
    avg_binary_losses = []
    avg_regression_losses = []
    
    if epoch_binary_losses:
        # Get the maximum number of components
        max_binary_components = max(len(bl) for bl in epoch_binary_losses)
        for i in range(max_binary_components):
            component_losses = [bl[i] for bl in epoch_binary_losses if i < len(bl)]
            if component_losses:
                avg_binary_losses.append(sum(component_losses) / len(component_losses))
    
    if epoch_regression_losses:
        # Get the maximum number of components
        max_regression_components = max(len(rl) for rl in epoch_regression_losses)
        for i in range(max_regression_components):
            component_losses = [rl[i] for rl in epoch_regression_losses if i < len(rl)]
            if component_losses:
                avg_regression_losses.append(sum(component_losses) / len(component_losses))
    
    # Average per-label accuracies and F1s
    avg_per_label_accuracies = {}
    avg_per_label_f1s = {}
    for label_name in label_names:
        if epoch_per_label_accuracies[label_name]:
            avg_per_label_accuracies[label_name] = sum(epoch_per_label_accuracies[label_name]) / len(epoch_per_label_accuracies[label_name])
        else:
            avg_per_label_accuracies[label_name] = 0.0
        if epoch_per_label_f1s[label_name]:
            avg_per_label_f1s[label_name] = sum(epoch_per_label_f1s[label_name]) / len(epoch_per_label_f1s[label_name])
        else:
            avg_per_label_f1s[label_name] = 0.0
    # ...existing code...
    # Clear memory after validation (less aggressive)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_loss, avg_accuracy, avg_f1_score, avg_smape, avg_binary_losses, avg_regression_losses, avg_per_label_accuracies, avg_per_label_f1s, avg_returns_sign_accuracy

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, train_smape, val_smape, train_sign_acc, val_sign_acc,
                   train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_smapes, val_smapes, train_sign_accuracies, val_sign_accuracies,
                   train_binary_losses, val_binary_losses, train_regression_losses, val_regression_losses, train_per_label_accuracies, val_per_label_accuracies,
                   train_per_label_f1s, val_per_label_f1s,
                   filename):
    """Save training checkpoint with full metric histories"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_smape': train_smape,
        'val_smape': val_smape,
        'train_sign_acc': train_sign_acc,
        'val_sign_acc': val_sign_acc,
        # Full histories:
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_smapes': train_smapes,
        'val_smapes': val_smapes,
        'train_sign_accuracies': train_sign_accuracies,
        'val_sign_accuracies': val_sign_accuracies,
        'train_binary_losses': train_binary_losses,
        'val_binary_losses': val_binary_losses,
        'train_regression_losses': train_regression_losses,
        'val_regression_losses': val_regression_losses,
        'train_per_label_accuracies': train_per_label_accuracies,
        'val_per_label_accuracies': val_per_label_accuracies,
        'train_per_label_f1s': train_per_label_f1s,
        'val_per_label_f1s': val_per_label_f1s,
        'h_params': h_params
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    """Load training checkpoint and restore metric histories"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    # ...existing code...
    # Restore metric lists if present
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    train_f1_scores = checkpoint.get('train_f1_scores', [])
    val_f1_scores = checkpoint.get('val_f1_scores', [])
    train_smapes = checkpoint.get('train_smapes', [])
    val_smapes = checkpoint.get('val_smapes', [])
    train_sign_accuracies = checkpoint.get('train_sign_accuracies', [])
    val_sign_accuracies = checkpoint.get('val_sign_accuracies', [])
    train_binary_losses = checkpoint.get('train_binary_losses', [])
    val_binary_losses = checkpoint.get('val_binary_losses', [])
    train_regression_losses = checkpoint.get('train_regression_losses', [])
    val_regression_losses = checkpoint.get('val_regression_losses', [])
    train_per_label_accuracies = checkpoint.get('train_per_label_accuracies', {'top_or_bottom': [], 'risk': []})
    val_per_label_accuracies = checkpoint.get('val_per_label_accuracies', {'top_or_bottom': [], 'risk': []})
    train_per_label_f1s = checkpoint.get('train_per_label_f1s', {'ranking': [], 'risk': []})
    val_per_label_f1s = checkpoint.get('val_per_label_f1s', {'ranking': [], 'risk': []})
    train_acc = checkpoint.get('train_acc', 0.0)
    val_acc = checkpoint.get('val_acc', 0.0)
    return (start_epoch, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_smapes, val_smapes,
            train_binary_losses, val_binary_losses, train_regression_losses, val_regression_losses,
            train_per_label_accuracies, val_per_label_accuracies, train_per_label_f1s, val_per_label_f1s, train_sign_accuracies, val_sign_accuracies, checkpoint['train_loss'], checkpoint['val_loss'], train_acc, val_acc)

def monitor_gpu_memory(tag=""):
    """Monitor GPU memory usage efficiently"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0


def training(train_split = 0.25,val_split = 0.1, test_split = 0.1, dataset_file="dataset.h5",model_filename_prefix="100_stocks"):

    # Set CUDA memory allocation config for expandable segments (for PyTorch >= 2.0)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else 
                        "mps" if torch.mps.is_available() else
                        "cpu")
    # print(f"Using device: {device}")

    # Load cached datasets


    # Example: using a single HDF5 file for train/val/test split
    # Set exclude_abs_close=True to remove the absolute final close feature
    dataset = HDF5SequenceDataset(dataset_file, exclude_abs_close=True)
    total_len = dataset.__len__()
    if total_len is None:
        raise ValueError("Could not determine dataset length")

    train_split_end = int(train_split * total_len)
    val_split_end = train_split_end + int(val_split * total_len)
    test_split_end = val_split_end + int(test_split * total_len)

    # print(f"split: {train_split_end}")
    train_dataset = HDF5SequenceDataset(dataset_file, start=0, stop=train_split_end, exclude_abs_close=True)
    val_dataset = HDF5SequenceDataset(dataset_file, start=train_split_end, stop=val_split_end, exclude_abs_close=True)
    test_dataset = HDF5SequenceDataset(dataset_file, start=val_split_end, stop=test_split_end, exclude_abs_close=True)

    train_loader = DataLoader(train_dataset, batch_size=h_params["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=h_params["batch_size"], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=h_params["batch_size"], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Inspect a sample to get dataset dimensions
    x, y = train_dataset[0]
    # print("x.shape =", x.shape)  # (seq_len, num_stocks, features+nanmask) - abs_final_close excluded
    # print("y.shape =", y.shape)  # (num_stocks, num_labels)

    # Extract dimensions from the dataset
    seq_len = x.shape[0]
    num_stocks = x.shape[1] 
    num_features = x.shape[2]

    # print(f"📊 Dataset info: seq_len={seq_len}, num_stocks={num_stocks}, num_features={num_features}")

    # Adjust num_features and num_stocks as needed based on your HDF5 data
    model = FORTUNETransformer(
        num_features=num_features,
        d_model=h_params["d_model"],
        num_layers=h_params["num_layers"],
        num_stocks=num_stocks,
        seq_len=seq_len,  
        horizons=h_params["horizons"],
        nhead=h_params["nhead"],
        chunk_size=h_params["chunk_size"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=h_params["learning_rate"], weight_decay=h_params.get("weight_decay", 1e-2))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    epochs = h_params["epochs"]
    
    # Check if final model already exists
    final_model_file = f"{model_filename_prefix}_final.pt"
    
    if os.path.exists(final_model_file):
        # print(f"🎯 Final model found: {final_model_file}")
        skip_training = input("Skip training and go straight to test evaluation? (y/n): ")
        
        if skip_training.lower() == 'y':
            # print("⏭️  Skipping training, loading final model for test evaluation...")
            
            # Load the final model
            model.load_state_dict(torch.load(final_model_file))
            model.eval()
            
            # Evaluate on test dataset
            # print(f"\n🧪 Evaluating final model on test dataset...")
            test_loss, test_acc, test_f1, test_smape, _, _, _, test_sign_acc = eval_epoch(model, test_loader, device)
            # print(f"🎯 Test Results: Loss = {test_loss:.4f}, Acc = {test_acc:.4f}, F1 = {test_f1:.4f}, SMAPE = {test_smape:.2f}%, , SIGN ACC = {test_sign_acc:.4f}")
            
            # Try to load and display previous training metrics if available
            metrics_file = f"{model_filename_prefix}_training_metrics.pt"
            if os.path.exists(metrics_file):
                # print(f"\n📊 Loading previous training metrics...")
                try:
                    metrics = torch.load(metrics_file)
                    train_losses = metrics.get('train_losses', [])
                    val_losses = metrics.get('val_losses', [])
                    best_val_loss = metrics.get('best_val_loss', 'N/A')
                    best_val_epoch = metrics.get('best_val_epoch', 'N/A')
                    # Patch: ensure per-label accuracy dicts have 'ranking' and 'risk' keys
                    train_per_label_accuracies = metrics.get('train_per_label_accuracies', {})
                    val_per_label_accuracies = metrics.get('val_per_label_accuracies', {})
                    for d in (train_per_label_accuracies, val_per_label_accuracies):
                        if 'ranking' not in d:
                            d['ranking'] = [0.0] * len(train_losses)
                        if 'risk' not in d:
                            d['risk'] = [0.0] * len(train_losses)
                    # ...existing code for plotting...
                    if train_losses and val_losses:
                        show_plot = input("Show previous training plot? (y/n): ")
                        if show_plot.lower() == 'y':
                            train_accuracies = metrics.get('train_accuracies', [0] * len(train_losses))
                            val_accuracies = metrics.get('val_accuracies', [0] * len(val_losses))
                            create_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, 
                                                save_path=None, title_prefix="Previous Training")
                            train_binary_losses = metrics.get('train_binary_losses', [])
                            val_binary_losses = metrics.get('val_binary_losses', [])
                            train_regression_losses = metrics.get('train_regression_losses', [])
                            val_regression_losses = metrics.get('val_regression_losses', [])
                            if train_binary_losses or train_regression_losses:
                                show_components = input("Show loss components plot? (y/n): ")
                                if show_components.lower() == 'y':
                                    create_loss_components_plot(train_binary_losses, val_binary_losses, 
                                                               train_regression_losses, val_regression_losses,
                                                               save_path=None, title_prefix="Previous Training")
                            train_f1_scores = metrics.get('train_f1_scores', [])
                            val_f1_scores = metrics.get('val_f1_scores', [])
                            train_smapes = metrics.get('train_smapes', [])
                            val_smapes = metrics.get('val_smapes', [])
                            if train_f1_scores and train_smapes:
                                show_metrics = input("Show F1 and SMAPE metrics plot? (y/n): ")
                                if show_metrics.lower() == 'y':
                                    create_metrics_plot(train_f1_scores, val_f1_scores, train_smapes, val_smapes,
                                                       save_path=None, title_prefix="Previous Training")
                            if train_per_label_accuracies and any(train_per_label_accuracies.values()):
                                show_per_label = input("Show per-label accuracy plot? (y/n): ")
                                if show_per_label.lower() == 'y':
                                    create_per_label_accuracy_plot(train_per_label_accuracies, val_per_label_accuracies,
                                                                 save_path=None, title_prefix="Previous Training")
                                    print("📊 Per-label accuracy plot displayed (not saved)")
                except Exception as e:
                    print(f"❌ Error loading metrics: {e}")
            else:
                print(f"📊 No previous training metrics found.")
            
            print("✅ Test evaluation completed!")
            exit(0)  # Exit the program
    
    # Continue with training if final model doesn't exist or user chose to retrain
    print("🚀 Starting training process...")
    
    # Check if we should resume from checkpoint
    checkpoint_file = f"{model_filename_prefix}_checkpoint.pt"
    start_epoch = 0
    
    # Initialize metric tracking lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_smapes = []
    val_smapes = []
    train_binary_losses = []
    val_binary_losses = []
    train_regression_losses = []
    val_regression_losses = []
    train_sign_accuracies = []
    val_sign_accuracies = []

    # Per-label accuracy and F1 tracking
    train_per_label_accuracies = {'ranking': [], 'risk': []}
    val_per_label_accuracies = {'ranking': [], 'risk': []}
    train_per_label_f1s = {'ranking': [], 'risk': []}
    val_per_label_f1s = {'ranking': [], 'risk': []}
    
    initial_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if os.path.exists(checkpoint_file):
        resume = input(f"Found checkpoint {checkpoint_file}. Resume training? (y/n): ")
        if resume.lower() == 'y':
            # Unpack all returned values
            checkpoint_tuple = load_checkpoint(checkpoint_file, model, optimizer)
            start_epoch = checkpoint_tuple[0]
            train_losses = checkpoint_tuple[1]
            val_losses = checkpoint_tuple[2]
            train_accuracies = checkpoint_tuple[3]
            val_accuracies = checkpoint_tuple[4]
            train_f1_scores = checkpoint_tuple[5]
            val_f1_scores = checkpoint_tuple[6]
            train_smapes = checkpoint_tuple[7]
            val_smapes = checkpoint_tuple[8]
            train_binary_losses = checkpoint_tuple[9]
            val_binary_losses = checkpoint_tuple[10]
            train_regression_losses = checkpoint_tuple[11]
            val_regression_losses = checkpoint_tuple[12]
            train_per_label_accuracies = checkpoint_tuple[13]
            val_per_label_accuracies = checkpoint_tuple[14]
            train_per_label_f1s = checkpoint_tuple[15]
            val_per_label_f1s = checkpoint_tuple[16]
            train_sign_accuracies = checkpoint_tuple[17]
            val_sign_accuracies = checkpoint_tuple[18]
            last_train_loss = checkpoint_tuple[-4]
            last_val_loss = checkpoint_tuple[-3]
            last_train_acc = checkpoint_tuple[-2]
            last_val_acc = checkpoint_tuple[-1]
            print(f"Previous: Train Loss = {last_train_loss:.4f}, Val Loss = {last_val_loss:.4f}")
            print(f"Previous: Train Acc = {last_train_acc:.4f}, Val Acc = {last_val_acc:.4f}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs):
        # ...existing code...
        train_loss, train_acc, train_f1, train_smape, train_binary_comp, train_regression_comp, train_per_label_acc, train_per_label_f1, train_sign_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, val_smape, val_binary_comp, val_regression_comp, val_per_label_acc, val_per_label_f1, val_sign_acc = eval_epoch(model, val_loader, device)
        # Step the scheduler using the current epoch (for CosineAnnealingWarmRestarts)
        scheduler.step(epoch + 1)
        # ...existing code...

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_smapes.append(train_smape)
        val_smapes.append(val_smape)
        train_binary_losses.append(train_binary_comp)
        val_binary_losses.append(val_binary_comp)
        train_regression_losses.append(train_regression_comp)
        val_regression_losses.append(val_regression_comp)
        train_sign_accuracies.append(train_sign_acc)
        val_sign_accuracies.append(val_sign_acc)

        # Store per-label accuracies and F1s
        for label_name in ['ranking', 'risk']:
            train_val = train_per_label_acc.get(label_name, None)
            val_val = val_per_label_acc.get(label_name, None)
            train_f1_val = train_per_label_f1.get(label_name, None)
            val_f1_val = val_per_label_f1.get(label_name, None)
            train_per_label_accuracies[label_name].append(train_val if train_val is not None else None)
            val_per_label_accuracies[label_name].append(val_val if val_val is not None else None)
            train_per_label_f1s[label_name].append(train_f1_val if train_f1_val is not None else None)
            val_per_label_f1s[label_name].append(val_f1_val if val_f1_val is not None else None)

        print(f"✅ Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        for label in ['ranking', 'risk']:
            print(f"   Train Acc ({label}) = {train_per_label_accuracies[label][-1]:.4f}, Val Acc ({label}) = {val_per_label_accuracies[label][-1]:.4f}")
            print(f"   Train F1  ({label}) = {train_per_label_f1s[label][-1]:.4f}, Val F1  ({label}) = {val_per_label_f1s[label][-1]:.4f}")
        print(f"✅ Epoch {epoch+1}: Train SMAPE = {train_smape:.2f}%, Val SMAPE = {val_smape:.2f}%")
        print(f"✅ Epoch {epoch+1}: Train SIGN ACC = {train_sign_acc:.2f}%, Val SIGN ACC = {val_sign_acc:.2f}%")

        # Save checkpoint after each epoch
        save_checkpoint(
            epoch, model, optimizer, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, train_smape, val_smape, train_sign_acc, val_sign_acc,
            train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_smapes, val_smapes, train_sign_accuracies, val_sign_accuracies,
            train_binary_losses, val_binary_losses, train_regression_losses, val_regression_losses, train_per_label_accuracies, val_per_label_accuracies,
            train_per_label_f1s, val_per_label_f1s,
            checkpoint_file
        )

        # Create a unique directory for plots for this epoch: plot/epoch_{epoch}_timestamp
        plot_dir = os.path.join('plot', f'{initial_timestamp}', f'epoch_{epoch+1}')
        os.makedirs(plot_dir, exist_ok=True)

        try:
            from plot_from_checkpoint import plot_from_checkpoint
            plot_from_checkpoint(checkpoint_file, outdir=plot_dir, prefix=f'epoch{epoch+1}')
        except Exception:
            pass

        # Also save best model if validation loss improves
        if epoch == start_epoch or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_filename_prefix}_best.pt")
            print(f"🏆 New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save final model (which is the best model)
    print("🎯 Training completed!")
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(f"{model_filename_prefix}_best.pt"))
    model.eval()
    
    # Save the best model as the final model
    torch.save(model.state_dict(), f"{model_filename_prefix}_final.pt")
    print("💾 Final model saved (best validation loss model).")
    
    # Evaluate the best model on test dataset
    print(f"\n🧪 Evaluating final model (best validation) on test dataset...")
    test_loss, test_acc, test_f1, test_smape, _, _, _, test_sign_acc = eval_epoch(model, test_loader, device)
    print(f"🎯 Final Test Results: Loss = {test_loss:.4f}, Acc = {test_acc:.4f}, F1 = {test_f1:.4f}, SMAPE = {test_smape:.2f}%, SIGN ACC = {test_sign_acc:.4f}")
    
    # Plot training metrics
    if len(train_losses) > 0:
        # Ensure finalplot directory exists
        os.makedirs("finalplot", exist_ok=True)
        create_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, 
                            save_path=f"finalplot/training_metrics.png")

        # Plot F1 and SMAPE metrics
        if len(train_f1_scores) > 0 and len(train_smapes) > 0:
            create_metrics_plot(train_f1_scores, val_f1_scores, train_smapes, val_smapes,
                               save_path=f"finalplot/f1_smape_metrics.png")

        # Plot per-label accuracies
        if train_per_label_accuracies and any(train_per_label_accuracies.values()):
            create_per_label_accuracy_plot(train_per_label_accuracies, val_per_label_accuracies,
                                         save_path=f"finalplot/per_label_accuracy.png")

        # Also plot loss components if available
        if train_binary_losses or train_regression_losses:
            create_loss_components_plot(train_binary_losses, val_binary_losses, 
                                       train_regression_losses, val_regression_losses,
                                       save_path=f"finalplot/loss_components.png")

        if train_sign_accuracies:
            create_sign_accuracy_plot(train_sign_accuracies,
                                      val_sign_accuracies, 
                                      save_path=f"finalplot/returns_sign_accuracy.png")
        
        # Plot per-label F1 scores
        if train_per_label_f1s and any(train_per_label_f1s.values()):
            create_per_label_f1_plot(
                train_per_label_f1s, val_per_label_f1s,
                save_path=f"finalplot/per_label_f1.png"
            )
        
        # Also save metrics to file including test results
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_f1_scores': train_f1_scores,
            'val_f1_scores': val_f1_scores,
            'train_smapes': train_smapes,
            'val_smapes': val_smapes,
            'train_binary_losses': train_binary_losses,
            'val_binary_losses': val_binary_losses,
            'train_regression_losses': train_regression_losses,
            'val_regression_losses': val_regression_losses,
            'train_per_label_accuracies': train_per_label_accuracies,
            'val_per_label_accuracies': val_per_label_accuracies,
            'train_per_label_f1s': train_per_label_f1s,
            'val_per_label_f1s': val_per_label_f1s,
            'train_sign_accuracies': train_sign_accuracies,
            'val_sign_accuracies': val_sign_accuracies,
            'final_test_loss': test_loss,
            'final_test_acc': test_acc,
            'final_test_f1': test_f1,
            'final_test_smape': test_smape,
            'final_test_sign_acc': test_sign_acc,
            'best_val_loss': best_val_loss,
            'best_val_epoch': val_losses.index(min(val_losses)) + 1
        }
        torch.save(metrics, f"{model_filename_prefix}_training_metrics.pt")
        print(f"📊 Training metrics saved to {model_filename_prefix}_training_metrics.pt")
        m
        # Print final summary
        print(f"\n{'='*60}")
        print(f"📊 FINAL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"📈 Training:   Final Loss = {train_losses[-1]:.4f}, Final Accuracy = {train_accuracies[-1]:.4f}")
        print(f"📈 Training:   Final F1 = {train_f1_scores[-1]:.4f}, Final SMAPE = {train_smapes[-1]:.2f}%")
        print(f"📈 Training:   Final SIGN ACC= {train_sign_accuracies[-1]:.4f}")
        print(f"📊 Validation: Final Loss = {val_losses[-1]:.4f}, Final Accuracy = {val_accuracies[-1]:.4f}")
        print(f"📊 Validation: Final F1 = {val_f1_scores[-1]:.4f}, Final SMAPE = {val_smapes[-1]:.2f}%")
        print(f"📊 Validation: Final SIGN ACC= {val_sign_accuracies[-1]:.4f}")

        # Print final per-label accuracies
        final_train_per_label = ", ".join([f"{label}: {train_per_label_accuracies[label][-1]:.3f}" for label in ['ranking', 'risk']])
        final_val_per_label = ", ".join([f"{label}: {val_per_label_accuracies[label][-1]:.3f}" for label in ['ranking', 'risk']])
        print(f"📈 Training   Per-label Acc: {final_train_per_label}")
        print(f"📊 Validation Per-label Acc: {final_val_per_label}")
        
        print(f"🏆 Best Val Loss: {best_val_loss:.4f} (epoch {val_losses.index(best_val_loss) + 1})")
        print(f"🎯 Test (Final/Best Model): Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")
        print(f"🎯 Test (Final/Best Model): F1 = {test_f1:.4f}, SMAPE = {test_smape:.2f}%")
        print(f"🎯 Test (Final/Best Model): SIGN ACC = {test_sign_acc:.4f}")

        print(f"{'='*60}")

if __name__ == "__main__":
    training(train_split = 0.2,dataset_file = '0_to_end_stride_185_seqlen_390_stocks_100_with_close_fixed.h5', model_filename_prefix="last_one")