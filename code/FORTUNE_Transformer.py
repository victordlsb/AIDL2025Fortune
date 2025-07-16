import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --------------------------
# Memory Monitoring Function
# --------------------------
# Disable memory logging for better performance
def log_memory_usage(stage="", device=None, verbose=False):
    """Disabled memory logging for performance"""
    return 0, 0

# Original memory logging function (commented out for performance)
def log_memory_usage_original(stage="", device=None, verbose=False):
    """Log current memory usage - set verbose=True to see output"""
    try:
        import psutil
        # System RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        
        memory_info = f"[{stage}] RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({ram_percent:.1f}%)"
        
        # GPU memory if available
        if torch.cuda.is_available() and device is not None and device.type == 'cuda':
            gpu_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            memory_info += f" | GPU: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved"
        
        if verbose:
            pass
        return ram_used_gb, gpu_allocated if torch.cuda.is_available() and device is not None else 0
    except ImportError:
        # Fallback if psutil not available
        if torch.cuda.is_available() and device is not None and device.type == 'cuda':
            gpu_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            if verbose:
                pass
            return 0, gpu_allocated
        else:
            if verbose:
                pass
            return 0, 0

# --------------------------
# Positional Encoding
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# --------------------------
# Attention Pooling Module
# --------------------------
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, S, d_model)
        scores = self.attn_fc(x)  # (B, T, S, 1)
        weights = torch.softmax(scores, dim=1)  # softmax over time (T)
        pooled = (x * weights).sum(dim=1)  # weighted sum over T → (B, S, d_model)
        return pooled

# --------------------------
# Main Model
# --------------------------
class FORTUNETransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, seq_len=900, num_stocks=500, horizons=None, chunk_size=20):
        super().__init__()
        self.d_model = d_model
        self.num_stocks = num_stocks
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.horizons = horizons or {
            '1h':   {"volatility_threshold": 0.02},
            '1d':   {"volatility_threshold": 0.03}
        }
        self.feature_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.stock_embedding = nn.Embedding(num_stocks, d_model)

        # MEMORY-EFFICIENT TRANSFORMER with gradient checkpointing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dropout=0.1,
            activation='gelu'  # More efficient than relu
        )

        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = AttentionPooling(d_model)

        # Cross-stock attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)

        # Output heads - separate heads for ranking (5 classes), risk (binary), and return (regression)
        self.heads = nn.ModuleDict({
            h: nn.ModuleDict({
                'ranking': nn.Linear(d_model, 5),  # 5 classes for multiclass classification
                'risk': nn.Linear(d_model, 1),     # binary
                'return': nn.Linear(d_model, 1)    # regression
            }) for h in self.horizons
        })
        # Initialize output heads for financial data
        self._init_output_heads()

    def forward(self, x):  # x: (B, T, S, F)
        B, T, S, F = x.shape
        device = x.device
        
        # Initial memory check
        log_memory_usage("Forward Start", device)

        x = self.feature_embedding(x)  # (B, T, S, d_model)
        log_memory_usage("After Embedding", device)

        # Add stock embeddings
        stock_ids = torch.arange(S).to(x.device)  # (S,)
        stock_embed = self.stock_embedding(stock_ids)  # (S, d_model)
        stock_embed = stock_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, S, d_model)
        x = x + stock_embed  # (B, T, S, d_model)
        log_memory_usage("After Stock Embeddings", device)

        # MEMORY-EFFICIENT TRANSFORMER APPROACH
        # Process stocks in manageable chunks to control memory usage
        
       
        stock_outputs = []
        
        import torch.utils.checkpoint as cp
        for start_idx in range(0, S, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, S)
            current_chunk_size = end_idx - start_idx
            log_memory_usage(f"Chunk {start_idx} Start", device)
            x_chunk = x[:, :, start_idx:end_idx, :]
            x_reshaped = x_chunk.permute(0, 2, 1, 3).reshape(B * current_chunk_size, T, self.d_model)
            log_memory_usage(f"Chunk {start_idx} After Reshape", device)
            x_pos = self.pos_encoder(x_reshaped)
            log_memory_usage(f"Chunk {start_idx} After Pos Encoding", device)
            # Gradient checkpointing for transformer (wrap in lambda for compatibility)
            x_transformed = cp.checkpoint(lambda _x: self.temporal_transformer(_x), x_pos, use_reentrant=False)
            log_memory_usage(f"Chunk {start_idx} After Transformer (checkpointed)", device)
            x_back = x_transformed.reshape(B, current_chunk_size, T, self.d_model).permute(0, 2, 1, 3)
            stock_outputs.append(x_back)
            del x_chunk, x_reshaped, x_pos, x_transformed, x_back
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_memory_usage(f"Chunk {start_idx} After Cleanup", device)
        
        # Concatenate all chunks: (B, T, S, d_model)
        log_memory_usage("Before Concatenation", device)
        x = torch.cat(stock_outputs, dim=2)
        
        # Free the chunk list immediately after concatenation
        del stock_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_memory_usage("After Concatenation", device)
        
        # Apply attention pooling over time dimension first
        x = self.pool(x)  # (B, S, d_model)
        log_memory_usage("After Attention Pooling", device)
        
        # Apply cross attention between stocks on pooled representations
        x_cross, _ = self.cross_attn(x, x, x)  # Self-attention across stocks
        x = x + x_cross  # Residual connection
        log_memory_usage("After Cross Attention", device)
        
        # Predict per horizon and concatenate
        preds = []
        for h in self.horizons:
            out = self.heads[h]
            ranking_logits = out['ranking'](x)  # (B, S, 5) - 5 class logits
            risk_logit = out['risk'](x)         # (B, S, 1)
            return_val = out['return'](x)       # (B, S, 1)
            preds.append(torch.cat([ranking_logits, risk_logit, return_val], dim=-1))  # (B, S, 7)
        preds = torch.cat(preds, dim=-1)  # (B, S, 7 * num_horizons)
        log_memory_usage("Forward End", device)
        return preds

    def _init_output_heads(self):
        """Initialize output heads for financial data - small weights for stability"""
        for head_dict in self.heads.values():
            for head in head_dict.values():
                # Small weight initialization to prevent extreme initial predictions
                nn.init.xavier_uniform_(head.weight, gain=0.1)  # Much smaller gain
                if head.bias is not None:
                    nn.init.constant_(head.bias, 0)  # Zero bias

# --------------------------
# Focal BCE Loss for top_or_bottom
# --------------------------
def focal_bce_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal Binary Cross-Entropy Loss for binary classification

    logits: raw outputs (before sigmoid)
    targets: binary ground truth (0 or 1)
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
    loss = alpha * (1 - pt) ** gamma * bce
    return loss

# --------------------------
# Mixed Loss Function for Binary + Regression
# --------------------------
def huber_loss(pred, target, delta=0.01):
    """
    Huber loss - robust to outliers, quadratic for small errors, linear for large errors
    
    Args:
        pred: Predictions
        target: Ground truth
        delta: Threshold for switching from quadratic to linear loss
    """
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2
    return torch.where(abs_error <= delta, quadratic, linear)

def huber_with_sign_loss(pred, target, delta = 0.01, huber_weight = 1.0, sign_penalty_weight = 0.1):
    """
    Huber with sign loss - 
    
    Args:
        pred: Predictions
        target: Ground truth
        delta: Threshold for switching from quadratic to linear loss
        huber_weight: weight for Huber loss
        sign_penalty_weight: weight for the sign penalty
    """
    h_loss = huber_loss(pred, target, delta)*huber_weight
    sign_match = torch.sign(pred) == torch.sign(target)
    sign_penalty = 1 - sign_match.float()
    s_loss = sign_penalty*sign_penalty_weight
    return h_loss + s_loss
    
    
def mixed_loss(pred, target, classification_indices=[0, 1], regression_indices=[2], classification_loss_weight=[1.0,1.0], regression_loss_weight=1.0, huber_delta=0.01, huber_weight = 1, sign_penalty_weight = 0.1, batch_idx=None):
    """
    Mixed loss for binary classification + regression targets with equal weighting per label.
    Uses Huber loss for regression (robust to outliers).
    
    New label structure (3 labels per horizon):
    - Index 0: ranking (multiclass: 5 classes, integer target)
    - Index 1: risk (binary: 1.0=high risk, 0.0=low risk)
    - Index 2: future_return (regression: float)

    Args:
        pred: Model predictions (B, S, num_outputs)
        target: Ground truth targets (B, S, num_outputs)
        binary_indices: Indices of binary targets (e.g., [0,1])
        regression_indices: Indices of regression targets (e.g., [2])
        huber_delta: Threshold for Huber loss (default: 0.01 for 1% return threshold)
    """
    binary_losses = []
    regression_losses = []
    batch_info = f"[mixed_loss] Batch idx: {batch_idx}" if batch_idx is not None else "[mixed_loss]"

    # Handle classification targets
    if classification_indices:
        for horizon_start in range(0, pred.shape[-1], 3):  # Each horizon has 3 outputs
            for idx in classification_indices:
                target_idx = horizon_start + idx
                if target_idx < target.shape[-1]:
                    if idx == 0:
                        # Apply CrossEntropyLoss for ranking classification (5 logits, integer target)
                        pred_multi = pred[:, :, horizon_start:horizon_start+5]  # (B, S, 5)
                        target_multi_raw = target[:, :, target_idx]  # (B, S), float, may contain NaN
                        mask = ~torch.isnan(target_multi_raw)
                        if mask.any():
                            tvals = target_multi_raw[mask].long()  # Only convert to long after masking NaNs
                            out_of_bounds = (tvals < 0) | (tvals > 4)
                            if out_of_bounds.any():
                                tvals_cpu = tvals[out_of_bounds].detach().cpu().numpy()
                                idxs_cpu = torch.nonzero(out_of_bounds, as_tuple=False).detach().cpu().numpy()
                                # print(f"[mixed_loss] FATAL: Out-of-bounds values in ranking targets at idx={target_idx} (batch_idx={batch_idx}) just before CrossEntropyLoss!")
                                # print(f"  Offending values (target_multi[mask]): {tvals_cpu}")
                                # print(f"  Indices: {idxs_cpu}")
                                # print(f"  All unique values: {torch.unique(tvals).detach().cpu().numpy()}")
                                # print(f"  Min: {tvals.min().item()}, Max: {tvals.max().item()}")
                                # print(f"  Sample: {tvals.flatten()[:20].detach().cpu().numpy()}")
                                raise ValueError(f"Out-of-bounds values in ranking targets for CrossEntropyLoss at idx={target_idx} (batch_idx={batch_idx})")
                            try:
                                ce_loss = F.cross_entropy(
                                    pred_multi[mask],
                                    tvals,
                                )
                                binary_losses.append(ce_loss)
                            except Exception as e:
                                # print(f"[mixed_loss] ERROR during CrossEntropyLoss for ranking label at idx={target_idx} (batch_idx={batch_idx})")
                                # print(f"  pred_multi[mask] shape: {pred_multi[mask].shape}, dtype: {pred_multi[mask].dtype}")
                                # print(f"  target_multi[mask] shape: {tvals.shape}, dtype: {tvals.dtype}")
                                # print(f"  target_multi[mask] unique values: {torch.unique(tvals).detach().cpu().numpy()}")
                                # print(f"  target_multi[mask] min: {tvals.min().item()}, max: {tvals.max().item()}")
                                # print(f"  target_multi[mask] sample: {tvals.flatten()[:20].detach().cpu().numpy()}")
                                raise e
                    elif idx == 1:
                        pred_binary = pred[:, :, target_idx]
                        target_binary = target[:, :, target_idx]
                        # Diagnostics BEFORE masking
                        if torch.isnan(target_binary).any():
                            # print(f"{batch_info} WARNING: NaN detected in risk targets at idx={target_idx} (before mask)")
                            nan_locs = torch.nonzero(torch.isnan(target_binary), as_tuple=False)
                            # print(f"{batch_info} NaN locations: {nan_locs}")
                        unique_vals = torch.unique(target_binary[~torch.isnan(target_binary)])
                        # print(f"{batch_info} Unique values in risk targets at idx={target_idx} (before mask): {unique_vals}")
                        # Check for out-of-bounds or non-binary values before mask
                        invalid_all = ((target_binary != 0.0) & (target_binary != 1.0) & ~torch.isnan(target_binary))
                        num_invalid_all = invalid_all.sum().item()
                        if num_invalid_all > 0:
                            # print(f"{batch_info} ERROR: {num_invalid_all} invalid risk targets (not 0.0 or 1.0) out of {target_binary.numel()} at idx={target_idx} (before mask)")
                            # print(f"{batch_info} Invalid values: {target_binary[invalid_all]}")
                            # Immediately raise error to halt training and avoid CUDA assert
                            raise ValueError(f"{batch_info} Invalid risk targets detected (not 0/1, NaN) at idx={target_idx}. See above for details.")
                        mask = ~torch.isnan(target_binary)
                        if mask.any():
                            # Diagnostic: check for invalid risk targets (after mask)
                            invalid = ((target_binary[mask] != 0.0) & (target_binary[mask] != 1.0))
                            num_invalid = invalid.sum().item()
                            if num_invalid > 0:
                                print(f"{batch_info} ERROR: {num_invalid} invalid risk targets (not 0.0 or 1.0) out of {mask.sum().item()} at idx={target_idx} (after mask)")
                                print(f"{batch_info} Unique values in risk targets (after mask): {torch.unique(target_binary[mask])}")
                                # Immediately raise error to halt training and avoid CUDA assert
                                raise ValueError(f"{batch_info} Invalid risk targets detected after mask (not 0/1) at idx={target_idx}. See above for details.")
                            # Standard BCE for binary risk label (only valid 0/1)
                            valid_mask = (target_binary[mask] == 0.0) | (target_binary[mask] == 1.0)
                            if valid_mask.any():
                                try:
                                    loss = F.binary_cross_entropy_with_logits(pred_binary[mask][valid_mask], target_binary[mask][valid_mask])
                                    binary_losses.append(loss)
                                except Exception as e:
                                    print(f"{batch_info} ERROR during BCE loss computation for risk label at idx={target_idx}")
                                    print(f"pred_binary shape: {pred_binary[mask][valid_mask].shape}, target_binary shape: {target_binary[mask][valid_mask].shape}")
                                    print(f"pred_binary sample: {pred_binary[mask][valid_mask][:10]}")
                                    print(f"target_binary sample: {target_binary[mask][valid_mask][:10]}")
                                    raise e

    # Handle regression targets (Huber Loss)
    if regression_indices:
        for horizon_start in range(0, pred.shape[-1], 3):
            for idx in regression_indices:
                target_idx = horizon_start + idx
                if target_idx < target.shape[-1]:
                    pred_reg = pred[:, :, target_idx]
                    target_reg = target[:, :, target_idx]
                    mask = ~torch.isnan(target_reg)
                    if mask.any():
                        loss = huber_with_sign_loss(pred_reg[mask], target_reg[mask], delta=huber_delta, huber_weight=huber_weight, sign_penalty_weight=sign_penalty_weight).mean()  # Increased delta to 5%
                        regression_losses.append(loss)

    # Combine all losses with optional weighting per label
    if binary_losses or regression_losses:
        binary_stack = torch.stack(binary_losses) if binary_losses else torch.tensor([], device=pred.device)
        regression_stack = torch.stack(regression_losses) if regression_losses else torch.tensor([], device=pred.device)

        # Apply individual weights to each binary loss component
        if binary_losses and len(binary_stack) > 0:
            # Create weight tensor that matches the number of binary losses
            # Each horizon contributes len(binary_indices) losses, so we repeat the pattern
            num_horizons = pred.shape[-1] // 3
            weight_pattern = classification_loss_weight * num_horizons  # Repeat weights for each horizon
            weight_tensor = torch.tensor(weight_pattern[:len(binary_stack)], device=pred.device, dtype=binary_stack.dtype)
            binary_weighted = binary_stack * weight_tensor
        else:
            binary_weighted = binary_stack
        regression_weighted = regression_stack * regression_loss_weight

        all_losses = torch.cat([binary_weighted, regression_weighted])
        return all_losses.mean(), binary_losses, regression_losses
    else:
        return torch.tensor(0.0, device=pred.device), binary_losses, regression_losses

# --------------------------
# Legacy masked BCE loss (kept for compatibility)
# --------------------------
def masked_bce_loss(pred, target):
    """Legacy function - use mixed_loss instead for proper handling"""
    mask = ~torch.isnan(target)
    pred = pred[mask]
    target = target[mask]
    return F.binary_cross_entropy_with_logits(pred, target)