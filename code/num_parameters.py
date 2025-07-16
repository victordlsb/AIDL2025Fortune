from FORTUNE_Transformer import FORTUNETransformer
from parameters import h_params

num_features = 6
num_stocks = 100
seq_len = 390

model = FORTUNETransformer(
    num_features=num_features,
    d_model=h_params["d_model"],
    num_layers=h_params["num_layers"],
    num_stocks=num_stocks,
    seq_len=seq_len,
    horizons=h_params["horizons"],
    nhead=h_params["nhead"],
    chunk_size=h_params["chunk_size"]
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Detailed breakdown
print("\nDetailed parameter breakdown:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters")
    
# Group by component
print("\nGrouped by component:")
feature_emb = sum(p.numel() for n, p in model.named_parameters() if 'feature_embedding' in n)
stock_emb = sum(p.numel() for n, p in model.named_parameters() if 'stock_embedding' in n)
transformer = sum(p.numel() for n, p in model.named_parameters() if 'temporal_transformer' in n)
pool = sum(p.numel() for n, p in model.named_parameters() if 'pool' in n)
cross_attn = sum(p.numel() for n, p in model.named_parameters() if 'cross_attn' in n)
heads = sum(p.numel() for n, p in model.named_parameters() if 'heads' in n)

print(f"Feature embedding: {feature_emb:,}")
print(f"Stock embedding: {stock_emb:,}")
print(f"Transformer: {transformer:,}")
print(f"Pooling: {pool:,}")
print(f"Cross attention: {cross_attn:,}")
print(f"Output heads: {heads:,}")
print(f"Total: {feature_emb + stock_emb + transformer + pool + cross_attn + heads:,}")