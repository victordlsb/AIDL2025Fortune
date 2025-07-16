horizons_900 = {
    '1h':   {"volatility_threshold": 0.02},
    '1d':   {"volatility_threshold": 0.03}
}

h_params = {
    "horizons": horizons_900,
    "learning_rate": 1e-4,
    "epochs": 50,
    "batch_size": 25,
    "chunk_size": 50,
    "d_model": 128,
    "num_layers": 3,
    "nhead": 8,
    "regression_loss_weight": 1,
    "classification_loss_weight": [0, 0],
    "sign_penalty_weight": 3,
    "huber_weight": 1,
    "huber_delta": 0.02,
    "weight_decay": 5e-3
}