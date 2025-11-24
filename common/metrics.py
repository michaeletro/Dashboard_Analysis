from __future__ import annotations

import numpy as np


def nmise_mape(y_true, y_pred):
    """
    Compute NMISE and MAPE between truth and prediction vectors.

    NMISE = mean( (y_pred - y_true)**2 ) / mean( y_true**2 )
    MAPE  = mean( |y_pred - y_true| / |y_true| ) * 100
            computed only on entries where y_true != 0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match, got {y_true.shape} and {y_pred.shape}")

    mse = np.mean((y_pred - y_true) ** 2)
    denom = np.mean(y_true ** 2)
    nmise = np.nan if denom == 0 else mse / denom

    mask = y_true != 0
    if not np.any(mask):
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0

    return float(nmise), float(mape)
