"""Regression metrics (paper Eq. 12 and 22-25)."""
import numpy as np


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    mse = sse / len(err)
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"MSE": mse, "RMSE": mse ** 0.5, "MAE": float(np.mean(np.abs(err))),
            "SSE": sse, "R2": 1 - sse / sst}
