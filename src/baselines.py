"""Non-fuzzy reference models, so the ANFIS numbers can be put in context."""
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


class _Mean:
    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def make_baselines(seed):
    return {
        "Mean predictor": _Mean(),
        "Ridge regression": Ridge(alpha=1.0),
        "MLP (32-16)": MLPRegressor(hidden_layer_sizes=(32, 16), learning_rate_init=0.01, max_iter=1000,
                                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
                                    random_state=seed),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed),
    }


def fit_baselines(run):
    """Returns {name: (fitted model, metrics row)} for one Run."""
    d, out = run.data, {}
    for name, model in make_baselines(run.seed).items():
        t0 = time.perf_counter()
        model.fit(d.X_train, d.y_train)
        n_par = {"Mean predictor": 1, "Ridge regression": d.X_train.shape[1] + 1}.get(name, 0)
        if name.startswith("MLP"):
            n_par = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        out[name] = (model, run.evaluate_fn(model.predict, n_par, time.perf_counter() - t0))
    return out
