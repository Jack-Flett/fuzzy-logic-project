"""Loading and leakage-free preprocessing of the UCI Water Treatment Plant dataset.

Every statistic used for preprocessing (medians, IQR fences, feature ranking,
min/max) is computed on the TRAINING part only and then applied to the
validation and test parts.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "water-treatment.data"

COLUMNS = ("Q-E ZN-E PH-E DBO-E DQO-E SS-E SSV-E SED-E COND-E "
           "PH-P DBO-P SS-P SSV-P SED-P COND-P "
           "PH-D DBO-D DQO-D SS-D SSV-D SED-D COND-D "
           "PH-S DBO-S DQO-S SS-S SSV-S SED-S COND-S "
           "RD-DBO-P RD-SS-P RD-SED-P RD-DBO-S RD-DQO-S RD-DBO-G RD-DQO-G RD-SS-G RD-SED-G").split()

# Candidate inputs.
#  "process":    the 22 measurements taken upstream of the effluent (plant input E,
#                primary settler P, secondary settler input D). Mirrors the paper's
#                "influent + aeration tank -> effluent BOD" set-up.
#  "softsensor": additionally the fast effluent measurements (pH, COD, solids,
#                conductivity). BOD needs a 5-day lab test, the others do not,
#                so estimating BOD from them is a classic soft-sensor task.
# The RD-* columns are removal efficiencies computed FROM the effluent values
# (RD-DBO-S and RD-DBO-G directly from the target), so they are never used.
PROCESS = COLUMNS[:22]
EFFLUENT_FAST = ["PH-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S"]
SCENARIOS = {"process": PROCESS, "softsensor": PROCESS + EFFLUENT_FAST}


def load_raw() -> pd.DataFrame:
    """Read the raw file, parse the dates and sort chronologically."""
    df = pd.read_csv(RAW, header=None, names=["date"] + COLUMNS, na_values="?", skip_blank_lines=True)
    df["date"] = pd.to_datetime(df["date"].str[2:], format="%d/%m/%y")
    df = df.sort_values("date").reset_index(drop=True)
    return df[df[C.TARGET].notna()].reset_index(drop=True)   # rows without a target cannot be used


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray      # test target scaled but NOT clipped (honesty check)
    features: list              # names of the selected inputs
    x_min: np.ndarray           # for mapping physical inputs to [0, 1] (demo)
    x_max: np.ndarray
    x_median: np.ndarray
    y_min: float
    y_max: float

    def y_to_mgl(self, y_scaled):
        """Scaled target -> mg/L."""
        return np.asarray(y_scaled) * (self.y_max - self.y_min) + self.y_min

    def x_to_scaled(self, x_physical):
        x = np.clip(np.asarray(x_physical, dtype=float), self.x_min, self.x_max)
        return ((x - self.x_min) / (self.x_max - self.x_min)).astype(np.float32)


def _split_indices(n, split, seed):
    n_tr = int(round(C.SPLIT[0] * n))
    n_va = int(round(C.SPLIT[1] * n))
    idx = np.arange(n)                       # rows are already in time order
    if split == "random":
        idx = np.random.default_rng(seed).permutation(n)
    elif split != "chrono":
        raise ValueError(split)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def prepare(seed=0, split="random", scenario="process", k=C.N_FEATURES) -> Dataset:
    """Split, impute, clip, select features and scale. Train statistics only."""
    df = load_raw()
    cols = SCENARIOS[scenario]
    X = df[cols].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy(dtype=float)
    tr, va, te = _split_indices(len(df), split, seed)

    # 1) median imputation
    med = np.nanmedian(X[tr], axis=0)
    X = np.where(np.isnan(X), med, X)

    # 2) IQR clipping (replaces the paper's AutoGAN anomaly handling)
    def fences(a):
        q1, q3 = np.percentile(a, [25, 75], axis=0)
        return q1 - C.IQR_K * (q3 - q1), q3 + C.IQR_K * (q3 - q1)
    lo, hi = fences(X[tr])
    X = np.clip(X, lo, hi)
    y_lo, y_hi = fences(y[tr])
    y_clip = np.clip(y, y_lo, y_hi)

    # 3) feature selection: k inputs with the largest |Spearman rho| to the target
    #    (replaces the paper's AutoGAN importance score, whose formula is not given)
    rho = np.array([abs(pd.Series(X[tr, j]).corr(pd.Series(y_clip[tr]), method="spearman"))
                    for j in range(X.shape[1])])
    rho = np.nan_to_num(rho)
    keep = np.sort(np.argsort(-rho)[:k])
    X, lo, hi, med = X[:, keep], lo[keep], hi[keep], med[keep]

    # 4) min-max scaling to [0, 1]
    x_min, x_max = X[tr].min(axis=0), X[tr].max(axis=0)
    x_max = np.where(x_max > x_min, x_max, x_min + 1.0)
    # values outside the training range (possible in val/test) are clipped to [0, 1]
    Xs = np.clip((X - x_min) / (x_max - x_min), 0.0, 1.0).astype(np.float32)
    y_min, y_max = float(y_clip[tr].min()), float(y_clip[tr].max())
    ys = ((y_clip - y_min) / (y_max - y_min)).astype(np.float32)
    ys_raw = ((y - y_min) / (y_max - y_min)).astype(np.float32)

    return Dataset(Xs[tr], ys[tr], Xs[va], ys[va], Xs[te], ys[te], ys_raw[te],
                   [cols[j] for j in keep], x_min, x_max, med, y_min, y_max)
