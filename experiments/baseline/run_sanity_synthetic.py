"""Implementation check on a SYNTHETIC benchmark (not wastewater data, clearly labelled as such).

    python -m experiments.baseline.run_sanity_synthetic

Purpose: show that the code behaves as an ANFIS should when the data really contain a smooth
non-linear relationship and enough samples. The target is the Friedman #1 function
    y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + noise,   x6 is irrelevant,
with 1,564 samples (the size of the paper's private dataset) and 6 inputs.
Output: results/tables/sanity_synthetic_runs.csv
"""
import numpy as np

from src.baselines import fit_baselines
from src.data import Dataset
from src.experiment import run_jobs
from src.pipeline import LADDER, Run

N, NOISE_STD = 1564, 1.0


def synthetic_dataset(seed):
    rng = np.random.default_rng(seed)
    X = rng.random((N, 6))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
         + rng.normal(0, NOISE_STD, N))
    n_tr, n_va = int(0.70 * N), int(0.15 * N)
    tr, va, te = np.arange(n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, N)
    x_min, x_max = X[tr].min(0), X[tr].max(0)
    Xs = np.clip((X - x_min) / (x_max - x_min), 0, 1).astype(np.float32)
    y_min, y_max = float(y[tr].min()), float(y[tr].max())
    ys = ((y - y_min) / (y_max - y_min)).astype(np.float32)
    return Dataset(Xs[tr], ys[tr], Xs[va], ys[va], Xs[te], ys[te], ys[te], [f"x{i + 1}" for i in range(6)],
                   x_min, x_max, np.median(X[tr], axis=0), y_min, y_max)


def job(seed):
    run = Run(seed=seed, data=synthetic_dataset(seed))
    rows = [dict(seed=seed, model=name, **run.fit(spec)[2]) for name, spec in LADDER.items()]
    rows += [dict(seed=seed, model=name, **row) for name, (_, row) in fit_baselines(run).items()]
    return rows


if __name__ == "__main__":
    df = run_jobs(job, list(range(10)), "sanity_synthetic_runs.csv")
    print(df.groupby("model", sort=False).test_R2.agg(["mean", "std"]).round(3))
