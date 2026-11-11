"""Proposed extension: controlled experiments the reference paper does not contain.

    python -m experiments.proposed.run_extension            # full run (about 15 min on 2 cores)
    python -m experiments.proposed.run_extension --quick    # 2 seeds

For every repetition (random 70/15/15 split, 20 seeds, both input scenarios) it trains:
  E1 attention ablation   SOM-PSO-ANFIS with every subset of {Feature, MF, Rule} attention, plus
                          a per-sample ("dynamic") variant of the full attention
  E2 structure control    PANFIS (9+16+25 rules) against a single ANFIS with the same number of
                          rules and parameters (50 rules), single 9/16/25-rule models, a PANFIS
                          with equal fusion weights and a homogeneous 16+16+16 PANFIS
  E3 fitness              PSO fitness measured on the training set (paper) vs on the validation set
  E4 noise robustness     Gaussian noise added to the TEST inputs of already-trained models
Outputs: results/tables/extension_runs.csv and results/tables/noise_runs.csv
"""
import argparse
from itertools import combinations

import numpy as np

from src.baselines import fit_baselines
from src.experiment import run_jobs
from src.metrics import regression_metrics
from src.pipeline import Run
from src.train import predict

BASE = dict(init="som", opt="pso")
ATTENTION = {"none": ""} | {"+".join(c): "".join(c) for r in (1, 2, 3) for c in combinations("FMR", r)}

E1 = {f"att:{name}": dict(BASE, attention=letters) for name, letters in ATTENTION.items()}
E1["att:F+M+R (per-sample)"] = dict(BASE, attention="FMR", attention_mode="sample")

E2 = {
    "single 9 rules":               dict(BASE, grid=(3, 3)),
    "single 16 rules":              dict(BASE, grid=(4, 4)),
    "single 25 rules":              dict(BASE, grid=(5, 5)),
    "single 50 rules (matched)":    dict(BASE, grid=(5, 10)),
    "PANFIS 9+16+25 learned":       dict(kind="panfis"),
    "PANFIS 9+16+25 equal":         dict(kind="panfis", fusion="equal"),
    "PANFIS 16+16+16 learned":      dict(kind="panfis", grids=((4, 4), (4, 4), (4, 4))),
}
E3 = {
    "PSO fitness = train MSE (paper)": dict(BASE),
    "PSO fitness = validation MSE":    dict(init="som", opt="pso_val"),
    "no PSO (SOM only)":               dict(init="som"),
}
NOISE_MODELS = {
    "ANFIS":                    dict(init="uniform"),
    "SOM-PSO-ANFIS":            dict(BASE),
    "SOM-PSO-ANFIS-Attention":  dict(BASE, attention="FMR"),
    "SOM-PSO-PANFIS":           dict(kind="panfis"),
    "SOM-PSO-PANFIS-Attention": dict(kind="panfis", attention="FMR"),
}
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20)     # std of the noise, in units of the [0, 1] input range
NOISE_DRAWS = 10


def job(args):
    scenario, seed = args
    run = Run(seed=seed, split="random", scenario=scenario)
    d = run.data
    rows, noise_rows = [], []
    for exp, specs in (("E1 attention", E1), ("E2 structure", E2), ("E3 fitness", E3)):
        for name, spec in specs.items():
            _, _, row = run.fit(spec)
            rows.append(dict(scenario=scenario, seed=seed, experiment=exp, model=name, **row))

    predictors = {name: (lambda X, m=run.fit(spec)[0]: predict(m, X)) for name, spec in NOISE_MODELS.items()}
    for name, (model, _) in fit_baselines(run).items():
        if name != "Mean predictor":
            predictors[name] = model.predict
    rng = np.random.default_rng(10_000 + seed)
    for level in NOISE_LEVELS:
        noisy = [np.clip(d.X_test + rng.normal(0, level, d.X_test.shape), 0, 1).astype(np.float32)
                 for _ in range(NOISE_DRAWS if level > 0 else 1)]
        for name, fn in predictors.items():
            m = [regression_metrics(d.y_test, fn(X)) for X in noisy]
            noise_rows.append(dict(scenario=scenario, seed=seed, model=name, noise=level,
                                   test_R2=np.mean([x["R2"] for x in m]),
                                   test_RMSE=np.mean([x["RMSE"] for x in m])))
    return [dict(r, _table="main") for r in rows] + [dict(r, _table="noise") for r in noise_rows]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(sc, s) for sc in ("process", "softsensor") for s in range(2 if a.quick else 20)]
    suffix = "_quick" if a.quick else ""
    df = run_jobs(job, jobs, f"_extension_all{suffix}.csv", a.procs)
    from src.experiment import TABLES
    for table, out in (("main", "extension_runs"), ("noise", "noise_runs")):
        part = df[df["_table"] == table].drop(columns="_table").dropna(axis=1, how="all")
        part.to_csv(TABLES / f"{out}{suffix}.csv", index=False)
    (TABLES / f"_extension_all{suffix}.csv").unlink()
