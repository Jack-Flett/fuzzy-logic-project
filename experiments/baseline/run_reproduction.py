"""Reproduction experiment: the paper's model progression on the public UCI dataset.

    python -m experiments.baseline.run_reproduction            # full run (about 15 min on 2 cores)
    python -m experiments.baseline.run_reproduction --quick    # 2 seeds, for a fast check

Protocol A  'random': 20 repetitions, each with its own random 70/15/15 split and initialisation.
Protocol B  'chrono': one chronological 70/15/15 split (train on the past, test on the future),
                      10 repetitions that differ only in the random initialisation.
Both protocols are run for the two input scenarios defined in src/data.py.
Output: results/tables/reproduction_runs.csv (one row per model per repetition).
"""
import argparse

from src.baselines import fit_baselines
from src.experiment import run_jobs
from src.pipeline import LADDER, OPTIMISER_VARIANTS, Run


def job(args):
    scenario, split, seed = args
    run = Run(seed=seed, split=split, scenario=scenario)
    rows = []
    for name, spec in {**LADDER, **OPTIMISER_VARIANTS}.items():
        _, _, row = run.fit(spec)
        rows.append(dict(scenario=scenario, split=split, seed=seed, model=name, **row))
    for name, (_, row) in fit_baselines(run).items():
        rows.append(dict(scenario=scenario, split=split, seed=seed, model=name, **row))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    n_random, n_chrono = (2, 2) if a.quick else (20, 10)
    jobs = [(sc, "random", s) for sc in ("process", "softsensor") for s in range(n_random)]
    jobs += [(sc, "chrono", s) for sc in ("process", "softsensor") for s in range(n_chrono)]
    run_jobs(job, jobs, "reproduction_runs_quick.csv" if a.quick else "reproduction_runs.csv", a.procs)
