"""Small helper to run independent jobs on several CPU cores and collect rows in a CSV."""
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def _init():
    torch.set_num_threads(1)        # one thread per worker; the models are tiny


def run_jobs(job_fn, jobs, out_csv, n_proc=2):
    """job_fn(job) -> list of dict rows. Writes all rows to results/tables/<out_csv>."""
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("spawn").Pool(n_proc, initializer=_init) as pool:
        for i, part in enumerate(pool.imap(job_fn, jobs), 1):
            rows.extend(part)
            print(f"  job {i}/{len(jobs)} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / out_csv, index=False)
    print("wrote", TABLES / out_csv, df.shape)
    return df
