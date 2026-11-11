"""Builds, trains and evaluates every model variant of the study.

A Run object belongs to one (seed, split, scenario). It caches the SOM and
metaheuristic results, so e.g. SOM-ANFIS, SOM-PSO-ANFIS and
SOM-PSO-ANFIS-Attention start from the very same membership functions and
differ only in the component under test (paired comparison).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import config as C
from .anfis import ANFIS
from .data import Dataset, prepare
from .metrics import regression_metrics
from .optimizers import OPTIMIZERS, make_fitness, pack, unpack
from .som import som_membership_init, uniform_membership_init
from .train import predict, train_adam


class PANFIS(nn.Module):
    """Parallel ANFIS: several ANFIS branches whose outputs are fused linearly.

    fusion='equal'   y = mean of the branch outputs               (paper Eq. 21 / Table 1)
    fusion='learned' y = sum_m v_m * y_m, v fitted on held-out data (paper Sec. 3.3 / Fig. 14-15)
    """

    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.register_buffer("fusion", torch.full((len(branches),), 1.0 / len(branches)))

    def branch_outputs(self, x):
        return torch.stack([b(x) for b in self.branches], dim=1)

    def forward(self, x):
        return self.branch_outputs(x) @ self.fusion

    def n_parameters(self):
        return sum(b.n_parameters() for b in self.branches)


@dataclass
class Run:
    seed: int = 0
    split: str = "random"
    scenario: str = "process"
    data: Dataset = None
    _cache: dict = field(default_factory=dict)
    _cost: float = 0.0          # seconds spent on SOM / metaheuristics for the model being built

    def __post_init__(self):
        if self.data is None:
            self.data = prepare(self.seed, self.split, self.scenario)

    # ----------------------------------------------------------- premise parameters
    def premise(self, init, opt, grid, seed_offset=0):
        """Membership-function parameters after initialisation (+ optional metaheuristic)."""
        key = (init, opt, grid, seed_offset)
        if key in self._cache:
            self._cost += self._cache[key][3]      # a cached step still counts in the reported time
            return self._cache[key]
        t0, cost0 = time.perf_counter(), self._cost
        d, R, seed = self.data, grid[0] * grid[1], self.seed + seed_offset
        if init == "som":
            centers, sigmas = som_membership_init(d.X_train, grid, seed)
        else:
            centers, sigmas = uniform_membership_init(R, d.X_train.shape[1], seed)
        history = None
        if opt:
            # "pso_val" = PSO whose fitness is measured on the validation set (our extension)
            fn, kwargs = OPTIMIZERS[opt.replace("_val", "")]
            held_out = (d.X_val, d.y_val) if opt.endswith("_val") else ()
            c0, s0 = self.premise(init, None, grid, seed_offset)[:2]
            fitness = make_fitness(d.X_train, d.y_train, R, *held_out)
            theta, history = fn(fitness, pack(c0, s0), seed=seed, **kwargs)
            centers, sigmas = unpack(theta, R, d.X_train.shape[1])
        seconds = time.perf_counter() - t0
        self._cost = cost0 + seconds
        self._cache[key] = (centers, sigmas, history, seconds)
        return self._cache[key]

    # ----------------------------------------------------------- single ANFIS
    def build_anfis(self, init="som", opt=None, grid=C.SINGLE_GRID, attention="", attention_mode="batch",
                    seed_offset=0):
        d = self.data
        centers, sigmas = self.premise(init, opt, grid, seed_offset)[:2]
        torch.manual_seed(self.seed + seed_offset)
        model = ANFIS(d.X_train.shape[1], grid[0] * grid[1], attention, attention_mode)
        model.set_premise(centers, sigmas)
        # consequents start at zero and are learned, together with the premises and the
        # attention networks, by back-propagation (Adam + early stopping), as in the paper
        history = train_adam(model, d.X_train, d.y_train, d.X_val, d.y_val)
        return model, history

    # ----------------------------------------------------------- parallel ANFIS
    def build_panfis(self, grids=C.PANFIS_GRIDS, opt="pso", attention="", attention_mode="batch",
                     fusion="learned"):
        d = self.data
        branches = [self.build_anfis("som", opt, g, attention, attention_mode, seed_offset=1000 * i)[0]
                    for i, g in enumerate(grids)]
        model = PANFIS(branches)
        if fusion == "learned":
            # 3 fusion weights, fitted by least squares on the validation predictions (stacking)
            with torch.no_grad():
                P = model.branch_outputs(torch.as_tensor(d.X_val)).numpy().astype(float)
            v = np.linalg.lstsq(P, d.y_val.astype(float), rcond=None)[0]
            model.fusion.copy_(torch.as_tensor(v, dtype=torch.float32))
        return model, None

    # ----------------------------------------------------------- evaluation
    def evaluate(self, model, seconds=0.0):
        return self.evaluate_fn(lambda X: predict(model, X), model.n_parameters(), seconds)

    def evaluate_fn(self, predict_fn, params, seconds=0.0):
        """Metrics of any predictor (also used for the non-fuzzy baselines)."""
        d = self.data
        row = {"params": params, "train_time_s": seconds}
        for name, X, y in (("train", d.X_train, d.y_train), ("val", d.X_val, d.y_val),
                           ("test", d.X_test, d.y_test)):
            for k, v in regression_metrics(y, predict_fn(X)).items():
                row[f"{name}_{k}"] = v
        pred = predict_fn(d.X_test)
        row["test_R2_raw_target"] = regression_metrics(d.y_test_raw, pred)["R2"]   # unclipped test target
        row["test_RMSE_mgL"] = row["test_RMSE"] * (d.y_max - d.y_min)
        return row

    def fit(self, spec):
        """spec: dict with 'kind' = 'anfis' | 'panfis' plus the builder's keyword arguments."""
        spec = dict(spec)
        kind = spec.pop("kind", "anfis")
        t0, self._cost = time.perf_counter(), 0.0
        cached_before = set(self._cache)
        model, history = (self.build_panfis if kind == "panfis" else self.build_anfis)(**spec)
        # wall time, plus the cost of initialisation steps that were reused from the cache
        fresh = sum(v[3] for k, v in self._cache.items() if k not in cached_before)
        seconds = time.perf_counter() - t0 + max(self._cost - fresh, 0.0)
        return model, history, self.evaluate(model, seconds)


# The six models of the paper's main progression (its Tables 2-4) plus the two other optimisers.
LADDER = {
    "ANFIS":                    dict(init="uniform"),
    "SOM-ANFIS":                dict(init="som"),
    "SOM-PSO-ANFIS":            dict(init="som", opt="pso"),
    "SOM-PSO-ANFIS-Attention":  dict(init="som", opt="pso", attention="FMR"),
    "SOM-PSO-PANFIS":           dict(kind="panfis"),
    "SOM-PSO-PANFIS-Attention": dict(kind="panfis", attention="FMR"),
}
OPTIMISER_VARIANTS = {
    "SOM-GA-ANFIS":  dict(init="som", opt="ga"),
    "SOM-ACO-ANFIS": dict(init="som", opt="aco"),
}
# Reported test R2 on the authors' private dataset (paper Tables 2-4), for reference only.
PAPER_TEST_R2 = {"ANFIS": 0.8838, "SOM-ANFIS": 0.9136, "SOM-GA-ANFIS": 0.9346, "SOM-ACO-ANFIS": 0.9321,
                 "SOM-PSO-ANFIS": 0.9409, "SOM-PSO-ANFIS-Attention": 0.9576, "SOM-PSO-PANFIS": 0.9751,
                 "SOM-PSO-PANFIS-Attention": 0.9866}
