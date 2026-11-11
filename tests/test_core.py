"""Unit tests:  python -m pytest -q"""
import numpy as np
import torch

from src.anfis import ANFIS, firing_np, lse_consequents
from src.data import prepare
from src.optimizers import aco, ga, make_fitness, pack, pso
from src.som import som_membership_init


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2)).astype(np.float32)
    y = (np.sin(3 * X[:, 0]) + X[:, 1] ** 2).astype(np.float32)
    return X, y


def test_forward_matches_hand_computation():
    """Two rules, two inputs: compare the network with the five ANFIS layers written out by hand."""
    m = ANFIS(2, 2)
    c = np.array([[0.2, 0.3], [0.7, 0.8]]); s = np.array([[0.2, 0.2], [0.3, 0.1]])
    coef = np.array([[1.0, 2.0, 0.5], [-1.0, 0.5, 2.0]])
    m.set_premise(c, s)
    with torch.no_grad():
        m.coef.copy_(torch.tensor(coef, dtype=torch.float32))
    x = np.array([0.4, 0.6])
    mu = np.exp(-(x - c) ** 2 / (2 * s ** 2))            # layer 1
    w = mu.prod(axis=1)                                  # layer 2
    wbar = w / w.sum()                                   # layer 3
    f = coef[:, :2] @ x + coef[:, 2]                     # layer 4
    expected = (wbar * f).sum()                          # layer 5
    got = m(torch.tensor(x[None], dtype=torch.float32)).item()
    assert abs(got - expected) < 1e-5
    assert np.allclose(firing_np(x[None], c, s)[0], wbar, atol=1e-6)


def test_attention_starts_neutral():
    """Zero-initialised attention must not change the plain ANFIS output."""
    X, _ = toy()
    torch.manual_seed(0); base = ANFIS(2, 4)
    for mode in ("batch", "sample"):
        att = ANFIS(2, 4, attention="FMR", attention_mode=mode)
        att.load_state_dict(base.state_dict(), strict=False)
        att.train()
        xb = torch.tensor(X)
        assert torch.allclose(att(xb), base(xb), atol=1e-5)


def test_normalised_firing_sums_to_one():
    X, _ = toy()
    c, s = som_membership_init(X, (3, 3))
    assert c.shape == (9, 2) and s.shape == (9, 2) and (s >= 0.05).all()
    assert np.allclose(firing_np(X, c, s).sum(axis=1), 1.0)


def test_metaheuristics_do_not_get_worse():
    X, y = toy()
    c, s = som_membership_init(X, (2, 2))
    fit = make_fitness(X, y, 4)
    start = fit(pack(c, s))
    for opt in (pso, ga, aco):
        best, hist = opt(fit, pack(c, s), seed=0)
        assert hist[-1] <= start + 1e-12 and abs(fit(best) - hist[-1]) < 1e-9
        assert (np.diff(hist) <= 1e-12).all()            # best-so-far curve never increases


def test_lse_solves_linear_target():
    X, _ = toy()
    y = 2 * X[:, 0] - X[:, 1] + 0.5
    c, s = som_membership_init(X, (2, 2))
    _, mse = lse_consequents(X, y, c, s, ridge=1e-8)
    assert mse < 1e-8


def test_preprocessing_has_no_leakage_and_is_scaled():
    d = prepare(seed=0)
    assert len(d.features) == 6 and not any(f.startswith("RD-") for f in d.features)
    assert d.X_train.min() >= 0 and d.X_train.max() <= 1
    assert d.X_test.min() >= 0 and d.X_test.max() <= 1      # clipped with TRAIN fences, scaled with TRAIN range
    assert len(d.X_train) + len(d.X_val) + len(d.X_test) == 504
    c = prepare(seed=0, split="chrono")
    assert len(c.X_train) == 353
