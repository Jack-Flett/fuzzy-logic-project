"""Live demo: train a model in a few seconds and explain one prediction step by step.

    python demo.py                                   # median operating point, SOM-PSO-ANFIS
    python demo.py --set DBO-D=180 DQO-D=350         # change some inputs (physical units)
    python demo.py --row 3                           # a day from the test set, with the measured value
    python demo.py --model panfis-attention          # anfis | som | pso | attention | panfis | panfis-attention
    python demo.py --scenario softsensor --seed 1

It prints every ANFIS layer for the chosen input: membership degrees, firing strengths,
normalised firing strengths, rule outputs and the final weighted average, and saves the
membership functions to results/figures/demo_membership.png.
"""
import argparse

import numpy as np
import torch

from src.pipeline import LADDER, PANFIS, Run

MODELS = {"anfis": "ANFIS", "som": "SOM-ANFIS", "pso": "SOM-PSO-ANFIS", "attention": "SOM-PSO-ANFIS-Attention",
          "panfis": "SOM-PSO-PANFIS", "panfis-attention": "SOM-PSO-PANFIS-Attention"}


def explain_anfis(model, x_scaled, d, top=3):
    """Print the five layers of one ANFIS for one input vector."""
    x = torch.tensor(x_scaled[None], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y, p = model(x, return_parts=True)
    span = d.x_max - d.x_min
    centers = model.centers.detach().numpy() * span + d.x_min           # centres in physical units
    sigmas = model.sigmas.detach().clamp(min=0.05).numpy() * span
    mu, wbar, f, w_used = (p[k][0].numpy() for k in ("mu", "wbar", "rule_out", "weights"))
    order = np.argsort(-wbar)[:top]
    print(f"  {model.R} rules, {model.n} inputs. Showing the {top} rules that fire most strongly.\n")
    for k in order:
        print(f"  Rule {k + 1}:  normalised firing strength = {wbar[k]:.3f}")
        for i, name in enumerate(d.features):
            print(f"      IF {name:7s} is about {centers[k, i]:8.1f} (width {sigmas[k, i]:7.1f})"
                  f"   membership degree = {mu[k, i]:.3f}")
        terms = " ".join(f"{model.coef[k, i].item():+.3f}*{n}" for i, n in enumerate(d.features))
        print(f"      THEN y = {model.coef[k, -1].item():+.3f} {terms}   (scaled units)")
        print(f"      rule output = {f[k]:.3f} (= {d.y_to_mgl(f[k]):.1f} mg/L),"
              f" contribution to the answer = {w_used[k] * f[k]:+.4f}")
        if "beta" in p:
            print(f"      rule attention beta = {p['beta'][0, k].item():+.3f}")
        print()
    if "alpha" in p:
        a = p["alpha"][0].numpy()
        print("  Feature attention (1/n = %.3f means 'neutral'): " % (1 / model.n)
              + ", ".join(f"{n}={v:.3f}" for n, v in zip(d.features, a)))
    print(f"  Sum of normalised firing strengths = {wbar.sum():.3f}")
    print(f"  Output = sum_k weight_k * f_k = {y.item():.4f} (scaled) = {d.y_to_mgl(y.item()):.2f} mg/L\n")
    return y.item()


def plot_memberships(model, d, x_scaled, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 1, 300)
    c = model.centers.detach().numpy()
    s = model.sigmas.detach().clamp(min=0.05).numpy()
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        phys = grid * (d.x_max[i] - d.x_min[i]) + d.x_min[i]
        for k in range(model.R):
            ax.plot(phys, np.exp(-(grid - c[k, i]) ** 2 / (2 * s[k, i] ** 2)), color="#2a78d6", lw=1.2, alpha=0.7)
        ax.axvline(x_scaled[i] * (d.x_max[i] - d.x_min[i]) + d.x_min[i], color="#eb6834", lw=2)
        ax.set_title(d.features[i], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    fig.suptitle("Gaussian membership functions, one curve per rule (orange line = current input)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="pso")
    ap.add_argument("--scenario", choices=["process", "softsensor"], default="process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row", type=int, help="index of a test-set sample to explain")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="inputs in physical units")
    a = ap.parse_args()

    run = Run(seed=a.seed, scenario=a.scenario)
    d = run.data
    name = MODELS[a.model]
    print(f"\nTraining {name} (seed {a.seed}, scenario '{a.scenario}') ...")
    model, _, row = run.fit(LADDER[name])
    print(f"  done in {row['train_time_s']:.1f} s | test R2 = {row['test_R2']:.3f},"
          f" test RMSE = {row['test_RMSE_mgL']:.2f} mg/L, {row['params']} parameters\n")

    truth = None
    if a.row is not None:
        x_scaled = d.X_test[a.row]
        truth = d.y_to_mgl(d.y_test[a.row])
        physical = x_scaled * (d.x_max - d.x_min) + d.x_min
    else:
        physical = d.x_median.copy()                       # typical operating point
        for item in a.set:
            key, value = item.split("=")
            if key not in d.features:
                raise SystemExit(f"unknown input '{key}'. Inputs of this model: {d.features}")
            physical[d.features.index(key)] = float(value)
        x_scaled = d.x_to_scaled(physical)

    print("Inputs (crisp values -> scaled to [0, 1] with the training range):")
    for n, p, s, lo, hi in zip(d.features, physical, x_scaled, d.x_min, d.x_max):
        print(f"  {n:7s} = {p:8.1f}   scaled {s:.3f}   (training range {lo:.1f} .. {hi:.1f})")
    print()

    if isinstance(model, PANFIS):
        outs = []
        for b, branch in enumerate(model.branches):
            print(f"--- Branch {b + 1} ---")
            outs.append(explain_anfis(branch, x_scaled, d, top=2))
        v = model.fusion.numpy()
        y = float(np.dot(v, outs))
        print("Fusion: " + " + ".join(f"{w:.3f}*{o:.4f}" for w, o in zip(v, outs)) + f" = {y:.4f}")
        first = model.branches[0]
    else:
        y = explain_anfis(model, x_scaled, d)
        first = model
    print(f"PREDICTED effluent BOD (DBO-S) = {d.y_to_mgl(y):.2f} mg/L")
    if truth is not None:
        print(f"MEASURED  effluent BOD (DBO-S) = {truth:.2f} mg/L")
    out = "results/figures/demo_membership.png"
    plot_memberships(first, d, x_scaled, out)
    print(f"\nMembership functions saved to {out}")
