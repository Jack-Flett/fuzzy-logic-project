"""Turns the raw per-run CSV files into the summary tables and figures used in the report.

    python -m experiments.make_report_assets

Reads   results/tables/{reproduction_runs,extension_runs,noise_runs}.csv
Writes  results/tables/*.csv / *.md / *.tex   and   results/figures/*.png
Every number in the report comes from these files, nothing is typed in by hand.

Statistics. Models are compared pairwise on the same repetitions (same split, same
initialisation). We report the mean paired difference in test R2 and two p-values:
  * Wilcoxon signed-rank test on the paired differences;
  * the corrected resampled t-test of Nadeau & Bengio (2003), which inflates the variance by
    (1/k + n_test/n_train) because the repeated random splits overlap and are not independent.
"""
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment import FIGURES, TABLES
from src.pipeline import LADDER, PAPER_TEST_R2, Run

N_TRAIN, N_TEST = 353, 75
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight"})
SCEN = {"process": "Process inputs", "softsensor": "Soft-sensor inputs"}
BASELINES = ["Ridge regression", "MLP (32-16)", "Random forest", "Mean predictor"]


# ------------------------------------------------------------------ statistics
def paired(df, a, b, metric="test_R2"):
    """Paired comparison b - a over the seeds. Returns mean diff, std, Wilcoxon p, corrected-t p, CI half-width."""
    p = df[df.model.isin([a, b])].pivot(index="seed", columns="model", values=metric)
    d = (p[b] - p[a]).to_numpy()
    k = len(d)
    if np.allclose(d, 0):
        return 0.0, 0.0, 1.0, 1.0, 0.0
    w = stats.wilcoxon(d).pvalue
    se = np.sqrt((1 / k + N_TEST / N_TRAIN) * d.var(ddof=1))
    t = d.mean() / se
    return d.mean(), d.std(ddof=1), w, 2 * stats.t.sf(abs(t), k - 1), stats.t.ppf(0.975, k - 1) * se


def pm(x, digits=3):
    return f"{np.mean(x):.{digits}f} $\\pm$ {np.std(x, ddof=1):.{digits}f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_table(name, header, rows, caption_note=""):
    """Save one table as CSV, Markdown and LaTeX (tabular only)."""
    plain = [[str(c).replace("$\\pm$", "±").replace("$<$", "<").replace("$", "").replace("\\", "") for c in r]
             for r in rows if r[0] != "---"]
    pd.DataFrame(plain, columns=header).to_csv(TABLES / f"{name}.csv", index=False)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in plain]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\hline",
           " & ".join(h.replace("%", "\\%").replace("#", "\\#").replace("R2", "$R^2$").replace("Delta", "$\\Delta$")
                      for h in header) + " \\\\", "\\hline"]
    for r in rows:
        if r[0] == "---":
            tex.append("\\hline")
        else:
            tex.append(" & ".join(str(c).replace("%", "\\%").replace("#", "\\#") for c in r) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(tex) + "\n")
    lines = [r for r in md if not r.startswith("| ---")]
    print(f"\n== {name} ==\n" + "\n".join(lines))


# ------------------------------------------------------------------ tables
def ladder_tables(rep):
    names = list(LADDER)
    for scen in SCEN:
        df = rep[(rep.scenario == scen) & (rep.split == "random")]
        rows = []
        for i, m in enumerate(names + ["SOM-GA-ANFIS", "SOM-ACO-ANFIS"]):
            g = df[df.model == m]
            ref = names[i - 1] if 0 < i < len(names) else ("SOM-ANFIS" if i >= len(names) else None)
            if ref:
                d, _, w, pc, _ = paired(df, ref, m)
                delta, pcol = f"${d:+.3f}$", fmt_p(pc)
            else:
                delta, pcol = "--", "--"
            if i == len(names):
                rows.append(["---"] + [""] * 7)
            rows.append([m, f"{int(g.params.iloc[0])}", pm(g.test_RMSE), pm(g.test_MAE), pm(g.test_R2),
                         delta, pcol, f"{PAPER_TEST_R2[m]:.4f}"])
        rows.append(["---"] + [""] * 7)
        for m in BASELINES:
            g = df[df.model == m]
            par = f"{int(g.params.iloc[0])}" if g.params.iloc[0] > 0 else "--"
            rows.append([m, par, pm(g.test_RMSE), pm(g.test_MAE), pm(g.test_R2), "--", "--", "--"])
        write_table(f"ladder_{scen}", ["Model", "#Par.", "RMSE", "MAE", "R2", "Delta R2", "p", "Paper R2"], rows)

    # chronological protocol + raw (unclipped) target, compact
    rows = []
    for m in names + BASELINES[:3]:
        r = [m]
        for scen in SCEN:
            a = rep[(rep.scenario == scen) & (rep.split == "random") & (rep.model == m)]
            c = rep[(rep.scenario == scen) & (rep.split == "chrono") & (rep.model == m)]
            r += [f"{a.test_R2_raw_target.mean():.3f}", pm(c.test_R2)]
        if m == BASELINES[0]:
            rows.append(["---"] + [""] * 4)
        rows.append(r)
    write_table("ladder_chrono_raw", ["Model", "Process, raw", "Process, chrono.", "Soft-s., raw", "Soft-s., chrono."], rows)

    # training time
    df = rep[(rep.split == "random")]
    rows = [[m, f"{df[df.model == m].train_time_s.mean():.2f}"] for m in names + ["SOM-GA-ANFIS", "SOM-ACO-ANFIS"] + BASELINES[:3]]
    write_table("train_time", ["Model", "Mean training time (s)"], rows)


def sanity_table(san):
    """Synthetic benchmark: every repetition uses freshly generated data, so a plain paired t-test is valid."""
    rows = []
    piv = san.pivot(index="seed", columns="model", values="test_R2")
    for m in list(LADDER) + BASELINES[:3]:
        g = san[san.model == m]
        p = "--" if m == "Ridge regression" else fmt_p(stats.ttest_rel(piv[m], piv["Ridge regression"]).pvalue)
        if m == BASELINES[0]:
            rows.append(["---", "", "", ""])
        rows.append([m, pm(g.test_RMSE, 4), pm(g.test_R2), p])
    write_table("sanity_synthetic", ["Model", "RMSE", "R2", "p vs ridge"], rows)


def extension_tables(ext):
    for exp, ref, name in (("E1 attention", "att:none", "ablation"),
                           ("E2 structure", "PANFIS 9+16+25 learned", "structure"),
                           ("E3 fitness", "PSO fitness = train MSE (paper)", "fitness")):
        e = ext[ext.experiment == exp]
        rows = []
        for m in e.model.unique():
            r = [m.replace("att:", ""), f"{int(e[e.model == m].params.iloc[0])}"]
            for scen in SCEN:
                df = e[e.scenario == scen]
                g = df[df.model == m]
                if m == ref:
                    r += [pm(g.test_R2), "ref.", "--"]
                else:
                    d, _, w, pc, _ = paired(df, ref, m)
                    r += [pm(g.test_R2), f"${d:+.3f}$", fmt_p(pc)]
            rows.append(r)
        write_table(name, ["Variant", "#Par.", "Proc. R2", "Delta", "p", "Soft. R2", "Delta", "p"], rows)


def noise_table(noise):
    rows = []
    for scen in SCEN:
        df = noise[noise.scenario == scen]
        for m in df.model.unique():
            g = df[df.model == m].groupby("noise").test_R2.mean()
            rows.append([SCEN[scen].split()[0], m] + [f"{v:.3f}" for v in g.values] + [f"${g.iloc[-1] - g.iloc[0]:+.3f}$"])
        rows.append(["---"] + [""] * (len(g) + 2))
    levels = [f"s={v:g}" for v in sorted(noise.noise.unique())]
    write_table("noise", ["Inputs", "Model"] + levels + ["Loss"], rows[:-1])


def summary_table(rep, ext):
    """Guideline-style 'baseline vs developed method' table, soft-sensor and process scenarios."""
    rows = []
    for scen in SCEN:
        base = rep[(rep.scenario == scen) & (rep.split == "random") & (rep.model == "SOM-PSO-PANFIS-Attention")]
        prop = ext[(ext.scenario == scen) & (ext.model == "PANFIS 9+16+25 equal")]
        for metric, label, digits in (("test_RMSE", "RMSE (scaled)", 4), ("test_MAE", "MAE (scaled)", 4),
                                      ("test_R2", "R2", 3), ("params", "Parameters", 0),
                                      ("train_time_s", "Training time (s)", 2)):
            b, p = base[metric].mean(), prop[metric].mean()
            rows.append([SCEN[scen].split()[0], label, f"{b:.{digits}f}", f"{p:.{digits}f}", f"${(p - b) / abs(b) * 100:+.1f}$%"])
        rows.append(["---", "", "", "", ""])
    write_table("summary_baseline_vs_proposed", ["Inputs", "Metric", "Baseline", "Simplified", "Change"], rows[:-1])


def facts(rep, san):
    """Single numbers quoted in the report text, written as LaTeX macros (results/tables/facts.tex)."""
    from collections import Counter
    from src.data import load_raw, prepare
    f = {}
    raw = load_raw()
    f["nrows"] = len(raw)
    f["targetmedian"] = f"{raw['DBO-S'].median():.0f}"
    f["targetmax"] = f"{raw['DBO-S'].max():.0f}"
    for scen, tag in (("process", "proc"), ("softsensor", "soft")):
        cnt = Counter(n for s in range(20) for n in prepare(s, "random", scen).features)
        f[f"features{tag}"] = ", ".join(f"{n} ({c})" for n, c in cnt.most_common())
        df = rep[(rep.scenario == scen) & (rep.split == "random")]
        for m, mtag in (("SOM-PSO-PANFIS-Attention", "full"), ("ANFIS", "anfis"), ("Ridge regression", "ridge"),
                        ("Mean predictor", "mean")):
            f[f"rmsemgl{tag}{mtag}"] = f"{df[df.model == m].test_RMSE_mgL.mean():.2f}"
        d, _, w, pc, ci = paired(df, "Ridge regression", "SOM-PSO-PANFIS-Attention")
        f[f"fullvsridge{tag}"] = f"${d:+.3f}$"
        f[f"fullvsridgep{tag}"] = fmt_p(pc)
        d, _, w, pc, ci = paired(df, "ANFIS", "SOM-PSO-PANFIS-Attention")
        f[f"fullvsanfis{tag}"] = f"${d:+.3f}$"
        f[f"fullvsanfisp{tag}"] = fmt_p(pc)
        f[f"fullvsanfisw{tag}"] = fmt_p(w)
    d0 = prepare(0)
    f["ysplit"] = f"{len(d0.X_train)}/{len(d0.X_val)}/{len(d0.X_test)}"
    f["yclipmax"] = f"{d0.y_max:.1f}"
    piv = san.pivot(index="seed", columns="model", values="test_R2")
    f["sanityanfis"] = f"{piv['ANFIS'].mean():.3f}"
    f["sanityridge"] = f"{piv['Ridge regression'].mean():.3f}"
    f["sanitysomp"] = fmt_p(stats.ttest_rel(piv["SOM-ANFIS"], piv["ANFIS"]).pvalue)
    f["sanitysomd"] = f"${(piv['SOM-ANFIS'] - piv['ANFIS']).mean():+.3f}$"
    f["sanitypsod"] = f"${(piv['SOM-PSO-ANFIS'] - piv['SOM-ANFIS']).mean():+.3f}$"
    f["sanitypsop"] = fmt_p(stats.ttest_rel(piv["SOM-PSO-ANFIS"], piv["SOM-ANFIS"]).pvalue)
    tex = "".join(f"\\newcommand{{\\fact{k}}}{{{v}}}\n" for k, v in f.items())
    (TABLES / "facts.tex").write_text(tex)
    (TABLES / "facts.md").write_text("".join(f"- {k}: {str(v).replace('$', '')}\n" for k, v in f.items()))
    print("\n== facts ==\n" + (TABLES / "facts.md").read_text())


# ------------------------------------------------------------------ figures
def fig_ladder(rep):
    names = list(LADDER)
    short = ["ANFIS", "SOM-\nANFIS", "SOM-PSO-\nANFIS", "+Atten-\ntion", "SOM-PSO-\nPANFIS", "PANFIS+\nAttention"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(range(6), [PAPER_TEST_R2[m] for m in names], "-o", color=GREY, lw=2, ms=6)
    for i, m in enumerate(names):
        ax.annotate(f"{PAPER_TEST_R2[m]:.3f}", (i, PAPER_TEST_R2[m]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_title("Reported in the paper (private data, single run)", fontsize=9)
    ax.set_ylim(0.86, 1.0)
    ax_top = {s: rep[(rep.scenario == s) & (rep.split == "random") & rep.model.isin(names)].test_R2.max() + 0.03
              for s in SCEN}
    for ax, scen, color in zip(axes[1:], SCEN, (BLUE, ORANGE)):
        df = rep[(rep.scenario == scen) & (rep.split == "random")]
        rng = np.random.default_rng(0)
        for i, m in enumerate(names):
            v = df[df.model == m].test_R2.to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=color, alpha=0.25, linewidths=0)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="o", color=color, ms=6, capsize=3, lw=1.5)
            ax.annotate(f"{v.mean():.3f}", (i, ax_top[scen]), ha="center", fontsize=8)
        ridge = df[df.model == "Ridge regression"].test_R2.mean()
        ax.axhline(ridge, color=GREY, lw=1, ls="--")
        ax.annotate(f"dashed: ridge regression, {ridge:.3f}", (0.02, 0.03), xycoords="axes fraction",
                    fontsize=8, color=GREY)
        ax.set_ylim(top=ax_top[scen] + 0.04)
        ax.set_title(f"Ours, UCI data, {SCEN[scen].lower()} (20 repetitions, mean ± std)", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(6), short, fontsize=7.5)
        ax.set_ylabel("test $R^2$")
    fig.savefig(FIGURES / "ladder.png")
    plt.close(fig)


def fig_deltas(ext, exp, ref, fname, title):
    e = ext[ext.experiment == exp]
    models = [m for m in e.model.unique() if m != ref]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(models) + 1.3), constrained_layout=True)
    for j, (scen, color) in enumerate(zip(SCEN, (BLUE, ORANGE))):
        df = e[e.scenario == scen]
        for i, m in enumerate(models):
            d, _, _, _, ci = paired(df, ref, m)
            y = len(models) - 1 - i + (0.14 if j == 0 else -0.14)
            ax.errorbar(d, y, xerr=ci, fmt="o", color=color, ms=5, capsize=3, lw=1.5,
                        label=SCEN[scen] if i == 0 else None)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(range(len(models)), [m.replace("att:", "") for m in models][::-1])
    ax.set_xlabel(f"change in test $R^2$ relative to '{ref.replace('att:', '')}' "
                  "(mean, 95% CI, corrected resampled t)")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def fig_noise(noise):
    show = {"SOM-PSO-PANFIS-Attention": BLUE, "SOM-PSO-ANFIS": ORANGE, "ANFIS": AQUA,
            "Ridge regression": YELLOW, "MLP (32-16)": MAGENTA}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, scen in zip(axes, SCEN):
        df = noise[noise.scenario == scen]
        for m, color in show.items():
            g = df[df.model == m].groupby("noise").test_R2.mean()
            ax.plot(range(len(g)), g.values, "-o", color=color, lw=2, ms=5, label=m)
        ax.set_xticks(range(len(g)), [f"{v:g}" for v in g.index])
        ax.set_xlabel("std of Gaussian noise added to the test inputs (fraction of input range)")
        ax.set_ylabel("test $R^2$ (mean of 20 repetitions)")
        ax.set_title(SCEN[scen], fontsize=9.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "noise.png")
    plt.close(fig)


def seed_figures():
    """Figures that need a trained model: re-train seed 0 (a few seconds)."""
    run = Run(seed=0, scenario="softsensor")
    d = run.data

    # --- membership functions of one input at the four stages
    j = d.features.index("DBO-D") if "DBO-D" in d.features else 0
    grid = np.linspace(0, 1, 300)
    phys = grid * (d.x_max[j] - d.x_min[j]) + d.x_min[j]
    stages = [("Uniform initialisation", *run.premise("uniform", None, (3, 3))[:2]),
              ("SOM initialisation", *run.premise("som", None, (3, 3))[:2]),
              ("After PSO", *run.premise("som", "pso", (3, 3))[:2])]
    model, history, _ = run.fit(LADDER["SOM-PSO-ANFIS"])
    stages.append(("After gradient training", model.centers.detach().numpy(), model.sigmas.detach().numpy()))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6), sharey=True, constrained_layout=True)
    for ax, (title, c, s) in zip(axes, stages):
        for k in range(len(c)):
            ax.plot(phys, np.exp(-(grid - c[k, j]) ** 2 / (2 * np.maximum(s[k, j], 0.05) ** 2)), color=BLUE, lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"{d.features[j]} (mg/L)")
    axes[0].set_ylabel("membership degree")
    fig.savefig(FIGURES / "membership_stages.png")
    plt.close(fig)

    # --- convergence of the three metaheuristics (mean of seeds 0-4), x axis = fitness evaluations
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    per_iter = {"pso": 50, "ga": 50, "aco": 10}
    for opt, color in (("pso", BLUE), ("ga", ORANGE), ("aco", AQUA)):
        hs = np.array([Run(seed=s, scenario="softsensor").premise("som", opt, (3, 3))[2] for s in range(5)])
        x = np.arange(hs.shape[1]) * per_iter[opt]
        ax.plot(x, hs.mean(axis=0), color=color, lw=2, label=opt.upper())
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best training MSE (fitness)")
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "convergence.png")
    plt.close(fig)

    # --- loss curves and predicted-vs-measured
    from src.train import predict
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), constrained_layout=True)
    axes[0].plot(history["train"], color=BLUE, lw=2, label="training")
    axes[0].plot(history["val"], color=ORANGE, lw=2, label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE (scaled target)")
    axes[0].set_title("SOM-PSO-ANFIS, early stopping", fontsize=9.5, loc="left")
    axes[0].legend(frameon=False)
    pan, _, row = run.fit(LADDER["SOM-PSO-PANFIS-Attention"])
    y, p = d.y_to_mgl(d.y_test), d.y_to_mgl(predict(pan, d.X_test))
    lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
    axes[1].scatter(y, p, s=16, color=BLUE, alpha=0.7, linewidths=0)
    axes[1].plot(lim, lim, color=GREY, lw=1)
    axes[1].set_xlabel("measured effluent BOD (mg/L)"); axes[1].set_ylabel("predicted (mg/L)")
    axes[1].set_title(f"SOM-PSO-PANFIS-Attention, test set, $R^2$ = {row['test_R2']:.2f}", fontsize=9.5, loc="left")
    axes[2].scatter(p, y - p, s=16, color=BLUE, alpha=0.7, linewidths=0)
    axes[2].axhline(0, color=GREY, lw=1)
    axes[2].set_xlabel("predicted (mg/L)"); axes[2].set_ylabel("residual (mg/L)")
    axes[2].set_title("Residuals", fontsize=9.5, loc="left")
    fig.savefig(FIGURES / "training_and_predictions.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(TABLES / "reproduction_runs.csv")
    ext = pd.read_csv(TABLES / "extension_runs.csv")
    noise = pd.read_csv(TABLES / "noise_runs.csv")
    ladder_tables(rep)
    extension_tables(ext)
    noise_table(noise)
    summary_table(rep, ext)
    san = pd.read_csv(TABLES / "sanity_synthetic_runs.csv")
    sanity_table(san)
    facts(rep, san)
    fig_ladder(rep)
    fig_deltas(ext, "E1 attention", "att:none", "ablation.png", "E1  Attention ablation on SOM-PSO-ANFIS")
    fig_deltas(ext, "E2 structure", "PANFIS 9+16+25 learned", "structure.png",
               "E2  Is the gain from PANFIS more than 'more rules'?")
    fig_noise(noise)
    seed_figures()
    print("\nfigures written to", FIGURES)
