"""PSO, GA and ACO for tuning the ANFIS membership-function parameters.

All three minimise the same fitness: a flat vector theta = [centres, sigmas]
(length 2 * n_rules * n_inputs, every entry in [0, 1]) is turned into an ANFIS
whose consequents are solved by least squares on the training set; the fitness
is the resulting training MSE (paper Eq. 12). The paper does not say which
consequents are used while the metaheuristic runs; least squares is our choice.
make_fitness(..., X_eval, y_eval) is our extension: the MSE is then measured on
held-out data, which penalises membership functions that only memorise. Each optimiser starts from the SOM solution
(first individual = SOM, the rest = SOM + small Gaussian noise).
"""
import numpy as np

from . import config as C
from .anfis import firing_np, lse_consequents


def make_fitness(X, y, n_rules, X_eval=None, y_eval=None):
    n = X.shape[1]

    def fitness(theta):
        centers = theta[: n_rules * n].reshape(n_rules, n)
        sigmas = theta[n_rules * n:].reshape(n_rules, n)
        coef, train_mse = lse_consequents(X, y, centers, sigmas)
        if X_eval is None:
            return train_mse
        wbar = firing_np(X_eval, centers, sigmas)
        pred = (wbar * (X_eval @ coef[:, :-1].T + coef[:, -1])).sum(axis=1)
        return float(np.mean((pred - y_eval) ** 2))
    return fitness


def pack(centers, sigmas):
    return np.concatenate([centers.ravel(), sigmas.ravel()])


def unpack(theta, n_rules, n):
    return theta[: n_rules * n].reshape(n_rules, n), theta[n_rules * n:].reshape(n_rules, n)


def _bounds(theta0):
    """Centres in [0, 1]; sigmas in [SIGMA_MIN, 1]  (paper: parameter bounds [0, 1])."""
    lo = np.zeros_like(theta0)
    lo[len(theta0) // 2:] = C.SIGMA_MIN
    return lo, np.ones_like(theta0)


def _init_population(theta0, size, rng, lo, hi):
    pop = theta0 + rng.normal(0, C.INIT_SPREAD, size=(size, len(theta0)))
    pop[0] = theta0
    return np.clip(pop, lo, hi)


def pso(fitness, theta0, seed=0, n_particles=50, n_iter=40, w=0.7, c1=1.5, c2=1.5):
    """Particle Swarm Optimisation (paper Eq. 17-18)."""
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(theta0)
    x = _init_population(theta0, n_particles, rng, lo, hi)
    v = np.zeros_like(x)
    p_best, p_fit = x.copy(), np.array([fitness(p) for p in x])     # personal bests
    g = p_fit.argmin()
    g_best, g_fit = p_best[g].copy(), p_fit[g]                      # global best
    history = [g_fit]
    for _ in range(n_iter):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)   # velocity, Eq. 18
        x = np.clip(x + v, lo, hi)                                    # position
        fit = np.array([fitness(p) for p in x])
        better = fit < p_fit
        p_best[better], p_fit[better] = x[better], fit[better]
        if p_fit.min() < g_fit:
            g = p_fit.argmin()
            g_best, g_fit = p_best[g].copy(), p_fit[g]
        history.append(g_fit)
    return g_best, np.array(history)


def ga(fitness, theta0, seed=0, pop_size=50, n_gen=40, alpha=0.5, p_mut=0.1, tournament=3, sigma_mut=0.1):
    """Genetic Algorithm: tournament selection, blend crossover, Gaussian mutation (paper Eq. 13-14).
    One elite individual is copied unchanged so the best solution is never lost."""
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(theta0)
    pop = _init_population(theta0, pop_size, rng, lo, hi)
    fit = np.array([fitness(p) for p in pop])
    history = [fit.min()]
    for _ in range(n_gen):
        def select():
            idx = rng.choice(pop_size, tournament, replace=False)
            return pop[idx[fit[idx].argmin()]]
        children = [pop[fit.argmin()].copy()]                         # elitism
        while len(children) < pop_size:
            p1, p2 = select(), select()
            child = alpha * p1 + (1 - alpha) * p2                     # blend crossover
            mask = rng.random(child.shape) < p_mut
            child = child + mask * rng.normal(0, sigma_mut, child.shape)   # Gaussian mutation
            children.append(np.clip(child, lo, hi))
        pop = np.array(children)
        fit = np.array([fitness(p) for p in pop])
        history.append(fit.min())
    return pop[fit.argmin()].copy(), np.array(history)


def aco(fitness, theta0, seed=0, n_ants=10, n_iter=50, rho=0.5, tau0=0.1):
    """Ant-colony style search as described in the paper (Eq. 15-16).

    Every parameter has a pheromone value tau that acts as a step size: an ant
    proposes  theta = best + RP * tau  with RP ~ N(0, 1) (Eq. 15). After each
    iteration the pheromone evaporates and the best ant deposits the size of the
    move it made:  tau <- (1 - rho) * tau + |best_ant - best|  (Eq. 16).
    """
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(theta0)
    best, best_fit = theta0.copy(), fitness(theta0)
    tau = np.full_like(theta0, tau0)
    history = [best_fit]
    for _ in range(n_iter):
        ants = np.clip(best + rng.normal(size=(n_ants, len(theta0))) * tau, lo, hi)
        fit = np.array([fitness(a) for a in ants])
        leader = ants[fit.argmin()]
        tau = (1 - rho) * tau + np.abs(leader - best)
        tau = np.maximum(tau, 1e-3)                                    # keep a little exploration
        if fit.min() < best_fit:
            best, best_fit = leader.copy(), fit.min()
        history.append(best_fit)
    return best, np.array(history)


OPTIMIZERS = {"pso": (pso, C.PSO), "ga": (ga, C.GA), "aco": (aco, C.ACO)}
