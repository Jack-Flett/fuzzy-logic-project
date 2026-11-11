"""Self-Organising Map used to initialise the ANFIS membership functions.

Each SOM node becomes one fuzzy rule:
  centre  c_k  = the node's weight vector                          (paper Eq. 7)
  width sigma_k = per-feature std of the training samples mapped to the node
"""
import numpy as np

from . import config as C


class SOM:
    def __init__(self, rows, cols, sigma=C.SOM_SIGMA, lr=C.SOM_LR, n_iter=C.SOM_ITER, seed=0):
        self.rows, self.cols = rows, cols
        self.sigma, self.lr, self.n_iter = sigma, lr, n_iter
        self.rng = np.random.default_rng(seed)
        # grid coordinates of the nodes, used by the neighbourhood function
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
        self.weights = None

    def fit(self, X):
        n_nodes = self.rows * self.cols
        # initialise the weights with randomly chosen training samples
        self.weights = X[self.rng.choice(len(X), n_nodes, replace=len(X) < n_nodes)].astype(float)
        for t in range(self.n_iter):
            lr = self.lr * (1 - t / self.n_iter)                 # linearly decaying learning rate
            for x in X[self.rng.permutation(len(X))]:
                bmu = np.argmin(((self.weights - x) ** 2).sum(axis=1))          # best matching unit, Eq. 5
                d2 = ((self.grid - self.grid[bmu]) ** 2).sum(axis=1)
                h = np.exp(-d2 / (2 * self.sigma ** 2))                           # neighbourhood
                self.weights += lr * h[:, None] * (x - self.weights)             # update, Eq. 6
        return self

    def assign(self, X):
        d = ((X[:, None, :] - self.weights[None]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


def som_membership_init(X, grid, seed=0):
    """Return (centers, sigmas), each of shape (rows*cols, n_features)."""
    som = SOM(*grid, seed=seed).fit(X)
    labels = som.assign(X)
    global_std = X.std(axis=0)
    sigmas = np.tile(global_std, (len(som.weights), 1))
    for k in range(len(som.weights)):
        members = X[labels == k]
        if len(members) >= 3:                        # too few samples -> keep the global std
            sigmas[k] = members.std(axis=0)
    sigmas = np.clip(sigmas, C.SIGMA_MIN, C.UNIFORM_SIGMA[1])
    return som.weights.copy(), sigmas


def uniform_membership_init(n_rules, n_features, seed=0):
    """The paper's non-SOM baseline: random means in [0.1, 0.9] and stds in [0.1, 0.5]."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(*C.UNIFORM_CENTER, size=(n_rules, n_features))
    sigmas = rng.uniform(*C.UNIFORM_SIGMA, size=(n_rules, n_features))
    return centers, sigmas
