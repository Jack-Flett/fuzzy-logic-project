"""First-order Takagi-Sugeno ANFIS with optional three-level attention.

Rule k:  IF x1 is A_k1 AND ... AND xn is A_kn  THEN  f_k = a_k0 + a_k1 x1 + ... + a_kn xn

Layer 1  membership   mu_ki(x_i) = exp(-(x_i - c_ki)^2 / (2 sigma_ki^2))          (paper Eq. 1)
Layer 2  firing       w_k = prod_i mu_ki(x_i)                                      (paper Eq. 2)
Layer 3  normalise    wbar_k = w_k / sum_j w_j                                     (paper Eq. 3)
Layer 4  consequent   wbar_k * f_k                                                 (paper Eq. 4)
Layer 5  output       y = sum_k wbar_k f_k   (weighted average = Sugeno defuzzification)

Each rule owns one Gaussian per input (scatter partition), so the number of
rules equals the number of SOM nodes and does not explode with the inputs.

Attention (the paper gives the score formulas, Eq. 19-20, but not where the
scores enter the network; the placement below is OUR design decision):
  feature attention  alpha (softmax over the n inputs)        -> exponent of mu:  mu_ki ^ (n * alpha_i)
  MF attention       gamma (softmax over the R rules, per input) -> exponent of mu:  mu_ki ^ (R * gamma_ki)
  rule attention     beta  (tanh, in [-1, 1])                  -> wbar_k * (1 + beta_k)
All attention networks end in a zero-initialised layer, so at the start
alpha = 1/n, gamma = 1/R, beta = 0 and the model is EXACTLY the plain ANFIS.
"""
import numpy as np
import torch
from torch import nn

from . import config as C


class AttentionNet(nn.Module):
    """Two-layer feed-forward scorer  W2 * ReLU(W1 * s + b1) + b2  (paper Eq. 19-20)."""

    def __init__(self, d_in, d_out, hidden=C.ATT_HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.l2.weight)      # start neutral
        nn.init.zeros_(self.l2.bias)

    def forward(self, s):
        return self.l2(torch.relu(self.l1(s)))


class ANFIS(nn.Module):
    """attention: any subset of the letters 'F' (feature), 'M' (membership function), 'R' (rule).

    attention_mode:
      'batch'   paper-literal: the scorer sees the batch average. With full-batch
                training this is the training-set average, which we store and reuse
                at inference, so the attention weights are learned but static.
      'sample'  the scorer sees each sample, so the weights change with the input.
    """

    def __init__(self, n_inputs, n_rules, attention="", attention_mode="batch"):
        super().__init__()
        self.n, self.R = n_inputs, n_rules
        self.attention, self.mode = attention.upper(), attention_mode
        # premise parameters
        self.centers = nn.Parameter(torch.rand(n_rules, n_inputs))
        self.sigmas = nn.Parameter(torch.full((n_rules, n_inputs), 0.3))
        # consequent parameters, one row [a_k1 .. a_kn, a_k0] per rule
        self.coef = nn.Parameter(torch.zeros(n_rules, n_inputs + 1))
        if "F" in self.attention:
            self.att_f = AttentionNet(n_inputs, n_inputs)
            self.register_buffer("sum_f", torch.zeros(1, n_inputs))
        if "M" in self.attention:
            self.att_m = AttentionNet(n_rules * n_inputs, n_rules * n_inputs)
            self.register_buffer("sum_m", torch.zeros(1, n_rules * n_inputs))
        if "R" in self.attention:
            self.att_r = AttentionNet(n_rules, n_rules)
            self.register_buffer("sum_r", torch.zeros(1, n_rules))

    # ------------------------------------------------------------------ helpers
    def _summary(self, value, name):
        """What the attention scorer looks at (see attention_mode)."""
        if self.mode == "sample":
            return value
        if self.training:
            s = value.mean(dim=0, keepdim=True)
            getattr(self, name).copy_(s.detach())     # remember for inference
            return s
        return getattr(self, name)

    def set_premise(self, centers, sigmas):
        with torch.no_grad():
            self.centers.copy_(torch.as_tensor(centers, dtype=torch.float32))
            self.sigmas.copy_(torch.as_tensor(sigmas, dtype=torch.float32))

    # ------------------------------------------------------------------ forward
    def forward(self, x, return_parts=False):
        sig = self.sigmas.clamp(min=C.SIGMA_MIN)
        # log membership degrees, shape (batch, R, n)
        log_mu = -((x[:, None, :] - self.centers) ** 2) / (2 * sig ** 2)
        parts = {"mu": log_mu.exp()}

        if "F" in self.attention:
            alpha = torch.softmax(self.att_f(self._summary(x, "sum_f")), dim=-1)          # (b|1, n)
            log_mu = log_mu * (self.n * alpha)[:, None, :]
            parts["alpha"] = alpha
        if "M" in self.attention:
            s = self._summary(parts["mu"].flatten(1), "sum_m")
            gamma = torch.softmax(self.att_m(s).view(-1, self.R, self.n), dim=1)           # over rules
            log_mu = log_mu * (self.R * gamma)
            parts["gamma"] = gamma

        # product T-norm + normalisation, done in log space for numerical safety
        wbar = torch.softmax(log_mu.sum(dim=-1), dim=1)                                    # (batch, R)
        parts["wbar"] = wbar
        if "R" in self.attention:
            beta = torch.tanh(self.att_r(self._summary(wbar, "sum_r")))                    # (b|1, R)
            wbar = wbar * (1 + beta)
            parts["beta"] = beta

        f = x @ self.coef[:, :-1].T + self.coef[:, -1]                                     # rule outputs
        y = (wbar * f).sum(dim=1)
        if return_parts:
            parts.update(rule_out=f, weights=wbar)
            return y, parts
        return y

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# NumPy twins used by the metaheuristics (thousands of evaluations, no autograd)
# ---------------------------------------------------------------------------
def firing_np(X, centers, sigmas):
    """Normalised firing strengths, shape (N, R)."""
    sig = np.maximum(sigmas, C.SIGMA_MIN)
    log_w = -(((X[:, None, :] - centers) ** 2) / (2 * sig ** 2)).sum(axis=2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    return w / w.sum(axis=1, keepdims=True)


def lse_consequents(X, y, centers, sigmas, ridge=C.RIDGE):
    """Least-squares estimate of the consequent coefficients for fixed premises.

    With the premises fixed, the ANFIS output is linear in the coefficients:
    y = sum_k wbar_k * [x, 1] . a_k  ->  ridge regression on Phi = wbar (x) [x, 1].
    This is the classic 'hybrid learning' step of Jang's ANFIS.
    """
    wbar = firing_np(X, centers, sigmas)
    X1 = np.hstack([X, np.ones((len(X), 1))])
    phi = (wbar[:, :, None] * X1[:, None, :]).reshape(len(X), -1)
    A = phi.T @ phi + ridge * np.eye(phi.shape[1])
    coef = np.linalg.solve(A, phi.T @ y)
    mse = float(np.mean((phi @ coef - y) ** 2))
    return coef.reshape(centers.shape[0], -1), mse
