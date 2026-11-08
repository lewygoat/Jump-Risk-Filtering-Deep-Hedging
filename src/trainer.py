"""
Risk Objectives, End-to-End BPTT Trainer, and Statistical Inference Utilities.

Implements Section 3.3 and Section 4.1(4) of the manuscript:
- Rockafellar-Uryasev (2000) CVaR_alpha objective (Eq. 14):
  J_{CVaR}(Theta, b) = b + 1 / ((1 - alpha) * B) * sum_i max(L_T^{(i)} - b, 0)
- Symmetric Mean-Variance objective (Eq. 15):
  J_{MV}(Theta; lambda_{MV}) = E[L_T] + (lambda_{MV} / 2) * Var(L_T)
- Paired Student's t-test, Wilcoxon signed-rank test, 10,000-resample paired bootstrap CI,
  Holm-Bonferroni step-down adjusted p-values (p_holm), and 80%-power Minimum Detectable Effect (MDE_80%).
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn


class RockafellarUryasevCVaRLoss(nn.Module):
    """Jointly learnable Rockafellar-Uryasev CVaR_alpha objective (Eq. 14)."""

    def __init__(self, alpha: float = 0.95, init_b: float = 0.015) -> None:
        super().__init__()
        self.alpha = alpha
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float64))

    def forward(self, L_T: torch.Tensor) -> torch.Tensor:
        shortfall = torch.clamp(L_T - self.b, min=0.0)
        return self.b + shortfall.mean() / (1.0 - self.alpha)


class MeanVarianceLoss(nn.Module):
    """Symmetric Mean-Variance hedging objective (Eq. 15)."""

    def __init__(self, lambda_mv: float = 10.0) -> None:
        super().__init__()
        self.lambda_mv = lambda_mv

    def forward(self, L_T: torch.Tensor) -> torch.Tensor:
        return L_T.mean() + 0.5 * self.lambda_mv * L_T.var(unbiased=True)


def compute_empirical_cvar(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """Compute out-of-sample empirical VaR_alpha and CVaR_alpha (mean of worst 1-alpha tail)."""
    arr = np.asarray(losses, dtype=np.float64).ravel()
    n_tail = max(int(round((1.0 - alpha) * arr.size)), 1)
    sorted_l = np.sort(arr)
    var_alpha = float(sorted_l[-n_tail])
    cvar_alpha = float(np.mean(sorted_l[-n_tail:]))
    return var_alpha, cvar_alpha


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down multiple-testing adjustment across m hypothesis tests."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        mult = m - rank
        running_max = max(running_max, min(1.0, p_val * mult))
        adj[orig_idx] = running_max
    return adj


def compute_mde_relative_pct(
    sd_diff: float,
    mean_base: float,
    n_seeds: int = 10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Compute two-sided Minimum Detectable Relative Effect Size at alpha=0.05 and 80% power (Section 4.1):
    MDE_80% = (t_{1-alpha/2, M-1} + t_{power, M-1}) * SD(Delta) / (sqrt(M) * Mean_base).
    """
    df = n_seeds - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    t_pow = float(stats.t.ppf(power, df))
    mde_abs = (t_crit + t_pow) * sd_diff / math.sqrt(float(n_seeds))
    return 100.0 * (mde_abs / mean_base)
