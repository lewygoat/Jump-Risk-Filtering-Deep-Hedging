"""
Market Microstructure Simulator and Causal Online Hawkes Jump-Intensity Filter.

Implements Section 2.1 - 2.3 of the manuscript:
1. Discrete-time self-exciting Hawkes jump process with 300-step stationary burn-in:
   lambda_t - mu = exp(-beta) * (lambda_{t-1} - mu) + alpha * N_t
2. Asymmetric Kou (2002) double-exponential jump sizes J_{t,j} with martingale jump
   compensator k_bar = E[e^J - 1].
3. GFEX 8-tier statutory daily price-limit distribution ell in [0.07, 0.15] (542 trading days),
   settlement price truncation F_{t+1} = clip(F*_{t+1}, F_t(1-ell), F_t(1+ell)),
   and conditional order-book execution lock (p_lock = 1/7).
4. Optional censoring-drift neutralization (recentering truncated returns so E[F_T] = F_0)
   for the R-01 / M-1 martingale audit.
5. Causal limit-augmented threshold-Hawkes intensity proxy hat{lambda}_t (Eq. 7-8).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math
import numpy as np


# Empirical 8-tier GFEX Lithium Carbonate (LC) price-limit distribution (542 trading days)
GFEX_LIMIT_TIERS = np.array([0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15], dtype=np.float64)
GFEX_LIMIT_DAYS = np.array([87, 116, 9, 279, 7, 3, 37, 4], dtype=np.int64)
GFEX_LIMIT_PROBS = np.array(
    [0.1605, 0.2140, 0.0166, 0.5148, 0.0129, 0.0055, 0.0683, 0.0074],
    dtype=np.float64,
)


@dataclass
class MarketConfig:
    """Configuration parameters for S4 benchmark and S5 extreme stress regimes."""
    regime_name: str = "S4"
    n_paths: int = 100000
    horizon_T: int = 60
    burn_in: int = 300
    f0: float = 1.0
    strike_k: float = 1.0
    sigma_diffusive: float = 0.016144
    raw_sample_sigma: float = 0.02038  # Full-sample untrimmed return SD (Section 4.1)
    mu_d: float = 0.0
    lambda_bar: float = 0.04
    branching_n: float = 0.30
    beta_decay: float = 0.20
    kou_p_up: float = 0.40
    kou_eta_up: float = 40.0
    kou_eta_dn: float = 25.0
    jump_scale: float = 1.0
    uniform_limit: Optional[float] = None  # None -> sample from 8-tier GFEX empirical law
    p_lock: float = 1.0 / 7.0
    filter_k: float = 3.0
    neutralize_censoring_drift: bool = False

    @property
    def mu_base(self) -> float:
        return self.lambda_bar * (1.0 - self.branching_n)

    @property
    def alpha_excite(self) -> float:
        return self.branching_n * (1.0 - math.exp(-self.beta_decay))

    @property
    def eff_eta_up(self) -> float:
        return self.kou_eta_up / self.jump_scale

    @property
    def eff_eta_dn(self) -> float:
        return self.kou_eta_dn / self.jump_scale

    @property
    def kou_compensator_k_bar(self) -> float:
        """Martingale jump compensator k_bar = E[exp(J) - 1] under Kou double-exponential jumps."""
        eu = self.eff_eta_up
        ed = self.eff_eta_dn
        if eu <= 1.0:
            raise ValueError("eff_eta_up must exceed 1.0 for finite E[e^J].")
        e_exp_j = self.kou_p_up * (eu / (eu - 1.0)) + (1.0 - self.kou_p_up) * (ed / (ed + 1.0))
        return e_exp_j - 1.0


def make_s4_config(n_paths: int = 100000, neutralize_drift: bool = False) -> MarketConfig:
    """Create the primary S4 benchmark market configuration."""
    return MarketConfig(
        regime_name="S4",
        n_paths=n_paths,
        branching_n=0.30,
        beta_decay=0.20,
        lambda_bar=0.04,
        jump_scale=1.0,
        uniform_limit=None,
        p_lock=1.0 / 7.0,
        neutralize_censoring_drift=neutralize_drift,
    )


def make_s5_config(n_paths: int = 20000, neutralize_drift: bool = False) -> MarketConfig:
    """Create the S5 out-of-distribution (OOD) extreme stress market configuration."""
    return MarketConfig(
        regime_name="S5",
        n_paths=n_paths,
        branching_n=0.60,
        beta_decay=0.20,
        lambda_bar=0.04,
        jump_scale=1.5,
        uniform_limit=0.07,
        p_lock=1.0 / 7.0,
        neutralize_censoring_drift=neutralize_drift,
    )


def _sample_kou_jump_sums(
    n_jumps: np.ndarray,
    p_up: float,
    eta_up: float,
    eta_dn: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample compound sum_{j=1}^{N} J_j for an array of Poisson jump counts N."""
    jump_sum = np.zeros(n_jumps.shape, dtype=np.float64)
    active_idx = np.flatnonzero(n_jumps > 0)
    if active_idx.size == 0:
        return jump_sum

    counts = n_jumps.ravel()[active_idx]
    max_c = int(counts.max())
    for c_step in range(1, max_c + 1):
        mask = counts >= c_step
        idx_sub = active_idx[mask]
        n_sub = idx_sub.size
        is_up = rng.random(n_sub) < p_up
        exp_up = rng.exponential(scale=1.0 / eta_up, size=n_sub)
        exp_dn = rng.exponential(scale=1.0 / eta_dn, size=n_sub)
        j_draw = np.where(is_up, exp_up, -exp_dn)
        jump_sum.ravel()[idx_sub] += j_draw
    return jump_sum


def simulate_market_paths(
    cfg: MarketConfig,
    seed: int = 9001,
) -> Dict[str, np.ndarray]:
    """
    Simulate discrete-time Hawkes jump-diffusion futures trajectories with GFEX price limits,
    order-book execution locks, and causal online threshold-Hawkes intensity filtering.
    """
    rng = np.random.default_rng(seed)
    N = cfg.n_paths
    T = cfg.horizon_T
    burn = cfg.burn_in
    mu = cfg.mu_base
    alpha = cfg.alpha_excite
    decay = math.exp(-cfg.beta_decay)
    k_bar = cfg.kou_compensator_k_bar

    # 1. Draw pathwise price-limit ratio ell_i
    if cfg.uniform_limit is not None:
        limit_ratio = np.full(N, float(cfg.uniform_limit), dtype=np.float64)
    else:
        probs = GFEX_LIMIT_PROBS / GFEX_LIMIT_PROBS.sum()
        limit_ratio = rng.choice(GFEX_LIMIT_TIERS, size=N, p=probs)

    # 2. Stationary burn-in for Hawkes intensity lambda_t (300 steps = 60 relaxation lifetimes)
    lam_curr = np.full(N, mu, dtype=np.float64)
    for _ in range(burn):
        n_burn = rng.poisson(lam_curr)
        lam_curr = mu + decay * (lam_curr - mu) + alpha * n_burn

    # 3. Allocate arrays over t = 0..T
    F = np.ones((N, T + 1), dtype=np.float64) * cfg.f0
    F_unclipped = np.ones((N, T + 1), dtype=np.float64) * cfg.f0
    lam_true = np.zeros((N, T + 1), dtype=np.float64)
    lam_hat = np.zeros((N, T + 1), dtype=np.float64)
    sigma_ewma = np.zeros((N, T + 1), dtype=np.float64)
    log_returns = np.zeros((N, T), dtype=np.float64)
    jump_counts = np.zeros((N, T), dtype=np.int32)
    limit_hit = np.zeros((N, T), dtype=bool)
    locked_up = np.zeros((N, T), dtype=bool)
    locked_dn = np.zeros((N, T), dtype=bool)
    consec_lock_days = np.zeros((N, T + 1), dtype=np.int32)
    filter_indicator = np.zeros((N, T), dtype=bool)

    lam_true[:, 0] = lam_curr
    A_hat = np.zeros(N, dtype=np.float64)
    lam_hat[:, 0] = mu
    v_ewma = np.full(N, cfg.sigma_diffusive ** 2, dtype=np.float64)
    sigma_ewma[:, 0] = cfg.sigma_diffusive

    for t in range(T):
        lt = lam_true[:, t]
        n_t = rng.poisson(lt)
        jump_counts[:, t] = n_t
        j_sum = _sample_kou_jump_sums(n_t, cfg.kou_p_up, cfg.eff_eta_up, cfg.eff_eta_dn, rng)
        eps = rng.standard_normal(N)

        # Latent log-return r*_{t+1} (Eq. 1)
        r_star = (cfg.mu_d - 0.5 * (cfg.sigma_diffusive ** 2) - lt * k_bar) + cfg.sigma_diffusive * eps + j_sum
        F_unclipped[:, t + 1] = F_unclipped[:, t] * np.exp(r_star)

        # Price-limit truncation (Eq. 5)
        F_prev = F[:, t]
        F_star = F_prev * np.exp(r_star)
        low_b = F_prev * (1.0 - limit_ratio)
        high_b = F_prev * (1.0 + limit_ratio)
        hit_up = F_star >= high_b
        hit_dn = F_star <= low_b
        hit_any = hit_up | hit_dn
        limit_hit[:, t] = hit_any

        F_next = np.clip(F_star, low_b, high_b)
        if cfg.neutralize_censoring_drift:
            # Cross-sectional martingale recentering so E[F_{t+1}] = F_0
            mean_f_next = float(np.mean(F_next))
            if mean_f_next > 0.0:
                F_next = F_next * (cfg.f0 / mean_f_next)

        F[:, t + 1] = F_next
        r_obs = np.log(F_next / F_prev)
        log_returns[:, t] = r_obs

        # Order-book directional execution lock conditional on limit hit (p_lock = 1/7)
        lock_draw = rng.random(N) < cfg.p_lock
        l_up = hit_up & lock_draw
        l_dn = hit_dn & lock_draw
        locked_up[:, t] = l_up
        locked_dn[:, t] = l_dn
        consec_lock_days[:, t + 1] = np.where(l_up | l_dn, consec_lock_days[:, t] + 1, 0)

        # Causal online threshold-Hawkes filter update (Eq. 7-8)
        # Uses observable return relative to diffusive sigma and limit_hit indicator
        i_hat = (np.abs(r_obs) > cfg.filter_k * cfg.sigma_diffusive) | hit_any
        filter_indicator[:, t] = i_hat
        A_hat = decay * A_hat + alpha * i_hat.astype(np.float64)
        lam_hat[:, t + 1] = mu + A_hat

        # True Hawkes recursion for next step (Eq. 2)
        lam_true[:, t + 1] = mu + decay * (lt - mu) + alpha * n_t

        # EWMA volatility update: v_{t+1} = 0.94 * v_t + 0.06 * r_{t+1}^2
        v_ewma = 0.94 * v_ewma + 0.06 * (r_obs ** 2)
        sigma_ewma[:, t + 1] = np.sqrt(np.maximum(v_ewma, 1e-12))

    return {
        "F": F,
        "F_unclipped": F_unclipped,
        "lam_true": lam_true,
        "lam_hat": lam_hat,
        "sigma_ewma": sigma_ewma,
        "log_returns": log_returns,
        "jump_counts": jump_counts,
        "limit_ratio": limit_ratio,
        "limit_hit": limit_hit,
        "locked_up": locked_up,
        "locked_dn": locked_dn,
        "consec_lock_days": consec_lock_days,
        "filter_indicator": filter_indicator,
    }


def evaluate_filter_confusion_matrix(
    jump_counts: np.ndarray,
    log_returns: np.ndarray,
    limit_hit: np.ndarray,
    sigma_scale: float,
    k_threshold: float = 3.0,
) -> Dict[str, float]:
    """Compute TP, FP, FN, TN, Precision, and Recall for the online jump detector."""
    true_jump = jump_counts > 0
    detected = (np.abs(log_returns) > k_threshold * sigma_scale) | limit_hit
    tp = int(np.sum(detected & true_jump))
    fp = int(np.sum(detected & (~true_jump)))
    fn = int(np.sum((~detected) & true_jump))
    tn = int(np.sum((~detected) & (~true_jump)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "k": k_threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "total_detections": tp + fp,
        "true_jump_days": tp + fn,
        "precision": precision,
        "recall": recall,
    }
