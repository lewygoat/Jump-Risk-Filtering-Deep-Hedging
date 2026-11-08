"""
Differentiable single-instrument futures hedging environment (TorchHedgeEnv)
with directional price-limit lock freezes and LockGap accounting.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from .pricing76 import b76_call_price_torch, b76_delta_torch
from .networks import BandPolicy


def apply_directional_lock_freeze(
    delta_target: torch.Tensor,
    delta_prev: torch.Tensor,
    locked_up: torch.Tensor,
    locked_dn: torch.Tensor,
    pi_fill: float = 0.0,
) -> torch.Tensor:
    """Apply directional order-book lock execution freeze (Eq. 6) with optional partial fill pi_fill."""
    frozen_up = torch.minimum(delta_target, delta_prev)
    frozen_dn = torch.maximum(delta_target, delta_prev)
    if pi_fill > 0.0:
        frozen_up = frozen_up + pi_fill * (delta_target - frozen_up)
        frozen_dn = frozen_dn + pi_fill * (delta_target - frozen_dn)

    exec_pos = torch.where(locked_up, frozen_up, delta_target)
    exec_pos = torch.where(locked_dn, frozen_dn, exec_pos)
    return exec_pos


class TorchHedgeEnv:
    """Piecewise-differentiable 1D futures hedging environment for BPTT and out-of-sample auditing."""

    def __init__(
        self,
        horizon_T: int = 60,
        f0: float = 1.0,
        strike_k: float = 1.0,
        fair_p0: float = 0.058595,
        sigma_imp: float = 0.018822,
        cost_c: float = 1e-4,
        pi_fill: float = 0.0,
    ) -> None:
        self.horizon_T = horizon_T
        self.f0 = f0
        self.strike_k = strike_k
        self.fair_p0 = fair_p0
        self.sigma_imp = sigma_imp
        self.cost_c = cost_c
        self.pi_fill = pi_fill

    def build_state(
        self,
        t: int,
        F_t: torch.Tensor,
        delta_prev: torch.Tensor,
        W_t: torch.Tensor,
        sigma_ewma_t: torch.Tensor,
        limit_ratio: torch.Tensor,
        lam_signal_t: Optional[torch.Tensor] = None,
        locked_any_t: Optional[torch.Tensor] = None,
        consec_lock_t: Optional[torch.Tensor] = None,
        include_lam: bool = True,
        include_limit_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct nested causal state vector x_t in R^d (Section 3.2):
        - Core 6D: [tau_t, ln(F_t/K), delta_{t-1}^{exec}, W_t/F_0, Delta_{B76,t}, sigma_{EWMA,t}]
        - Baseline limit ratio 1D: ln(1 + ell_t) -> d=7 (DH_nolam)
        - Jump signal 1D: hat{lambda}_t or lambda_t -> d=8 (DH1 / DH_oracle)
        - Explicit limit state 3D: [I(lock_t), consec_lock_days_t, ell_t] -> d=11 (DH_limit / DH_full)
        Returns (x_t, x_{t,-lambda}).
        """
        tau_rem = float(self.horizon_T - t)
        tau_norm = torch.full_like(F_t, tau_rem / float(self.horizon_T))
        log_m = torch.log(F_t.clamp(min=1e-12) / self.strike_k)
        w_norm = W_t / self.f0
        delta_b76 = b76_delta_torch(F_t, self.strike_k, tau_rem, self.sigma_imp)
        log_ell = torch.log1p(limit_ratio)

        base_features = [tau_norm, log_m, delta_prev, w_norm, delta_b76, sigma_ewma_t, log_ell]
        limit_features = []
        if include_limit_features and locked_any_t is not None and consec_lock_t is not None:
            limit_features = [locked_any_t.to(F_t.dtype), consec_lock_t.to(F_t.dtype), limit_ratio]

        x_no_lam = torch.stack(base_features + limit_features, dim=-1)
        if include_lam and lam_signal_t is not None:
            x_full = torch.stack(base_features + [lam_signal_t] + limit_features, dim=-1)
        else:
            x_full = x_no_lam
        return x_full, x_no_lam

    def rollout(
        self,
        policy: nn.Module,
        batch: Dict[str, torch.Tensor],
        include_lam: bool = True,
        use_oracle_lam: bool = False,
        include_limit_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run differentiable 60-step trajectory rollout and return terminal loss and microstructure metrics."""
        F = batch["F"]  # [B, T+1]
        B = F.shape[0]
        T = self.horizon_T
        device = F.device
        dtype = F.dtype

        limit_ratio = batch["limit_ratio"]
        sigma_ewma = batch["sigma_ewma"]
        lam_seq = batch["lam_true"] if use_oracle_lam else batch["lam_hat"]
        locked_up = batch["locked_up"]
        locked_dn = batch["locked_dn"]
        consec_lock = batch["consec_lock_days"]

        delta_prev = torch.zeros(B, dtype=dtype, device=device)
        cum_cost = torch.zeros(B, dtype=dtype, device=device)
        cum_gain = torch.zeros(B, dtype=dtype, device=device)
        turnover = torch.zeros(B, dtype=dtype, device=device)
        trade_count = torch.zeros(B, dtype=dtype, device=device)
        lock_gap = torch.zeros(B, dtype=dtype, device=device)

        for t in range(T):
            F_t = F[:, t]
            V_t = b76_call_price_torch(F_t, self.strike_k, float(T - t), self.sigma_imp)
            W_t = self.fair_p0 - V_t + cum_gain - cum_cost
            l_up_t = locked_up[:, t]
            l_dn_t = locked_dn[:, t]
            l_any_t = l_up_t | l_dn_t
            lam_t = lam_seq[:, t]

            x_t, x_no_lam = self.build_state(
                t=t,
                F_t=F_t,
                delta_prev=delta_prev,
                W_t=W_t,
                sigma_ewma_t=sigma_ewma[:, t],
                limit_ratio=limit_ratio,
                lam_signal_t=lam_t,
                locked_any_t=l_any_t,
                consec_lock_t=consec_lock[:, t],
                include_lam=include_lam,
                include_limit_features=include_limit_features,
            )

            if isinstance(policy, BandPolicy):
                target_out, _ = policy(x_t, x_no_lam, delta_prev.unsqueeze(-1), lam_t.unsqueeze(-1))
            else:
                target_out, _ = policy(x_t, delta_prev.unsqueeze(-1), lam_t.unsqueeze(-1))
            delta_target = target_out.squeeze(-1)

            delta_exec = apply_directional_lock_freeze(
                delta_target=delta_target,
                delta_prev=delta_prev,
                locked_up=l_up_t,
                locked_dn=l_dn_t,
                pi_fill=self.pi_fill,
            )

            # Unexecuted lock-step hedging gap (Eq. 7)
            gap_step = torch.abs(delta_target - delta_exec) * l_any_t.to(dtype)
            lock_gap = lock_gap + gap_step

            # One-step trading cost and gain (Eq. 9)
            d_pos = torch.abs(delta_exec - delta_prev)
            turnover = turnover + d_pos
            trade_count = trade_count + (d_pos > 1e-6).to(dtype)
            cum_cost = cum_cost + self.cost_c * F_t * d_pos
            cum_gain = cum_gain + delta_exec * (F[:, t + 1] - F_t)
            delta_prev = delta_exec

        # Terminal liquidation at t = T
        F_T = F[:, T]
        term_liq = torch.abs(delta_prev)
        turnover = turnover + term_liq
        cum_cost = cum_cost + self.cost_c * F_T * term_liq
        V_T = torch.clamp(F_T - self.strike_k, min=0.0)
        W_T = self.fair_p0 - V_T + cum_gain - cum_cost
        L_T = -W_T / self.f0

        return {
            "L_T": L_T,
            "W_T": W_T,
            "cost": cum_cost / self.f0,
            "gain": cum_gain / self.f0,
            "turnover": turnover,
            "trade_freq": trade_count / float(T),
            "lock_gap": lock_gap,
        }
