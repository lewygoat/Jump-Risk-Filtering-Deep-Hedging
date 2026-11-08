"""
Differentiable two-instrument futures and rolling 20-day ATM call option hedging
environment (TorchHedgeEnvOpt) with dynamic implied volatility and cost multipliers.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .pricing76 import b76_call_price_torch, b76_delta_torch, b76_gamma_torch
from .torch_env import apply_directional_lock_freeze
from .networks import BandPolicy


class TorchHedgeEnvOpt:
    """Piecewise-differentiable 2D futures + rolling 20-day ATM call option hedging environment."""

    def __init__(
        self,
        horizon_T: int = 60,
        tau_h: float = 20.0,
        f0: float = 1.0,
        strike_k: float = 1.0,
        fair_p0: float = 0.058595,
        sigma_imp_0: float = 0.018822,
        cost_c_fut: float = 1e-4,
        cost_c_opt: float = 1e-4,
        kappa_iv: float = 0.0,
    ) -> None:
        self.horizon_T = horizon_T
        self.tau_h = tau_h
        self.f0 = f0
        self.strike_k = strike_k
        self.fair_p0 = fair_p0
        self.sigma_imp_0 = sigma_imp_0
        self.cost_c_fut = cost_c_fut
        self.cost_c_opt = cost_c_opt
        self.kappa_iv = kappa_iv

    def dynamic_sigma_imp(self, lam_hat_t: torch.Tensor) -> torch.Tensor:
        """Compute jump-modulated implied volatility sigma_{imp,t} = sigma_{imp,0} * (1 + kappa_iv * hat{lambda}_t)."""
        if self.kappa_iv == 0.0:
            return torch.full_like(lam_hat_t, self.sigma_imp_0)
        return self.sigma_imp_0 * (1.0 + self.kappa_iv * lam_hat_t)

    def rollout(
        self,
        policy: nn.Module,
        batch: Dict[str, torch.Tensor],
        include_lam: bool = True,
        use_oracle_lam: bool = False,
        include_limit_features: bool = False,
        include_ewma_nl: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run differentiable 60-step 2D trajectory rollout over (delta_t, eta_t)."""
        F = batch["F"]
        B = F.shape[0]
        T = self.horizon_T
        device = F.device
        dtype = F.dtype

        limit_ratio = batch["limit_ratio"]
        sigma_ewma = batch["sigma_ewma"]
        lam_hat = batch["lam_hat"]
        lam_seq = batch["lam_true"] if use_oracle_lam else lam_hat
        locked_up = batch["locked_up"]
        locked_dn = batch["locked_dn"]
        consec_lock = batch["consec_lock_days"]
        log_returns = batch["log_returns"]

        delta_prev = torch.zeros(B, dtype=dtype, device=device)
        eta_prev = torch.zeros(B, dtype=dtype, device=device)
        cum_cost_fut = torch.zeros(B, dtype=dtype, device=device)
        cum_cost_opt = torch.zeros(B, dtype=dtype, device=device)
        cum_gain_fut = torch.zeros(B, dtype=dtype, device=device)
        cum_gain_opt = torch.zeros(B, dtype=dtype, device=device)
        turn_fut = torch.zeros(B, dtype=dtype, device=device)
        turn_opt = torch.zeros(B, dtype=dtype, device=device)

        for t in range(T):
            F_t = F[:, t]
            F_next = F[:, t + 1]
            sig_imp_t = self.dynamic_sigma_imp(lam_hat[:, t])
            sig_imp_next = self.dynamic_sigma_imp(lam_hat[:, t + 1])

            V_liab_t = b76_call_price_torch(F_t, self.strike_k, float(T - t), sig_imp_t)
            W_t = self.fair_p0 - V_liab_t + cum_gain_fut + cum_gain_opt - cum_cost_fut - cum_cost_opt

            # Rolling 20-day ATM call option valuation and Gamma (Eq. 10)
            V_open_t = b76_call_price_torch(F_t, F_t, self.tau_h, sig_imp_t)
            V_close_next = b76_call_price_torch(F_next, F_t, self.tau_h - 1.0, sig_imp_next)
            gamma_roll_t = b76_gamma_torch(F_t, F_t, self.tau_h, sig_imp_t)

            tau_norm = torch.full_like(F_t, float(T - t) / float(T))
            log_m = torch.log(F_t.clamp(min=1e-12) / self.strike_k)
            w_norm = W_t / self.f0
            delta_b76 = b76_delta_torch(F_t, self.strike_k, float(T - t), sig_imp_t)
            log_ell = torch.log1p(limit_ratio)

            base_feats = [
                tau_norm,
                log_m,
                delta_prev,
                w_norm,
                delta_b76,
                sigma_ewma[:, t],
                log_ell,
                eta_prev,
                gamma_roll_t,
            ]
            if include_ewma_nl:
                # Nonlinear EWMA control features: [sigma_EWMA^2, sigma_EWMA^3, |r_t|/sigma]
                s_norm = sigma_ewma[:, t] / 0.016144
                r_prev = log_returns[:, max(t - 1, 0)] if t > 0 else torch.zeros_like(F_t)
                base_feats.extend([s_norm ** 2, s_norm ** 3, torch.abs(r_prev) / 0.016144])

            l_up_t = locked_up[:, t]
            l_dn_t = locked_dn[:, t]
            l_any_t = l_up_t | l_dn_t
            lim_feats = []
            if include_limit_features:
                lim_feats = [l_any_t.to(dtype), consec_lock[:, t].to(dtype), limit_ratio]

            lam_t = lam_seq[:, t]
            x_no_lam = torch.stack(base_feats + lim_feats, dim=-1)
            if include_lam:
                x_t = torch.stack(base_feats + [lam_t] + lim_feats, dim=-1)
            else:
                x_t = x_no_lam

            prev_pair = torch.stack([delta_prev, eta_prev], dim=-1)
            if isinstance(policy, BandPolicy):
                target_pair, _ = policy(x_t, x_no_lam, prev_pair, lam_t.unsqueeze(-1))
            else:
                target_pair, _ = policy(x_t, prev_pair, lam_t.unsqueeze(-1))

            delta_target = target_pair[:, 0]
            eta_target = target_pair[:, 1]

            # Freeze both legs when limit lock binds
            delta_exec = apply_directional_lock_freeze(delta_target, delta_prev, l_up_t, l_dn_t)
            eta_exec = torch.where(l_any_t, eta_prev, eta_target)

            d_delta = torch.abs(delta_exec - delta_prev)
            d_eta = torch.abs(eta_exec - eta_prev)
            turn_fut = turn_fut + d_delta
            turn_opt = turn_opt + d_eta

            cum_cost_fut = cum_cost_fut + self.cost_c_fut * F_t * d_delta
            cum_cost_opt = cum_cost_opt + self.cost_c_opt * V_open_t * d_eta
            cum_gain_fut = cum_gain_fut + delta_exec * (F_next - F_t)
            cum_gain_opt = cum_gain_opt + eta_exec * (V_close_next - V_open_t)

            delta_prev = delta_exec
            eta_prev = eta_exec

        # Terminal liquidation at t = T
        F_T = F[:, T]
        sig_imp_T = self.dynamic_sigma_imp(lam_hat[:, T])
        V_open_T = b76_call_price_torch(F_T, F_T, self.tau_h, sig_imp_T)
        turn_fut = turn_fut + torch.abs(delta_prev)
        turn_opt = turn_opt + torch.abs(eta_prev)
        cum_cost_fut = cum_cost_fut + self.cost_c_fut * F_T * torch.abs(delta_prev)
        cum_cost_opt = cum_cost_opt + self.cost_c_opt * V_open_T * torch.abs(eta_prev)

        V_T = torch.clamp(F_T - self.strike_k, min=0.0)
        tot_cost = cum_cost_fut + cum_cost_opt
        W_T = self.fair_p0 - V_T + cum_gain_fut + cum_gain_opt - tot_cost
        L_T = -W_T / self.f0

        return {
            "L_T": L_T,
            "W_T": W_T,
            "cost_tot": tot_cost / self.f0,
            "cost_fut": cum_cost_fut / self.f0,
            "cost_opt": cum_cost_opt / self.f0,
            "turn_fut": turn_fut,
            "turn_opt": turn_opt,
        }
