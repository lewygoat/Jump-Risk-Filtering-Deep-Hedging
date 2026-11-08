"""
Policy Networks for Microstructure-Aware Deep Hedging and Distributional Actor-Critic (D4PG-QR).

Implements Section 3.2 - 3.3 of the manuscript:
1. DirectPolicy: [128, 128] ReLU MLP mapping causal state vector x_t in R^d to target hedge
   delta_t^{target} = delta_max * tanh(f_psi(x_t)) (Eq. 11).
2. BandPolicy: Multiplicative jump-intensity-modulated no-trade band architecture (Eq. 12-13):
   - Center network f_psi(x_t): [128, 128] ReLU MLP -> c_t = delta_max * tanh(f_psi(x_t))
   - Baseline half-width network g_phi(x_{t,-lambda}): [32, 32] ReLU MLP (output weights=0, bias=-3.0)
   - Learnable scalar modulation parameter theta in R (initialized at theta_0 = 0.0):
     w_t(x_t) = softplus(g_phi(x_{t,-lambda})) * exp(theta * hat{lambda}_t)
   - Soft-projection operator:
     soft_proj(u, w) = -ReLU(u - w) + ReLU(-u - w)
3. QuantileCritic: Distributional critic with M_q = 51 quantile atoms for D4PG-QR (Eq. 15).
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_proj(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Piecewise-differentiable soft-projection onto [-w, +w] around center c_t (Eq. 13)."""
    return -F.relu(u - w) + F.relu(-u - w)


class DirectPolicy(nn.Module):
    """Unconstrained direct MLP policy for 1D or 2D Deep Hedging (Eq. 11)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (128, 128),
        action_bounds: Sequence[float] = (1.5,),
    ) -> None:
        super().__init__()
        layers = []
        in_d = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(in_d, action_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("bounds", torch.tensor(action_bounds, dtype=torch.float64))

    def forward(
        self,
        x_t: torch.Tensor,
        prev_exec: Optional[torch.Tensor] = None,
        lam_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw = self.net(x_t)
        target = torch.tanh(raw) * self.bounds
        return target, None


class BandPolicy(nn.Module):
    """
    Multiplicative jump-modulated no-trade band policy (DH_cband, DH_lband, DH_full, DH_opt_lband).
    Implements Eq. (12)-(13) in Section 3.2.
    """

    def __init__(
        self,
        state_dim: int,
        band_state_dim: int,
        action_dim: int = 1,
        center_hidden: Sequence[int] = (128, 128),
        band_hidden: Sequence[int] = (32, 32),
        action_bounds: Sequence[float] = (1.5,),
        learn_theta: bool = True,
        init_band_bias: float = -3.0,
    ) -> None:
        super().__init__()
        # Center MLP f_psi(x_t)
        c_layers = []
        in_d = state_dim
        for h in center_hidden:
            c_layers.append(nn.Linear(in_d, h))
            c_layers.append(nn.ReLU())
            in_d = h
        c_layers.append(nn.Linear(in_d, action_dim))
        self.center_net = nn.Sequential(*c_layers)

        # Baseline half-width MLP g_phi(x_{t,-lambda})
        b_layers = []
        in_b = band_state_dim
        for h in band_hidden:
            b_layers.append(nn.Linear(in_b, h))
            b_layers.append(nn.ReLU())
            in_b = h
        out_linear = nn.Linear(in_b, action_dim)
        nn.init.zeros_(out_linear.weight)
        nn.init.constant_(out_linear.bias, init_band_bias)
        b_layers.append(out_linear)
        self.band_net = nn.Sequential(*b_layers)

        # Unconstrained scalar jump-modulation parameter theta (initialized at 0.0)
        self.learn_theta = learn_theta
        if learn_theta:
            self.theta = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        else:
            self.register_buffer("theta", torch.zeros(1, dtype=torch.float64))

        self.register_buffer("bounds", torch.tensor(action_bounds, dtype=torch.float64))

    def forward(
        self,
        x_t: torch.Tensor,
        x_no_lam: torch.Tensor,
        prev_exec: torch.Tensor,
        lam_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c_t = torch.tanh(self.center_net(x_t)) * self.bounds
        base_w = F.softplus(self.band_net(x_no_lam))
        if self.learn_theta and lam_signal is not None:
            mod = torch.exp(self.theta * lam_signal)
            w_t = base_w * mod
        else:
            w_t = base_w

        u_t = prev_exec - c_t
        proj = prev_exec + soft_proj(u_t, w_t)
        target = torch.max(torch.min(proj, self.bounds), -self.bounds)
        return target, w_t


class QuantileCritic(nn.Module):
    """Distributional critic outputting M_q = 51 quantiles for D4PG-QR (Section 3.3)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (128, 128),
        num_quantiles: int = 51,
    ) -> None:
        super().__init__()
        self.num_quantiles = num_quantiles
        taus = (torch.arange(num_quantiles, dtype=torch.float64) + 0.5) / float(num_quantiles)
        self.register_buffer("taus", taus.view(1, num_quantiles))

        layers = []
        in_d = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(in_d, num_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Compute Quantile Huber Loss (Dabney et al., 2018) across M_q = 51 quantile atoms."""
    # pred_quantiles: [B, M], target_quantiles: [B, M]
    td_error = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # [B, M_pred, M_targ]
    abs_err = torch.abs(td_error)
    huber = torch.where(abs_err <= kappa, 0.5 * td_error * td_error, kappa * (abs_err - 0.5 * kappa))
    tau_weight = torch.abs(taus.unsqueeze(2) - (td_error < 0.0).to(pred_quantiles.dtype))
    loss = (tau_weight * huber / kappa).sum(dim=1).mean(dim=1)
    return loss.mean()
