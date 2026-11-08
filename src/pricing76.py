"""
Black-76 futures option valuation, implied volatility inversion, and analytical Greeks
(B76-Delta, Leland volatility-adjusted Delta, and Whalley-Wilmott no-trade bands).
"""

from __future__ import annotations
import math
from typing import Union
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import torch

ArrayLike = Union[float, np.ndarray, torch.Tensor]


def b76_call_price_np(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    tau: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    r_annual: float = 0.035,
    trading_days_per_year: float = 252.0,
) -> np.ndarray:
    """Compute discounted Black-76 call option price on futures: e^{-r*tau/252} [F*Phi(d1) - K*Phi(d2)]."""
    F_arr = np.asarray(F, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

    intrinsic = np.maximum(F_arr - K_arr, 0.0)
    vol_sqrt_t = sigma_arr * np.sqrt(np.maximum(tau_arr, 1e-12))
    d1 = (np.log(np.maximum(F_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    df = np.exp(-r_annual * (tau_arr / trading_days_per_year))
    price = df * (F_arr * norm.cdf(d1) - K_arr * norm.cdf(d2))
    return np.where(tau_arr <= 1e-9, intrinsic, price)


def b76_delta_np(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    tau: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> np.ndarray:
    """Compute Black-76 practitioner call Delta: Phi(d1)."""
    F_arr = np.asarray(F, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

    vol_sqrt_t = sigma_arr * np.sqrt(np.maximum(tau_arr, 1e-12))
    d1 = (np.log(np.maximum(F_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t
    delta = norm.cdf(d1)
    return np.where(tau_arr <= 1e-9, np.where(F_arr > K_arr, 1.0, 0.0), delta)


def b76_gamma_np(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    tau: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> np.ndarray:
    """Compute Black-76 call Gamma: phi(d1) / (F * sigma * sqrt(tau))."""
    F_arr = np.asarray(F, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

    vol_sqrt_t = sigma_arr * np.sqrt(np.maximum(tau_arr, 1e-12))
    d1 = (np.log(np.maximum(F_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t
    gamma = norm.pdf(d1) / np.maximum(F_arr * vol_sqrt_t, 1e-12)
    return np.where(tau_arr <= 1e-9, 0.0, gamma)


def implied_sigma_from_price(
    F0: float,
    K: float,
    T: float,
    target_price: float,
    r_annual: float = 0.0,
    tol: float = 1e-10,
) -> float:
    """Invert Black-76 formula to obtain constant implied daily volatility sigma_imp."""
    def obj(sig: float) -> float:
        return float(b76_call_price_np(F0, K, T, sig, r_annual=r_annual)) - target_price

    return float(brentq(obj, 1e-6, 1.0, xtol=tol))


def leland_modified_sigma(sigma_imp: float, cost_c: float, dt: float = 1.0) -> float:
    """
    Leland (1985) transaction-cost-adjusted volatility:
    tilde{sigma}^2 = sigma_imp^2 * (1 + sqrt(2/pi) * 2c / (sigma_imp * sqrt(dt))).
    """
    adj = 1.0 + math.sqrt(2.0 / math.pi) * (2.0 * cost_c) / (sigma_imp * math.sqrt(dt))
    return sigma_imp * math.sqrt(max(adj, 1e-8))


def whalley_wilmott_band_np(
    F: np.ndarray,
    K: float,
    tau: np.ndarray,
    sigma_imp: float,
    cost_c: float,
    gamma_ww: float = 0.25,
) -> np.ndarray:
    """
    Whalley & Wilmott (1997) asymptotic no-trade band half-width:
    w_t^{WW} = (3 * c * F_t * Gamma_{B76,t}^2 / (2 * gamma_WW))^(1/3).
    """
    gamma = b76_gamma_np(F, K, tau, sigma_imp)
    inside = (3.0 * cost_c * np.asarray(F, dtype=np.float64) * (gamma ** 2)) / (2.0 * max(gamma_ww, 1e-12))
    return np.power(np.maximum(inside, 0.0), 1.0 / 3.0)


# PyTorch differentiable counterparts for TorchHedgeEnv and TorchHedgeEnvOpt
_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _std_normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT_2))


def _std_normal_pdf_torch(x: torch.Tensor) -> torch.Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def b76_call_price_torch(
    F: torch.Tensor,
    K: Union[float, torch.Tensor],
    tau: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Differentiable Black-76 call option valuation in PyTorch."""
    tau_t = torch.as_tensor(tau, dtype=F.dtype, device=F.device).clamp(min=0.0)
    sigma_t = torch.as_tensor(sigma, dtype=F.dtype, device=F.device).clamp(min=1e-12)
    K_t = torch.as_tensor(K, dtype=F.dtype, device=F.device).clamp(min=1e-12)

    intrinsic = torch.clamp(F - K_t, min=0.0)
    vol_sqrt_t = sigma_t * torch.sqrt(tau_t.clamp(min=1e-12))
    d1 = (torch.log(F.clamp(min=1e-12) / K_t) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    price = F * _std_normal_cdf_torch(d1) - K_t * _std_normal_cdf_torch(d2)
    return torch.where(tau_t <= 1e-9, intrinsic, price)


def b76_delta_torch(
    F: torch.Tensor,
    K: Union[float, torch.Tensor],
    tau: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Differentiable Black-76 Delta in PyTorch."""
    tau_t = torch.as_tensor(tau, dtype=F.dtype, device=F.device).clamp(min=0.0)
    sigma_t = torch.as_tensor(sigma, dtype=F.dtype, device=F.device).clamp(min=1e-12)
    K_t = torch.as_tensor(K, dtype=F.dtype, device=F.device).clamp(min=1e-12)

    vol_sqrt_t = sigma_t * torch.sqrt(tau_t.clamp(min=1e-12))
    d1 = (torch.log(F.clamp(min=1e-12) / K_t) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    delta = _std_normal_cdf_torch(d1)
    return torch.where(tau_t <= 1e-9, (F > K_t).to(F.dtype), delta)


def b76_gamma_torch(
    F: torch.Tensor,
    K: Union[float, torch.Tensor],
    tau: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Differentiable Black-76 Gamma in PyTorch."""
    tau_t = torch.as_tensor(tau, dtype=F.dtype, device=F.device).clamp(min=0.0)
    sigma_t = torch.as_tensor(sigma, dtype=F.dtype, device=F.device).clamp(min=1e-12)
    K_t = torch.as_tensor(K, dtype=F.dtype, device=F.device).clamp(min=1e-12)

    vol_sqrt_t = sigma_t * torch.sqrt(tau_t.clamp(min=1e-12))
    d1 = (torch.log(F.clamp(min=1e-12) / K_t) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    gamma = _std_normal_pdf_torch(d1) / (F * vol_sqrt_t).clamp(min=1e-12)
    return torch.where(tau_t <= 1e-9, torch.zeros_like(F), gamma)
