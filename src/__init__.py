"""
Microstructure-Aware Deep Hedging under Self-Exciting Hawkes Jumps and Price-Limit Locks.

Modules:
- simulator: Discrete-time Hawkes-Kou jump-diffusion simulator with 8-tier GFEX price limits,
  order-book execution locks, drift neutralization option, and causal online intensity filter.
- pricing76: Black-76 futures option valuation, implied volatility inversion, and Greeks.
- torch_env: Differentiable 1D futures hedging environment (TorchHedgeEnv).
- torch_env_opt: Differentiable 2D futures + rolling short-dated ATM option environment (TorchHedgeEnvOpt).
- networks: Direct MLP policy, multiplicative jump-modulated no-trade band (BandPolicy), and D4PG-QR networks.
- trainer: End-to-end BPTT (Rockafellar-Uryasev CVaR & Mean-Variance) and D4PG-QR training routines.
"""

__version__ = "1.0.0"
