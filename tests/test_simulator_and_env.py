"""
Unit Tests for Market Microstructure Simulator, Black-76 Option Pricing,
Directional Price-Limit Execution Freeze, and Multiplicative No-Trade Band Networks.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations
import unittest
import numpy as np
import torch

from src.simulator import (
    GFEX_LIMIT_DAYS,
    GFEX_LIMIT_PROBS,
    GFEX_LIMIT_TIERS,
    make_s4_config,
    make_s5_config,
    simulate_market_paths,
)
from src.pricing76 import (
    b76_call_price_np,
    b76_delta_np,
    b76_gamma_np,
    implied_sigma_from_price,
    leland_modified_sigma,
)
from src.networks import BandPolicy, DirectPolicy, soft_proj
from src.torch_env import TorchHedgeEnv, apply_directional_lock_freeze
from src.torch_env_opt import TorchHedgeEnvOpt


class TestMicrostructureAndHedging(unittest.TestCase):
    def test_gfex_limit_tiers_sum_and_mean(self) -> None:
        """Verify 8-tier GFEX limit distribution sums to 542 days (100.00%) with mean ell_bar = 0.0934."""
        self.assertEqual(int(GFEX_LIMIT_DAYS.sum()), 542)
        self.assertAlmostEqual(float(GFEX_LIMIT_PROBS.sum()), 1.0, places=4)
        mean_ell = float(np.sum(GFEX_LIMIT_TIERS * GFEX_LIMIT_PROBS))
        self.assertAlmostEqual(mean_ell, 0.0934, places=3)

    def test_b76_implied_volatility_inversion(self) -> None:
        """Verify Black-76 call price inversion round-trip accuracy for S4 and S5."""
        p0_s4 = float(b76_call_price_np(1.0, 1.0, 60.0, 0.018822, r_annual=0.0))
        p0_s5 = float(b76_call_price_np(1.0, 1.0, 60.0, 0.020756, r_annual=0.0))
        sig_s4 = implied_sigma_from_price(1.0, 1.0, 60.0, p0_s4, r_annual=0.0)
        sig_s5 = implied_sigma_from_price(1.0, 1.0, 60.0, p0_s5, r_annual=0.0)
        self.assertAlmostEqual(sig_s4, 0.018822, places=6)
        self.assertAlmostEqual(sig_s5, 0.020756, places=6)

    def test_simulator_martingale_and_drift_recentering(self) -> None:
        """Verify untruncated martingale property and drift-neutralized recentering."""
        cfg_raw = make_s4_config(n_paths=4000, neutralize_drift=False)
        sim_raw = simulate_market_paths(cfg_raw, seed=9001)
        self.assertEqual(sim_raw["F"].shape, (4000, 61))
        self.assertAlmostEqual(float(np.mean(sim_raw["lam_true"])), 0.04, delta=0.003)

        cfg_neutral = make_s4_config(n_paths=4000, neutralize_drift=True)
        sim_neutral = simulate_market_paths(cfg_neutral, seed=9001)
        mean_ft_neutral = float(np.mean(sim_neutral["F"][:, -1]))
        self.assertAlmostEqual(mean_ft_neutral, 1.0, places=4)

    def test_directional_lock_freeze_mechanics(self) -> None:
        """Verify upper-limit lock prevents buying and lower-limit lock prevents selling (Eq. 6)."""
        prev = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        target = torch.tensor([0.9, 0.2, 0.8], dtype=torch.float64)
        l_up = torch.tensor([True, True, False])
        l_dn = torch.tensor([False, False, True])
        exec_pos = apply_directional_lock_freeze(target, prev, l_up, l_dn, pi_fill=0.0)
        # Path 0: locked_up & target(0.9) > prev(0.5) -> frozen at 0.5
        # Path 1: locked_up & target(0.2) < prev(0.5) -> allowed to sell down to 0.2
        # Path 2: locked_dn & target(0.8) > prev(0.5) -> allowed to buy up to 0.8
        np.testing.assert_allclose(exec_pos.numpy(), [0.5, 0.2, 0.8], rtol=1e-6)

    def test_1d_and_2d_differentiable_rollout_gradients(self) -> None:
        """Verify end-to-end BPTT differentiability in TorchHedgeEnv and TorchHedgeEnvOpt."""
        cfg = make_s4_config(n_paths=64)
        sim = simulate_market_paths(cfg, seed=9001)
        batch = {k: torch.as_tensor(v) for k, v in sim.items()}

        # 1D BandPolicy (DH_lband)
        policy_1d = BandPolicy(state_dim=8, band_state_dim=7, action_dim=1, action_bounds=(1.5,), learn_theta=True).double()
        env_1d = TorchHedgeEnv()
        out_1d = env_1d.rollout(policy_1d, batch, include_lam=True)
        loss_1d = out_1d["L_T"].mean()
        loss_1d.backward()
        self.assertIsNotNone(policy_1d.theta.grad)

        # 2D BandPolicy (DH_opt_lband)
        policy_2d = BandPolicy(state_dim=10, band_state_dim=9, action_dim=2, action_bounds=(1.5, 2.0), learn_theta=True).double()
        env_2d = TorchHedgeEnvOpt()
        out_2d = env_2d.rollout(policy_2d, batch, include_lam=True)
        loss_2d = out_2d["L_T"].mean()
        loss_2d.backward()
        self.assertIsNotNone(policy_2d.theta.grad)


if __name__ == "__main__":
    unittest.main()
