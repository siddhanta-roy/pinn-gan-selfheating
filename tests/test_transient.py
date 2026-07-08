"""
Regression tests for the 1D transient heat PINN.

Ensures that:
  - The collocation sampler returns tensors with correct shapes.
  - transient_residual / bc_loss_transient / ic_loss produce the right
    output shapes and remain scalar losses.
  - The analytical solution satisfies boundary and initial conditions.
  - A short training run (~500 epochs) hits a loose L2 upper bound —
    not a convergence guarantee, just a smoke test to catch regressions
    that would silently break training.
"""
import math

import torch

from src.model import MLP
from src.pde_loss import transient_residual, bc_loss_transient, ic_loss
from src.train import (
    sample_collocation,
    train_transient,
    analytical_solution,
    l2_error,
)


# ---------------------------------------------------------------- sampler
def test_sample_collocation_shapes():
    """Sampler returns four (N, 1) tensors with N matching the inputs."""
    x_pde, t_pde, t_bc, x_ic = sample_collocation(
        n_pde=64, n_bc=16, n_ic=16, t_final=0.2)
    assert x_pde.shape == (64, 1)
    assert t_pde.shape == (64, 1)
    assert t_bc.shape  == (16, 1)
    assert x_ic.shape  == (16, 1)


def test_sample_collocation_ranges():
    """x samples lie in [0,1], t samples in [0, t_final]."""
    t_final = 0.2
    x_pde, t_pde, t_bc, x_ic = sample_collocation(
        n_pde=500, n_bc=100, n_ic=100, t_final=t_final)
    assert x_pde.min() >= 0.0 and x_pde.max() <= 1.0
    assert x_ic.min()  >= 0.0 and x_ic.max()  <= 1.0
    assert t_pde.min() >= 0.0 and t_pde.max() <= t_final
    assert t_bc.min()  >= 0.0 and t_bc.max()  <= t_final


# ------------------------------------------------------------------ losses
def test_transient_residual_shape():
    """Residual has one value per (x,t) collocation point."""
    torch.manual_seed(0)
    model = MLP(in_dim=2, hidden_dim=8, n_hidden=2)
    x = torch.rand(32, 1)
    t = torch.rand(32, 1) * 0.2
    r = transient_residual(model, x, t, alpha=1.0)
    assert r.shape == (32, 1)


def test_bc_and_ic_losses_are_scalar():
    """BC and IC losses reduce to scalars (0-dim tensors)."""
    torch.manual_seed(0)
    model = MLP(in_dim=2, hidden_dim=8, n_hidden=2)
    x = torch.rand(16, 1)
    t = torch.rand(16, 1) * 0.2
    assert bc_loss_transient(model, t).ndim == 0
    assert ic_loss(model, x).ndim == 0


# ------------------------------------------------------------ analytical
def test_analytical_bcs_and_ic():
    """T(0,t)=T(1,t)=0 for all t and T(x,0)=sin(pi x)."""
    t = torch.linspace(0, 0.2, 10).unsqueeze(1)
    zeros = torch.zeros_like(t)
    ones  = torch.ones_like(t)
    assert torch.allclose(
        analytical_solution(zeros, t), torch.zeros_like(t), atol=1e-6)
    assert torch.allclose(
        analytical_solution(ones,  t), torch.zeros_like(t), atol=1e-6)

    x = torch.linspace(0, 1, 20).unsqueeze(1)
    t0 = torch.zeros_like(x)
    assert torch.allclose(
        analytical_solution(x, t0),
        torch.sin(math.pi * x),
        atol=1e-6,
    )


# ---------------------------------------------------------- training smoke
def test_smoke_train_and_l2():
    """Short training run must reduce error below a loose upper bound.

    This is NOT a convergence guarantee — the full notebook uses 5000
    epochs and lands ~1e-3. This test uses 300 epochs and asserts a much
    looser bound; it exists to catch regressions that silently break the
    training loop, not to validate accuracy.
    """
    model, history = train_transient(
        alpha=1.0, t_final=0.2,
        n_pde=500, n_bc=50, n_ic=50,
        epochs=300, lr=1e-3,
        w_pde=1.0, w_bc=10.0, w_ic=10.0,
        log_every=10_000,   # silence prints in CI
        seed=0,
    )
    err = l2_error(model, alpha=1.0, t_final=0.2)

    # Loose bound: real 5000-epoch training lands ~1e-3. This threshold
    # only trips if the trainer is genuinely broken (loss not decreasing,
    # NaN, sign flip in residual, etc.).
    assert err < 2.5e-1, f"L2 error {err:.3e} exceeded 2.5e-1 threshold"

    
# Sanity: loss actually went down (regression canary)
    start_loss = history["total"][0]
    end_loss = history["total"][-1]
    assert end_loss < 0.5 * start_loss, (
        f"Loss did not drop: start {start_loss:.3e}, end {end_loss:.3e}"
    )


    # History dict shape
    assert set(history.keys()) == {"total", "pde", "bc", "ic"}
    assert len(history["total"]) == 300