"""
pde_loss.py — PDE, boundary, and initial-condition losses for PINNs.

Sections:
  - Steady-state 1D heat equation (day 1)
      -d²T/dx² = Q(x),  Q(x) = π² sin(πx),  T(0)=T(1)=0
      Analytical: T(x) = sin(πx)

  - Transient   1D heat equation (day 2)
      ∂T/∂t = α ∂²T/∂x²  on x∈[0,1], t∈[0, t_final]
      T(0,t) = T(1,t) = 0            (Dirichlet BCs)
      T(x,0) = sin(πx)                (initial condition)
      Analytical: T(x,t) = sin(πx) · exp(-α π² t)

The PINN is trained to minimize weighted sums of these residual/constraint
losses.
"""

import math
import torch


# ============================================================
# Steady-state 1D heat equation  (day 1 — do not modify)
# ============================================================
def heat_source(x: torch.Tensor) -> torch.Tensor:
    """Source term Q(x) = π² sin(πx). Chosen so T(x)=sin(πx) is exact."""
    return (math.pi ** 2) * torch.sin(math.pi * x)


def pde_residual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """PDE residual r(x) = -T''(x) - Q(x) via double autograd."""
    T = model(x)

    dT_dx = torch.autograd.grad(
        outputs=T, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
    )[0]

    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx, inputs=x,
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
    )[0]

    return -d2T_dx2 - heat_source(x)


def boundary_loss(model: torch.nn.Module) -> torch.Tensor:
    """MSE penalty enforcing T(0) = T(1) = 0."""
    x_bc = torch.tensor([[0.0], [1.0]])
    T_bc = model(x_bc)
    target = torch.zeros_like(T_bc)
    return torch.mean((T_bc - target) ** 2)


# ============================================================
# Transient 1D heat equation  (day 2)
# ============================================================
def transient_residual(model: torch.nn.Module,
                       x: torch.Tensor,
                       t: torch.Tensor,
                       alpha: float = 1.0) -> torch.Tensor:
    """PDE residual r = T_t − α T_xx at collocation points (x, t).

    Uses autograd twice: once for first-order T_x and T_t, again for T_xx.
    create_graph=True on both calls so gradients flow through the loss.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    xt = torch.cat([x, t], dim=1)
    T = model(xt)

    grads = torch.autograd.grad(
        outputs=T, inputs=[x, t],
        grad_outputs=torch.ones_like(T),
        create_graph=True,
    )
    T_x, T_t = grads[0], grads[1]

    T_xx = torch.autograd.grad(
        outputs=T_x, inputs=x,
        grad_outputs=torch.ones_like(T_x),
        create_graph=True,
    )[0]

    return T_t - alpha * T_xx


def bc_loss_transient(model: torch.nn.Module,
                      t_bc: torch.Tensor) -> torch.Tensor:
    """Dirichlet BCs: T(0,t) = T(1,t) = 0 for t sampled in [0, t_final]."""
    zeros = torch.zeros_like(t_bc)
    ones = torch.ones_like(t_bc)
    T_left = model(torch.cat([zeros, t_bc], dim=1))
    T_right = model(torch.cat([ones, t_bc], dim=1))
    return T_left.pow(2).mean() + T_right.pow(2).mean()


def ic_loss(model: torch.nn.Module,
            x_ic: torch.Tensor) -> torch.Tensor:
    """Initial condition: T(x, 0) = sin(π x)."""
    t0 = torch.zeros_like(x_ic)
    T_pred = model(torch.cat([x_ic, t0], dim=1))
    T_true = torch.sin(math.pi * x_ic)
    return (T_pred - T_true).pow(2).mean()