"""
train.py — trainers for the PINN skeleton.

Sections:
  - Steady-state 1D heat equation (day 1)
      Script-style trainer with a main() entrypoint. Produces
      outputs/pinn_first_run.png (solution overlay + loss history).

  - Transient 1D heat equation (day 2)
      Library-style trainer used by notebooks/02_transient_heat.ipynb.
      Exposes train_transient, sample_collocation, analytical_solution,
      l2_error, and plot_loss_history.
"""

import math
import os
import time
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless SSH)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import MLP
from src.pde_loss import (
    # day-1
    pde_residual, boundary_loss,
    # day-2
    transient_residual, bc_loss_transient, ic_loss,
)


# ============================================================
# Reproducibility (day-1 script defaults; day-2 seeds its own run)
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Steady-state 1D heat equation  (day 1 — do not modify)
# ============================================================
N_COLLOCATION = 200
N_EPOCHS      = 5000
LR            = 1e-3
LAMBDA_BC     = 100.0
LOG_EVERY     = 200
OUT_DIR       = "outputs"
OUT_PLOT      = os.path.join(OUT_DIR, "pinn_first_run.png")


def analytical(x: torch.Tensor) -> torch.Tensor:
    """Ground-truth steady solution T(x) = sin(pi * x)."""
    return torch.sin(math.pi * x)


def _l2_relative(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Relative L2 error ||pred - truth||_2 / ||truth||_2. Day-1 helper."""
    return (torch.norm(pred - truth) / torch.norm(truth)).item()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    model = MLP(in_dim=1, out_dim=1, hidden_dim=32, n_hidden=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_colloc = torch.linspace(0.0, 1.0, N_COLLOCATION).view(-1, 1)
    x_colloc.requires_grad_(True)

    hist_total, hist_pde, hist_bc = [], [], []

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()

        r = pde_residual(model, x_colloc)
        loss_pde = torch.mean(r ** 2)
        loss_bc = boundary_loss(model)
        loss = loss_pde + LAMBDA_BC * loss_bc

        loss.backward()
        optimizer.step()

        hist_total.append(loss.item())
        hist_pde.append(loss_pde.item())
        hist_bc.append(loss_bc.item())

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | total {loss.item():.3e} "
                  f"| pde {loss_pde.item():.3e} | bc {loss_bc.item():.3e}")

    wall = time.time() - t0
    print(f"\nTraining done in {wall:.1f} s")

    model.eval()
    with torch.no_grad():
        x_eval = torch.linspace(0.0, 1.0, 500).view(-1, 1)
        T_pred = model(x_eval)
        T_true = analytical(x_eval)
        err = _l2_relative(T_pred, T_true)
    print(f"Relative L2 error vs analytical: {err:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(x_eval.numpy(), T_true.numpy(), label="analytical  sin(πx)",
            lw=2.5, color="black")
    ax.plot(x_eval.numpy(), T_pred.numpy(), "--", label="PINN prediction",
            lw=2.0, color="tab:red")
    ax.set_xlabel("x"); ax.set_ylabel("T(x)")
    ax.set_title(f"Solution overlay  |  L2 error = {err:.2e}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(hist_total, label="total", lw=1.8)
    ax.semilogy(hist_pde,   label="PDE",   lw=1.2, alpha=0.85)
    ax.semilogy(hist_bc,    label="BC",    lw=1.2, alpha=0.85)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss history")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140)
    print(f"Plot saved to: {OUT_PLOT}")


# ============================================================
# Transient 1D heat equation  (day 2)
# ============================================================
def sample_collocation(
    n_pde: int, n_bc: int, n_ic: int, t_final: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw fresh (x,t) collocation points for one epoch."""
    x_pde = torch.rand(n_pde, 1, device=device)
    t_pde = torch.rand(n_pde, 1, device=device) * t_final
    t_bc  = torch.rand(n_bc,  1, device=device) * t_final
    x_ic  = torch.rand(n_ic,  1, device=device)
    return x_pde, t_pde, t_bc, x_ic


def train_transient(
    alpha: float = 1.0,
    t_final: float = 0.2,
    n_pde: int = 4000,
    n_bc: int = 200,
    n_ic: int = 200,
    epochs: int = 5000,
    lr: float = 1e-3,
    w_pde: float = 1.0,
    w_bc: float = 10.0,
    w_ic: float = 10.0,
    log_every: int = 500,
    seed: int = 0,
) -> Tuple[MLP, Dict[str, list]]:
    """Train the 2-input MLP on the 1D transient heat equation."""
    torch.manual_seed(seed)
    model = MLP(in_dim=2, hidden_dim=32, n_hidden=4, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"total": [], "pde": [], "bc": [], "ic": []}

    for ep in range(epochs):
        x_pde, t_pde, t_bc, x_ic = sample_collocation(
            n_pde, n_bc, n_ic, t_final)

        r = transient_residual(model, x_pde, t_pde, alpha=alpha)
        L_pde = r.pow(2).mean()
        L_bc  = bc_loss_transient(model, t_bc)
        L_ic  = ic_loss(model, x_ic)
        loss = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic

        opt.zero_grad()
        loss.backward()
        opt.step()

        history["total"].append(loss.item())
        history["pde"].append(L_pde.item())
        history["bc"].append(L_bc.item())
        history["ic"].append(L_ic.item())

        if ep % log_every == 0 or ep == epochs - 1:
            print(f"ep {ep:5d} | L={loss.item():.3e} "
                  f"(pde {L_pde.item():.2e}, "
                  f"bc {L_bc.item():.2e}, ic {L_ic.item():.2e})")

    return model, history


def analytical_solution(x: torch.Tensor, t: torch.Tensor,
                        alpha: float = 1.0) -> torch.Tensor:
    """T(x,t) = sin(π x) · exp(-α π² t). Fundamental mode of the heat eq."""
    return torch.sin(math.pi * x) * torch.exp(-alpha * math.pi ** 2 * t)


def l2_error(model: MLP, alpha: float = 1.0, t_final: float = 0.2,
             nx: int = 128, nt: int = 64) -> float:
    """Grid-based L2 error of PINN vs analytical over the (x,t) strip."""
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, t_final, nt)
    X, T_ = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack([X.flatten(), T_.flatten()], dim=1)
    with torch.no_grad():
        T_pred = model(xt).reshape(nx, nt)
    T_true = analytical_solution(X, T_, alpha=alpha)
    return torch.sqrt(((T_pred - T_true) ** 2).mean()).item()


def plot_loss_history(history: Dict[str, list], ax=None):
    """Log-scale plot of the composite loss and its components."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for k, v in history.items():
        ax.semilogy(v, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


# ============================================================
# Script entry point (day-1 behavior preserved)
# ============================================================
if __name__ == "__main__":
    main()