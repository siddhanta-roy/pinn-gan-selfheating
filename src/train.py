"""
Training loop for the 1D steady heat-equation PINN.

Minimizes:
    L(theta) = L_pde  +  lambda_bc * L_bc

Success criteria:
    - Total loss drops ~4 decades over 5000 epochs
    - L2 error vs analytical sin(pi*x) <= 1e-2
    - Saves overlay + loss-history plot to outputs/pinn_first_run.png
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless SSH)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import MLP
from src.pde_loss import pde_residual, boundary_loss


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
N_COLLOCATION = 200        # collocation points on [0, 1]
N_EPOCHS      = 5000       # gradient steps
LR            = 1e-3       # Adam learning rate
LAMBDA_BC     = 100.0      # boundary loss weight
LOG_EVERY     = 200        # print diagnostics every K epochs
OUT_DIR       = "outputs"
OUT_PLOT      = os.path.join(OUT_DIR, "pinn_first_run.png")


def analytical(x: torch.Tensor) -> torch.Tensor:
    """Ground-truth solution T(x) = sin(pi * x)."""
    return torch.sin(math.pi * x)


def l2_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Relative L2 error ||pred - truth||_2 / ||truth||_2."""
    return (torch.norm(pred - truth) / torch.norm(truth)).item()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Model + optimizer
    # --------------------------------------------------------------------- #
    model = MLP(in_dim=1, out_dim=1, hidden_dim=32, n_hidden=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------------------------------------------------- #
    # Collocation points: uniform grid on [0, 1], requires_grad for autograd
    # --------------------------------------------------------------------- #
    x_colloc = torch.linspace(0.0, 1.0, N_COLLOCATION).view(-1, 1)
    x_colloc.requires_grad_(True)

    # --------------------------------------------------------------------- #
    # Loss history for plotting
    # --------------------------------------------------------------------- #
    hist_total, hist_pde, hist_bc = [], [], []

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()

        # ----- PDE loss ------------------------------------------------- #
        r = pde_residual(model, x_colloc)         # shape (N, 1)
        loss_pde = torch.mean(r ** 2)             # MSE of residual

        # ----- Boundary loss -------------------------------------------- #
        loss_bc = boundary_loss(model)

        # ----- Total loss ----------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # Evaluation on a dense grid (no grad needed here)
    # --------------------------------------------------------------------- #
    model.eval()
    with torch.no_grad():
        x_eval = torch.linspace(0.0, 1.0, 500).view(-1, 1)
        T_pred = model(x_eval)
        T_true = analytical(x_eval)
        err = l2_error(T_pred, T_true)
    print(f"Relative L2 error vs analytical: {err:.3e}")

    # --------------------------------------------------------------------- #
    # Plot: (a) prediction vs analytical, (b) loss history
    # --------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) overlay
    ax = axes[0]
    ax.plot(x_eval.numpy(), T_true.numpy(), label="analytical  sin(πx)",
            lw=2.5, color="black")
    ax.plot(x_eval.numpy(), T_pred.numpy(), "--", label="PINN prediction",
            lw=2.0, color="tab:red")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.set_title(f"Solution overlay  |  L2 error = {err:.2e}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) loss curves
    ax = axes[1]
    ax.semilogy(hist_total, label="total", lw=1.8)
    ax.semilogy(hist_pde,   label="PDE",   lw=1.2, alpha=0.85)
    ax.semilogy(hist_bc,    label="BC",    lw=1.2, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss history")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140)
    print(f"Plot saved to: {OUT_PLOT}")


if __name__ == "__main__":
    main()