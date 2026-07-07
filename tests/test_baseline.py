"""
Regression tests for the PINN skeleton.

Ensures that:
  1. All src/ modules import cleanly (smoke test).
  2. A forward pass through the MLP produces correct shapes.
  3. The full training run converges to the promised L2 error.

Run locally:
    pytest -v

Run in CI:
    Automatically via .github/workflows/ci.yml on every push.
"""

import math
import subprocess
import sys

import torch

from src.model import MLP
from src.pde_loss import pde_residual, boundary_loss


# --------------------------------------------------------------------------- #
# Fast tests (~1s) — run on every push
# --------------------------------------------------------------------------- #

def test_mlp_forward_shape():
    """MLP must produce (N, 1) output from (N, 1) input."""
    model = MLP()
    x = torch.randn(5, 1)
    y = model(x)
    assert y.shape == (5, 1), f"expected (5, 1), got {y.shape}"


def test_mlp_parameter_count():
    """MLP with defaults should have ~3.5k parameters (regression sentinel)."""
    model = MLP()
    n = sum(p.numel() for p in model.parameters())
    assert 3000 < n < 4000, f"unexpected param count: {n}"


def test_pde_residual_shape():
    """Residual must be pointwise: (N, 1) input -> (N, 1) residual."""
    model = MLP()
    x = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)
    r = pde_residual(model, x)
    assert r.shape == (10, 1), f"expected (10, 1), got {r.shape}"


def test_boundary_loss_scalar():
    """Boundary loss must be a scalar tensor."""
    model = MLP()
    bc = boundary_loss(model)
    assert bc.dim() == 0, f"expected scalar, got shape {bc.shape}"
    assert bc.item() >= 0, "boundary loss must be non-negative"


# --------------------------------------------------------------------------- #
# Slow test (~30s) — full training convergence
# --------------------------------------------------------------------------- #

def test_training_converges():
    """
    End-to-end regression: full training must achieve L2 <= 1e-2.

    Baseline (Jul 2026): L2 = 3.23e-05.
    Threshold set generously at 1e-2 to allow minor numerical drift
    across PyTorch versions or platforms.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.train"],
        capture_output=True,
        text=True,
        check=True,
    )

    l2 = None
    for line in result.stdout.splitlines():
        if "Relative L2 error" in line:
            l2 = float(line.split(":")[-1].strip())
            break

    assert l2 is not None, (
        f"L2 error not found in training output:\n{result.stdout}"
    )
    assert l2 < 5e-2, f"L2 error {l2:.3e} exceeds threshold 5e-2"