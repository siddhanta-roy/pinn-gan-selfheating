"""
PDE and boundary-condition losses for the 1D steady heat equation.

Problem:
    -d^2 T / dx^2 = Q(x),   x in [0, 1]
    Q(x) = pi^2 * sin(pi * x)
    T(0) = T(1) = 0
    Analytical solution: T(x) = sin(pi * x)

The PINN is trained to minimize:
    L(theta) = MSE( -T''(x) - Q(x) )  +  lambda * MSE( T(0), T(1) )
"""

import math
import torch


def heat_source(x: torch.Tensor) -> torch.Tensor:
    """
    Source term Q(x) = pi^2 * sin(pi * x).

    Chosen so that the analytical solution is T(x) = sin(pi * x):
        T''(x) = -pi^2 * sin(pi * x)  =>  -T''(x) = pi^2 * sin(pi * x) = Q(x). OK
    """
    return (math.pi ** 2) * torch.sin(math.pi * x)


def pde_residual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the PDE residual r(x) = -T''(x) - Q(x) at collocation points x.

    Uses torch.autograd.grad to differentiate the network output T = model(x)
    twice with respect to the input x. This is exact differentiation
    (not finite differences).

    Parameters
    ----------
    model : the neural network T_theta
    x     : collocation points, shape (N, 1), MUST have requires_grad=True

    Returns
    -------
    residual : tensor of shape (N, 1)
    """
    # Forward pass: T = T_theta(x), shape (N, 1)
    T = model(x)

    # First derivative dT/dx via autograd.
    # grad_outputs=ones is the "seed" for reverse-mode differentiation:
    # we want d(sum(T))/dx, which equals dT/dx elementwise for scalar output.
    # create_graph=True keeps the derivative itself in the computation graph
    # so we can differentiate AGAIN (for T'') and later backprop through the loss.
    dT_dx = torch.autograd.grad(
        outputs=T,
        inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
    )[0]

    # Second derivative d^2 T / dx^2, same trick applied to dT_dx.
    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx,
        inputs=x,
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
    )[0]

    # PDE residual: should be zero if the PDE is satisfied.
    residual = -d2T_dx2 - heat_source(x)
    return residual


def boundary_loss(model: torch.nn.Module) -> torch.Tensor:
    """
    MSE penalty enforcing T(0) = T(1) = 0.

    Returns a scalar loss.
    """
    # Boundary points as tensors of shape (1, 1). No requires_grad needed
    # here because we do not differentiate w.r.t. x at the boundary.
    x_bc = torch.tensor([[0.0], [1.0]])
    T_bc = model(x_bc)                    # shape (2, 1)
    target = torch.zeros_like(T_bc)       # both should be 0
    return torch.mean((T_bc - target) ** 2)