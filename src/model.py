"""
Multi-layer perceptron (MLP) used as the trial function T_theta(x) for the PINN.

Design choices:
    - 4 hidden layers × 32 units: small enough to train fast on CPU,
      expressive enough to represent smooth 1D solutions like sin(pi*x).
    - tanh activations: smooth (C-infinity), so all derivatives d^n T / dx^n
      computed via autograd are continuous and well-behaved.
    - Xavier (Glorot) initialization: keeps activation variance ~constant
      across layers at init, preventing early vanishing/exploding gradients.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 hidden_dim: int = 32, n_hidden: int = 4):
        """
        Parameters
        ----------
        in_dim     : dimensionality of input (1 for our 1D steady problem: just x)
        out_dim    : dimensionality of output (1: temperature T)
        hidden_dim : neurons per hidden layer
        n_hidden   : number of hidden layers
        """
        super().__init__()

        layers = []

        # Input layer: R^in_dim -> R^hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())

        # (n_hidden - 1) additional hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer: R^hidden_dim -> R^out_dim  (no activation: regression)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

        # Apply Xavier initialization to every Linear layer
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Xavier-uniform init for weights, zero for biases."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x : tensor of shape (N, in_dim) — a batch of N collocation points.
        Returns : tensor of shape (N, out_dim) — predicted T at those points.
        """
        return self.net(x)