"""
model.py
--------
Fully-connected MLP for PINNs.

Now parameterized by ``in_dim`` so the same class handles:
    - steady-state 1D:  in_dim = 1  (x)
    - transient    1D:  in_dim = 2  (x, t)
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP with tanh activations and Xavier-normal init.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dim : int
        Neurons per hidden layer.
    n_hidden : int
        Number of hidden layers.
    out_dim : int
        Number of output features (1 for scalar T).
    """

    def __init__(self, in_dim: int = 1, hidden_dim: int = 32,
                 n_hidden: int = 4, out_dim: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs shape: (N, in_dim). Returns (N, out_dim)."""
        return self.net(inputs)