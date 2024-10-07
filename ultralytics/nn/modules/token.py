"""Token module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "Token",
)


class Token(nn.Module):
    """
    Trainable token module.
    To be used at the beginning of a model.
    """

    def __init__(self, c_in=2, h=64):
        """Initializes the token module."""
        super().__init__()
        self.token = nn.Parameter(torch.randn(c_in, h), requires_grad=True)

    def forward(self, x):
        """Forward pass."""
        return self.token[x.long()]
