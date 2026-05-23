"""
Manual LSTM cell implementation.

Designed to be explicit at the tensor level for easy porting to Burn (Rust).
No autograd dependency — weights are plain tensors with requires_grad=False
by default (the training method decides how to update them).
"""

import torch
import torch.nn.functional as F


class LSTMCell:
    """
    Single LSTM cell operating on one time-step.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input vector.
    hidden_size : int
        Dimensionality of the hidden state.
    device : torch.device
        Device to place tensors on.
    """

    def __init__(self, input_size: int, hidden_size: int, device: torch.device):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Combined weight matrix for [input_gate, forget_gate, cell_gate, output_gate]
        # Using a slightly larger scale for FF/PEPITA to ensure initial activity
        import torch.nn as nn
        scale = 1.0 / (hidden_size ** 0.5)
        self.weight = nn.Parameter(
            torch.randn(4 * hidden_size, input_size + hidden_size, device=device) * 0.5
        )
        self.bias = nn.Parameter(
            torch.zeros(4 * hidden_size, device=device)
        )

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time-step.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size)
        h_prev : Tensor of shape (batch, hidden_size)
        c_prev : Tensor of shape (batch, hidden_size)

        Returns
        -------
        h_next : Tensor of shape (batch, hidden_size)
        c_next : Tensor of shape (batch, hidden_size)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_size + hidden_size)

        # Linear projection: gates = combined @ weight^T + bias
        gates = combined @ self.weight.t() + self.bias  # (batch, 4 * hidden_size)

        # Split into 4 gates
        hs = self.hidden_size
        i_gate = torch.sigmoid(gates[:, 0:hs])
        f_gate = torch.sigmoid(gates[:, hs:2 * hs])
        g_gate = torch.tanh(gates[:, 2 * hs:3 * hs])
        o_gate = torch.sigmoid(gates[:, 3 * hs:4 * hs])

        # Cell state update
        c_next = f_gate * c_prev + i_gate * g_gate

        # Hidden state
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, c) pair."""
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c

    def parameters(self) -> list[torch.Tensor]:
        """Return list of trainable tensors (for manual updates)."""
        return [self.weight, self.bias]
