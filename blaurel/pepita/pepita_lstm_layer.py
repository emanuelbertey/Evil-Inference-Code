"""
PEPITA LSTM Layer.

PEPITA (PErceptive Propagation by ITerative Adaptation) trains without
weight symmetry. It uses two forward passes:

1. Base pass: standard forward, compute output error.
2. Modulated pass: re-inject the error into the input and forward again.
   The difference between the two activations drives weight updates.

Key insight: the error signal modulates the input, not the gradients.
This avoids the weight transport problem of backpropagation.
"""

import torch
from common.lstm_cell import LSTMCell


class PEPITALSTMLayer:
    """
    LSTM layer trained with the PEPITA algorithm.

    Each layer has a fixed random feedback matrix B that projects the
    output error back to the input space. This is NOT learned — it
    breaks the weight symmetry requirement of backprop.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.cell = LSTMCell(input_size, hidden_size, device)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lr = lr
        self.device = device

        # Fixed random feedback matrix: projects output error -> input space
        # CRITICAL: Scale B by input/output size to keep projection stable
        self.B = torch.randn(input_size, output_size, device=device) * (1.0 / (input_size * output_size)**0.5)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Run LSTM over a sequence, return final hidden state."""
        batch_size, seq_len, _ = x_seq.shape
        h, c = self.cell.init_hidden(batch_size)

        for t in range(seq_len):
            h, c = self.cell.forward(x_seq[:, t, :], h, c)
            # Optional: slight normalization helps PEPITA too
            h = h / (h.norm(2, dim=-1, keepdim=True) + 1e-4)

        return h

    def train_step(
        self,
        x_seq: torch.Tensor,
        output_error: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PEPITA training step for this layer."""
        # Enable local grads
        params = self.cell.parameters()
        for p in params:
            p.requires_grad = True

        # --- Pass 1: Base forward ---
        h_base = self.forward_sequence(x_seq)

        # --- Pass 2: Modulated forward (re-inject error) ---
        # STRONG injection: modulate input significantly with projection
        error_proj = output_error @ self.B.t() 
        x_modulated = x_seq + 0.5 * error_proj.unsqueeze(1) 
        h_modulated = self.forward_sequence(x_modulated)

        # --- Weight update ---
        diff = h_modulated - h_base
        # Margin-based loss for PEPITA
        loss = torch.relu(0.5 + (diff * diff).mean()).mean()
        
        loss.backward()
        
        with torch.no_grad():
            for p in params:
                torch.nn.utils.clip_grad_norm_(p, 1.0)
                p.data -= self.lr * p.grad
                p.grad.zero_()
                p.requires_grad = False

        return h_base.detach(), h_modulated.detach()

    def parameters(self) -> list[torch.Tensor]:
        """Trainable parameters (B is fixed, not trained)."""
        return self.cell.parameters()
