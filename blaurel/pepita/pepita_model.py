"""
PEPITA LSTM Model.

Stacks PEPITALSTMLayer instances. The output error from the linear head
is propagated back through fixed random matrices (B) — no learned
backward path, no weight symmetry.
"""

import torch
from pepita.pepita_lstm_layer import PEPITALSTMLayer


class PEPITALSTMModel:
    """
    Multi-layer PEPITA-LSTM for sequence prediction.

    Architecture:
        Input -> PEPITALSTMLayer_0 -> PEPITALSTMLayer_1 -> ... -> Linear head

    The output error is computed once at the head, then each layer
    uses its own fixed random feedback matrix to modulate its input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.layers: list[PEPITALSTMLayer] = []
        self.device = device
        self.lr = lr
        self.output_size = output_size

        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            layer = PEPITALSTMLayer(in_sz, hidden_size, output_size, lr, device)
            self.layers.append(layer)

        # Linear prediction head
        scale = (1.0 / hidden_size) ** 0.5
        self.head_weight = torch.randn(output_size, hidden_size, device=device) * scale
        self.head_bias = torch.zeros(output_size, device=device)

    def predict(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass.

        Parameters
        ----------
        x_seq : Tensor (batch, seq_length, input_size)

        Returns
        -------
        output : Tensor (batch, output_size)
        """
        h = x_seq
        for layer in self.layers:
            h_out = layer.forward_sequence(h)
            h = h_out.unsqueeze(1)  # (batch, 1, hidden_size)

        return h_out @ self.head_weight.t() + self.head_bias

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """
        Full PEPITA training step.

        1. Forward pass to compute output and error.
        2. Each layer does base + modulated pass using the output error.
        3. Head weights updated via direct error.

        Parameters
        ----------
        x : Tensor (batch, seq_length, input_size)
        y : Tensor (batch, output_size)

        Returns
        -------
        loss : float — MSE loss value
        """
        # --- Step 1: Full forward to get output error ---
        prediction = self.predict(x)
        output_error = prediction - y  # (batch, output_size)
        loss = (output_error * output_error).mean().item()

        # --- Step 2: Train each layer with PEPITA ---
        current_input = x
        for layer in self.layers:
            h_base, h_modulated = layer.train_step(current_input, output_error)
            # Next layer gets base hidden as single-step sequence
            current_input = h_base.unsqueeze(1)

        # --- Step 3: Update head weights ---
        # Use a higher LR for the head to speed up convergence
        head_lr = 0.05
        batch_size = x.shape[0]
        grad_w = output_error.t() @ h_base / batch_size
        grad_b = output_error.mean(dim=0)
        self.head_weight.data -= head_lr * grad_w
        self.head_bias.data -= head_lr * grad_b

        return loss

    def parameters(self) -> list[torch.Tensor]:
        """All trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend([self.head_weight, self.head_bias])
        return params
