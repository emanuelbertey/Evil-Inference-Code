"""
Forward-Forward LSTM Layer.

Each layer trains locally using the "goodness" metric:
- Positive pass (real data): maximize sum of squared activations.
- Negative pass (corrupted data): minimize sum of squared activations.

No backpropagation. Weights update via a local contrastive loss.
"""

import torch
from common.lstm_cell import LSTMCell


class FFLSTMLayer:
    """
    A single LSTM layer that trains using the Forward-Forward algorithm.

    The 'goodness' of a hidden state h is defined as:
        goodness(h) = sum(h^2)  (L2 norm squared)

    Training objective per layer:
        - For positive (real) data: goodness should exceed a threshold θ.
        - For negative (fake) data: goodness should be below θ.

    Loss = log(1 + exp(-(goodness_pos - θ))) + log(1 + exp(goodness_neg - θ))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 2.0,
        lr: float = 0.003,
        device: torch.device = torch.device("cpu"),
    ):
        self.cell = LSTMCell(input_size, hidden_size, device)
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.lr = lr
        self.device = device
        
        # Local optimizer
        self.opt = torch.optim.Adam(self.cell.parameters(), lr=lr)

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Goodness: mean squared activity."""
        return (h ** 2).mean(dim=-1)

    def forward_sequence(self, x_seq: torch.Tensor, h_init=None, c_init=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning all states, without 1D normalization that destroys amplitude."""
        batch_size, seq_len, _ = x_seq.shape
        
        if h_init is None:
            h, c = self.cell.init_hidden(batch_size)
        else:
            h, c = h_init, c_init
            
        h_all = []
        for t in range(seq_len):
            h, c = self.cell.forward(x_seq[:, t, :], h, c)
            h_all.append(h)
            
        h_all = torch.stack(h_all, dim=1)
        return h_all, h, c

    def train_step(
        self, x_pos: torch.Tensor, x_neg: torch.Tensor
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Hinton FF training step focusing on the appended future step."""
        h_pos_all, _, _ = self.forward_sequence(x_pos)
        h_neg_all, _, _ = self.forward_sequence(x_neg)

        # Goodness is evaluated heavily on the final injected step
        h_pos_last = h_pos_all[:, -1, :]
        h_neg_last = h_neg_all[:, -1, :]

        g_pos = (h_pos_last ** 2).mean(dim=-1)
        g_neg = (h_neg_last ** 2).mean(dim=-1)

        # Threshold must be < 1.0 because LSTM outputs Tanh [-1, 1]
        thresh = 0.5
        loss = torch.log(1 + torch.exp(-g_pos + thresh)).mean() + \
               torch.log(1 + torch.exp(g_neg - thresh)).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), h_pos_all.detach(), h_neg_all.detach()

    def parameters(self) -> list[torch.Tensor]:
        return list(self.cell.parameters())


