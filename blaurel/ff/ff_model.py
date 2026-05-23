"""
Forward-Forward LSTM Model.

Stacks multiple FFLSTMLayer instances. Each layer trains independently
using its own local contrastive loss (goodness-based).
The output of each layer becomes the input to the next.
"""

import torch
from ff.ff_lstm_layer import FFLSTMLayer


class FFLSTMModel:
    """
    Multi-layer FF-LSTM for sequence prediction.

    Architecture:
        Input -> FFLSTMLayer_0 -> FFLSTMLayer_1 -> ... -> Linear head

    Each FFLSTMLayer trains locally. The linear head maps the final
    hidden state to the output prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        threshold: float = 2.0,
        lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.layers: list[FFLSTMLayer] = []
        self.device = device
        self.lr = lr

        # Build layers — first layer now takes input + target
        for i in range(num_layers):
            in_sz = (input_size + output_size) if i == 0 else hidden_size
            layer = FFLSTMLayer(in_sz, hidden_size, threshold, lr, device)
            self.layers.append(layer)

        # No linear head needed! FF predicts by maximizing goodness.


    def predict(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Hyper-fast FF Inference: Process the base sequence once, cache memory states,
        and test 50 candidates only on the final time step. Takes ~0.1 seconds.
        """
        batch_size = x_seq.size(0)
        num_cands = 50
        candidates = torch.linspace(-1.5, 1.5, num_cands, device=self.device)
        
        # 1. Process the base sequence through all layers to build up LSTM memory
        states = []
        h_input = x_seq
        for layer in self.layers:
            h_all, h, c = layer.forward_sequence(h_input)
            states.append((h, c))
            h_input = h_all
            
        # 2. Test candidates only for the final step using cached memory
        cands_expanded = candidates.view(1, num_cands, 1).expand(batch_size, num_cands, 1)
        cands_flat = cands_expanded.reshape(batch_size * num_cands, 1, 1)
        
        total_g = torch.zeros(batch_size * num_cands, device=self.device)
        
        h_input = cands_flat
        for i, layer in enumerate(self.layers):
            h_base, c_base = states[i]
            # Clone base memory 50 times for the 50 parallel candidate tests
            h_exp = h_base.unsqueeze(1).expand(batch_size, num_cands, -1).reshape(batch_size * num_cands, -1)
            c_exp = c_base.unsqueeze(1).expand(batch_size, num_cands, -1).reshape(batch_size * num_cands, -1)
            
            # Run just ONE time step
            h_out, _, _ = layer.forward_sequence(h_input, h_init=h_exp, c_init=c_exp)
            
            # Measure goodness
            g = (h_out[:, -1, :] ** 2).mean(dim=-1)
            total_g += g
            h_input = h_out
            
        # Pick the winner
        total_g = total_g.view(batch_size, num_cands)
        best_indices = total_g.argmax(dim=1)
        best_y = candidates[best_indices].unsqueeze(-1)
        return best_y

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[float, float]:
        """
        Target-Injected Forward-Forward.
        """
        batch_size, seq_len, _ = x.shape
        
        # Positive data: Append true Y as the final step
        x_pos = torch.cat([x, y.unsqueeze(1)], dim=1)

        # Negative data: Append a true Y from a DIFFERENT wave in the batch.
        # This prevents the network from cheating and forces it to learn the phase!
        y_wrong = y[torch.randperm(batch_size)]
        x_neg = torch.cat([x, y_wrong.unsqueeze(1)], dim=1)

        total_ff_loss = 0.0
        current_pos = x_pos
        current_neg = x_neg

        for layer in self.layers:
            layer_loss, h_pos, h_neg = layer.train_step(current_pos, current_neg)
            total_ff_loss += layer_loss
            current_pos = h_pos
            current_neg = h_neg

        # Simulate head loss by checking our prediction accuracy
        with torch.no_grad():
            pred_y = self.predict(x)
            head_loss = torch.nn.functional.mse_loss(pred_y, y).item()

        return total_ff_loss, head_loss

    def parameters(self) -> list[torch.Tensor]:
        """All trainable tensors across all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
