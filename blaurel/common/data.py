"""
Synthetic sequence dataset for training and evaluation.

Task: predict the next value in a noisy sine-wave sequence.
Simple enough to demonstrate all three training methods clearly.
"""

import torch
from torch.utils.data import Dataset


def generate_sequence_data(
    num_samples: int = 2000,
    seq_length: int = 20,
    num_features: int = 1,
    noise: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sine-wave sequences with noise.

    Returns
    -------
    X : Tensor of shape (num_samples, seq_length, num_features)
        Input sequences.
    Y : Tensor of shape (num_samples, num_features)
        Target = next value after the sequence.
    """
    t = torch.linspace(0, 4 * 3.14159, seq_length + 1, device=device)
    # Random phase offsets per sample
    phases = torch.rand(num_samples, 1, device=device) * 2 * 3.14159
    # Random frequencies for variety
    freqs = 0.5 + torch.rand(num_samples, 1, device=device) * 1.5

    # Shape: (num_samples, seq_length + 1)
    signals = torch.sin(freqs * t.unsqueeze(0) + phases)
    signals = signals + torch.randn_like(signals) * noise

    X = signals[:, :-1].unsqueeze(-1)  # (num_samples, seq_length, 1)
    Y = signals[:, -1].unsqueeze(-1)   # (num_samples, 1)

    # Expand features if needed (duplicate channels with different noise)
    if num_features > 1:
        extra = torch.randn(num_samples, seq_length, num_features - 1, device=device) * 0.1
        X = torch.cat([X, extra], dim=-1)
        Y_extra = torch.randn(num_samples, num_features - 1, device=device) * 0.1
        Y = torch.cat([Y, Y_extra], dim=-1)

    return X, Y


class SequenceDataset(Dataset):
    """Thin wrapper around tensors for DataLoader compatibility."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
