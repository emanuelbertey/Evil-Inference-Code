"""Shared metrics for evaluation across all training methods."""

import torch


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error — manual implementation (no autograd needed)."""
    diff = predictions - targets
    return (diff * diff).mean()


def accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.1
) -> float:
    """
    Regression 'accuracy': fraction of predictions within `threshold` of target.

    Parameters
    ----------
    predictions : Tensor of shape (batch, features)
    targets : Tensor of shape (batch, features)
    threshold : float
        Absolute tolerance.

    Returns
    -------
    float
        Fraction of samples within threshold.
    """
    within = (predictions - targets).abs() < threshold
    return within.float().mean().item()
