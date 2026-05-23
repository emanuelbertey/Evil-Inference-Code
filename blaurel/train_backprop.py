"""
Train LSTM with standard Backpropagation (baseline).

Uses PyTorch autograd for comparison against FF and PEPITA.
"""

import torch
import torch.nn as nn
import time

from common.data import generate_sequence_data
from common.metrics import mse_loss, accuracy


class BackpropLSTM(nn.Module):
    """Standard PyTorch LSTM with autograd — the baseline."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
        return self.head(last_hidden)       # (batch, output)


def main():
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Backprop] Device: {device}")

    input_size = 1
    hidden_size = 64
    output_size = 1
    num_layers = 2
    num_epochs = 50
    batch_size = 64
    lr = 0.001

    # --- Data ---
    X_train, Y_train = generate_sequence_data(2000, seq_length=20, device=device)
    X_test, Y_test = generate_sequence_data(500, seq_length=20, device=device)

    # --- Model ---
    model = BackpropLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Training ---
    print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Acc':>8} | {'Time':>8}")
    print("-" * 42)

    total_start = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()

        # Mini-batch training
        num_batches = X_train.shape[0] // batch_size
        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = X_train[start:end]
            y_batch = Y_train[start:end]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = mse_loss(test_pred, Y_test).item()
            test_acc = accuracy(test_pred, Y_test, threshold=0.15)

        elapsed = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {test_loss:>10.6f} | {test_acc:>7.1%} | {elapsed:>7.3f}s")

    total_time = time.time() - total_start
    print(f"\n[Backprop] Total training time: {total_time:.2f}s")
    print(f"[Backprop] Final MSE: {test_loss:.6f} | Accuracy: {test_acc:.1%}")

    # --- Memory report ---
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)  # float32
    print(f"[Backprop] Parameters: {param_count:,} ({param_mb:.2f} MB)")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"[Backprop] Peak VRAM: {peak_mb:.2f} MB")


if __name__ == "__main__":
    main()
