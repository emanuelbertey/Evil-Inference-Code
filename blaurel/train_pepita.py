"""
Train LSTM with PEPITA (Dellaferrera & Bhatt, 2022).

No backpropagation. Uses two forward passes: base + error-modulated.
The difference in activations drives weight updates via fixed random
feedback matrices (no weight symmetry).
"""

import torch
import time

from common.data import generate_sequence_data
from common.metrics import mse_loss, accuracy
from pepita.pepita_model import PEPITALSTMModel


def main():
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PEPITA] Device: {device}")

    input_size = 1
    hidden_size = 64
    output_size = 1
    num_layers = 2
    num_epochs = 50
    batch_size = 64
    lr = 0.009

    # --- Data ---
    X_train, Y_train = generate_sequence_data(2000, seq_length=20, device=device)
    X_test, Y_test = generate_sequence_data(500, seq_length=20, device=device)

    # --- Model ---
    model = PEPITALSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        lr=lr,
        device=device,
    )

    # --- Training ---
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Test Loss':>10} | {'Acc':>8} | {'Time':>8}")
    print("-" * 58)

    total_start = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Mini-batch training
        num_batches = X_train.shape[0] // batch_size
        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = X_train[start:end]
            y_batch = Y_train[start:end]

            loss = model.train_step(x_batch, y_batch)
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches

        # Evaluate
        test_pred = model.predict(X_test)
        test_loss = mse_loss(test_pred, Y_test).item()
        test_acc = accuracy(test_pred, Y_test, threshold=0.15)

        elapsed = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_loss:>10.6f} | {test_loss:>10.6f} | {test_acc:>7.1%} | {elapsed:>7.3f}s")

    total_time = time.time() - total_start
    print(f"\n[PEPITA] Total training time: {total_time:.2f}s")
    print(f"[PEPITA] Final MSE: {test_loss:.6f} | Accuracy: {test_acc:.1%}")

    # --- Memory report ---
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    print(f"[PEPITA] Parameters: {param_count:,} ({param_mb:.2f} MB)")
    print(f"[PEPITA] NOTE: Fixed feedback matrices B are NOT counted as trainable params")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"[PEPITA] Peak VRAM: {peak_mb:.2f} MB")


if __name__ == "__main__":
    main()
