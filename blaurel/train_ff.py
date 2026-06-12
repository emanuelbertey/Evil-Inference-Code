"""
Train LSTM with the Forward-Forward Algorithm (Hinton, 2022).

No backpropagation. Each layer trains locally using goodness-based
contrastive learning with positive (real) and negative (corrupted) data.
"""

import torch
import time

from common.data import generate_sequence_data
from common.metrics import mse_loss, accuracy
from ff.ff_model import FFLSTMModel


def main():
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FF] Device: {device}")

    input_size = 1
    hidden_size = 64
    output_size = 1
    num_layers = 2
    num_epochs = 50
    batch_size = 64
    lr = 0.005
    threshold = 2.0

    # --- Data ---
    X_train, Y_train = generate_sequence_data(2000, seq_length=20, device=device)
    X_test, Y_test = generate_sequence_data(500, seq_length=20, device=device)

    # --- Model ---
    model = FFLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        threshold=threshold,
        lr=lr,
        device=device,
    )

    # --- Training ---
    print(f"\n{'Epoch':>6} | {'FF Loss':>10} | {'Head Loss':>10} | {'Acc':>8} | {'Time':>8}")
    print("-" * 58)

    total_start = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Mini-batch training
        num_batches = X_train.shape[0] // batch_size
        epoch_ff_loss = 0.0
        epoch_head_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = X_train[start:end]
            y_batch = Y_train[start:end]

            ff_loss, head_loss = model.train_step(x_batch, y_batch)
            epoch_ff_loss += ff_loss
            epoch_head_loss += head_loss

        avg_ff = epoch_ff_loss / num_batches
        avg_head = epoch_head_loss / num_batches

        # Evaluate
        test_pred = model.predict(X_test)
        test_loss = mse_loss(test_pred, Y_test).item()
        test_acc = accuracy(test_pred, Y_test, threshold=0.15)

        elapsed = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_ff:>10.4f} | {test_loss:>10.6f} | {test_acc:>7.1%} | {elapsed:>7.3f}s")

    total_time = time.time() - total_start
    print(f"\n[FF] Total training time: {total_time:.2f}s")
    print(f"[FF] Final MSE: {test_loss:.6f} | Accuracy: {test_acc:.1%}")

    # --- Memory report ---
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    print(f"[FF] Parameters: {param_count:,} ({param_mb:.2f} MB)")
    print(f"[FF] NOTE: No activation storage needed for backward pass (saved VRAM)")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"[FF] Peak VRAM: {peak_mb:.2f} MB")


if __name__ == "__main__":
    main()
