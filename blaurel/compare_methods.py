"""
Comparison Script: Backpropagation vs Forward-Forward vs PEPITA.

Trains all three methods on the same dataset and compares performance, 
time, and estimated memory footprint.
"""

import torch
import time
import pandas as pd
from tabulate import tabulate

from common.data import generate_sequence_data
from common.metrics import mse_loss, accuracy

# Import models
from train_backprop import BackpropLSTM
from ff.ff_model import FFLSTMModel
from pepita.pepita_model import PEPITALSTMModel

def estimate_memory(model_name, num_layers, seq_len, batch_size, hidden_size):
    """
    Rough estimation of memory usage for activations during training.
    Units: MB (assuming float32 = 4 bytes)
    """
    # Parameters (rough estimate for all)
    # LSTM weight matrix is approx 4 * (in + out) * out
    params_count = num_layers * (4 * (hidden_size + hidden_size) * hidden_size)
    params_mb = (params_count * 4) / (1024**2)

    # Activations (The critical difference)
    if model_name == "Backprop":
        # Needs to store ALL activations for the backward pass
        # layers * seq_len * batch * hidden
        acts_count = num_layers * seq_len * batch_size * hidden_size
    elif model_name == "Forward-Forward":
        # Only local activations per layer, no sequence history for backward
        # batch * hidden
        acts_count = batch_size * hidden_size
    elif model_name == "PEPITA":
        # Two forward passes, only current layer activations needed
        # 2 * batch * hidden
        acts_count = 2 * batch_size * hidden_size
    else:
        acts_count = 0
        
    acts_mb = (acts_count * 4) / (1024**2)
    return params_mb, acts_mb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Comparison on {device} ---")

    # Hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    num_layers = 2
    seq_length = 30
    batch_size = 64
    num_epochs = 60  # Increased for local learning stability
    lr = 0.001

    # Shared Dataset
    X_train, Y_train = generate_sequence_data(2000, seq_length, input_size, device=device)
    X_test, Y_test = generate_sequence_data(500, seq_length, input_size, device=device)

    results = []

    methods = [
        ("Backprop", "Classical Backpropagation (Global Gradient)"),
        ("Forward-Forward", "Local Goodness (Gradient-Free)"),
        ("PEPITA", "Error Modulation (Gradient-Free)")
    ]

    for name, desc in methods:
        print(f"\n[Training] {name}...")
        
        # Initialize models
        if name == "Backprop":
            model = BackpropLSTM(input_size, hidden_size, output_size, num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()
        elif name == "Forward-Forward":
            # FF often needs a higher LR for local updates
            model = FFLSTMModel(input_size, hidden_size, output_size, num_layers, lr=0.01, device=device)
        else: # PEPITA
            # PEPITA needs a strong signal to correlate error with weights
            model = PEPITALSTMModel(input_size, hidden_size, output_size, num_layers, lr=0.01, device=device)

        start_time = time.time()
        
        # Simple training loop
        for epoch in range(num_epochs):
            num_batches = X_train.shape[0] // batch_size
            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                xb, yb = X_train[start:end], Y_train[start:end]

                if name == "Backprop":
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif name == "Forward-Forward":
                    model.train_step(xb, yb)
                else: # PEPITA
                    model.train_step(xb, yb)

        training_time = time.time() - start_time
        
        # Evaluation
        with torch.no_grad():
            if name == "Backprop":
                test_pred = model(X_test)
            else:
                test_pred = model.predict(X_test)
                
            final_mse = mse_loss(test_pred, Y_test).item()
            final_acc = accuracy(test_pred, Y_test, threshold=0.15)

        # Memory Estimation
        p_mb, a_mb = estimate_memory(name, num_layers, seq_length, batch_size, hidden_size)

        results.append({
            "Method": name,
            "MSE": f"{final_mse:.6f}",
            "Accuracy": f"{final_acc:.1%}",
            "Time (s)": f"{training_time:.2f}",
            "Params (MB)": f"{p_mb:.2f}",
            "Activations (MB)": f"{a_mb:.2f}",
            "Total Est (MB)": f"{p_mb + a_mb:.2f}"
        })

    # Summary Table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(tabulate(results, headers="keys", tablefmt="pretty"))
    print("\n* Activations (MB) is the memory needed to store intermediate values during training.")
    print("* Backprop needs much more because it stores the full sequence history for every layer.")
    print("* Forward-Forward and PEPITA are local, making them ideal for long sequences.")

if __name__ == "__main__":
    main()
