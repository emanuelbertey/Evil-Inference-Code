import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig

def test_grad_stability():
    print("=== mLSTM GRADIENT STABILITY TEST (PYTHON) ===")
    device = "cpu"
    torch.manual_seed(42)
    
    batch_size = 1
    seq_len = 16
    embedding_dim = 16
    lr = 5.0 # High LR to force instability
    
    config = mLSTMLayerConfig(
        embedding_dim=embedding_dim,
        num_heads=4,
        conv1d_kernel_size=4,
        qkv_proj_blocksize=4,
        proj_factor=2.0,
        bias=True,
        context_length=seq_len,
    )
    
    layer = mLSTMLayer(config).to(device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
    
    for i in range(1, 101):
        x = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
        
        optimizer.zero_grad()
        output = layer(x)
        loss = output.pow(2).mean() # Stronger gradient than sum()
        
        if torch.isnan(loss):
            print(f"Step {i:3}: ❌ LOSS IS NAN! Explosion detected.")
            break
            
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()
        
        if math.isnan(total_grad_norm) or math.isinf(total_grad_norm):
            print(f"Step {i:3}: [X] GRADIENT IS NAN/INF! Norm: {total_grad_norm}")
            break
            
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i:3}: Loss: {loss.item():.6f} | Grad Norm: {total_grad_norm:.4f} | OK")

    print("\nTest completed.")

import math
if __name__ == "__main__":
    test_grad_stability()
