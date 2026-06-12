import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import math
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig

def test_copy_task():
    print("=== mLSTM COPY TASK TEST (PYTHON) ===")
    device = "cpu"
    torch.manual_seed(42)
    
    batch_size = 1
    seq_len = 8
    embedding_dim = 16
    lr = 8e-4
    
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
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
    
    # generate a static "pattern" to copy
    # We use a fixed pattern so loss reduction is measurable
    fixed_x = torch.randn(batch_size, seq_len, embedding_dim)
    
    for i in range(1, 101):
        optimizer.zero_grad()
        output = layer(fixed_x)
        
        # Loss: how well it reconstructs its own input
        loss = torch.nn.functional.mse_loss(output, fixed_x)
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Step {i:3}: Loss: {loss.item():.8f}")

    print(f"\nFINAL LOSS (Python): {loss.item():.8f}")

if __name__ == "__main__":
    test_copy_task()
