import torch
import torch.nn as nn
from xlstm.components.conv import CausalConv1d, CausalConv1dConfig

def test_conv():
    torch.manual_seed(42)
    config = CausalConv1dConfig(
        feature_dim=4,
        kernel_size=3,
        causal_conv_bias=True,
    )
    conv = CausalConv1d(config)
    
    # Deterministic weights
    with torch.no_grad():
        w_vals = torch.tensor([[[0.1, 0.2, 0.3]]] * 4, dtype=torch.float32)
        conv.conv.weight.copy_(w_vals)
        conv.conv.bias.fill_(0.1)
        
    x = torch.tensor([[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0]
    ]], dtype=torch.float32) # (1, 5, 4)
    
    x.requires_grad = True
    
    # Parallel
    y_p = conv(x)
    print("Parallel Y:\n", y_p)
    loss = y_p.sum()
    loss.backward()
    
    print("\nParallel Grad X:\n", x.grad)
    
    # Recurrent
    x.grad.zero_()
    y_steps = []
    state = None
    for t in range(5):
        y_t, state = conv.step(x[:, t:t+1, :], state)
        y_steps.append(y_t)
        
    y_r = torch.cat(y_steps, dim=1)
    print("\nRecurrent Y:\n", y_r)
    loss_r = y_r.sum()
    loss_r.backward()
    
    print("\nRecurrent Grad X:\n", x.grad)
    
    print("\nDiff Y:", (y_p - y_r).abs().max().item())

test_conv()
