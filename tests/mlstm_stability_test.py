import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import math
from xlstm.blocks.mlstm.backends import recurrent_step_stabilized_simple

def test_stability_explosion():
    print("=== mLSTM STABILITY EXPLOSION TEST (PYTHON) ===")
    device = "cpu"
    batch_size = 1
    nh = 1
    dh = 4
    
    # Initialize states
    c_state = torch.zeros((batch_size, nh, dh, dh), device=device)
    n_state = torch.zeros((batch_size, nh, dh, 1), device=device)
    m_state = torch.zeros((batch_size, nh, 1, 1), device=device)
    
    # Simple inputs
    q = torch.ones((batch_size, nh, 1, dh), device=device)
    k = torch.ones((batch_size, nh, 1, dh), device=device)
    v = torch.ones((batch_size, nh, 1, dh), device=device)
    
    # Values to test for gates (pre-activations)
    # We will test very large positive and large negative values
    test_values = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0, 50000.0]
    
    for val in test_values:
        igate_preact = torch.full((batch_size, nh, 1, 1), val, device=device)
        fgate_preact = torch.full((batch_size, nh, 1, 1), val / 2, device=device) # some variation
        
        try:
            h, (next_c, next_n, next_m) = recurrent_step_stabilized_simple(
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
                q=q, k=k, v=v,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact
            )
            
            is_nan = torch.isnan(h).any().item()
            is_inf = torch.isinf(h).any().item()
            
            status = "OK" if not (is_nan or is_inf) else "EXPLODED (NaN/Inf)"
            print(f"Pre-act: {val:8.1f} | Result Mean: {h.mean().item():.6f} | State m: {next_m.mean().item():.1f} | {status}")
            
            if is_nan or is_inf:
                print(f"   !!! Instability detected at value {val}")
                # We don't break, let's see how much it takes to die
            
            # Update states for next "step" to see accumulation
            c_state, n_state, m_state = next_c, next_n, next_m
            
        except Exception as e:
            print(f"Pre-act: {val:8.1f} | ERROR: {e}")
            break

if __name__ == "__main__":
    test_stability_explosion()
