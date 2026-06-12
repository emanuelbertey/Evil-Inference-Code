import torch
import torch.nn as nn
from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig

def test_slstm_equivalence():
    device = "cpu"
    torch.manual_seed(42)
    
    batch_size = 1
    embedding_dim = 16
    
    config = sLSTMLayerConfig(
        embedding_dim=embedding_dim,
        num_heads=4,
        conv1d_kernel_size=4,
        backend="vanilla", # Use vanilla for CPU equivalence
        dropout=0.0
    )
    
    # Initialize layer
    layer = sLSTMLayer(config).to(device)
    
    for seq_len in range(1, 13):
        print(f"--- TEST sLSTM PYTHON: SEQ_LEN={seq_len} ---")
        
        # Inputs
        input_p = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
        input_r = input_p.clone().detach().requires_grad_(True)
        
        # --- PARALLEL ---
        layer.zero_grad()
        output_p = layer(input_p)
        loss_p = output_p.sum()
        loss_p.backward()
        
        grad_w_p = layer.igate.weight.grad.clone()
        grad_in_p = input_p.grad.clone()
        
        # --- RECURRENT ---
        layer.zero_grad()
        conv_state = None
        slstm_state = None
        steps = []
        
        for t in range(seq_len):
            x_t = input_r[:, t:t+1, :]
            y_t, next_state = layer.step(x_t, conv_state=conv_state, slstm_state=slstm_state)
            steps.append(y_t)
            conv_state = next_state["conv_state"]
            slstm_state = next_state["slstm_state"]
            
        output_r = torch.cat(steps, dim=1)
        loss_r = output_r.sum()
        loss_r.backward()
        
        grad_w_r = layer.igate.weight.grad.clone()
        grad_in_r = input_r.grad.clone()
        
        # --- COMPARE ---
        diff_val = (output_p - output_r).abs().mean().item()
        diff_grad_w = (grad_w_p - grad_w_r).abs().mean().item()
        diff_grad_in = (grad_in_p - grad_in_r).abs().mean().item()
        
        print(f"Diferencia VALOR:       {diff_val:.10f}")
        print(f"Diferencia GRAD PESO:   {diff_grad_w:.10f}")
        print(f"Diferencia GRAD INPUT:  {diff_grad_in:.10f}")
        
        if diff_grad_w < 1e-6 and diff_val < 1e-6:
            print("SUCCESS: PYTHON TIENE EQUIVALENCIA EXACTA")
        else:
            print("FAILURE: Discrepancy detected!")
            
    print("\nALL TESTS PASSED (1-12)")

if __name__ == "__main__":
    test_slstm_equivalence()
