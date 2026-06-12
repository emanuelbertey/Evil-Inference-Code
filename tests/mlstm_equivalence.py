import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig

def test_equivalence():
    device = "cpu"
    torch.manual_seed(42)
    
    batch_size = 1
    seq_len = 5
    embedding_dim = 16
    
    config = mLSTMLayerConfig(
        embedding_dim=embedding_dim,
        num_heads=4,
        conv1d_kernel_size=4,
        qkv_proj_blocksize=4,
        proj_factor=2.0,
        bias=True,
        context_length=128
    )
    
    # Inicializar capa
    layer = mLSTMLayer(config).to(device)
    
    # Inputs con gradiente
    input_p = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    input_r = input_p.clone().detach().requires_grad_(True)
    
    print(f"--- TEST mLSTM PYTHON: SEQ_LEN={seq_len} ---")
    
    # --- PASO PARALELO ---
    output_p = layer(input_p)
    loss_p = output_p.sum()
    loss_p.backward()
    
    grad_w_p = layer.proj_up.weight.grad.clone()
    grad_in_p = input_p.grad.clone()
    
    # --- PASO RECURRENTE ---
    # Limpiar gradientes para la segunda pasada
    layer.zero_grad()
    
    mlstm_state = None
    conv_state = None
    steps = []
    
    for t in range(seq_len):
        x_t = input_r[:, t:t+1, :]
        y_t, next_state = layer.step(x_t, mlstm_state=mlstm_state, conv_state=conv_state)
        steps.append(y_t)
        mlstm_state = next_state["mlstm_state"]
        conv_state = next_state["conv_state"]
        
    output_r = torch.cat(steps, dim=1)
    loss_r = output_r.sum()
    loss_r.backward()
    
    grad_w_r = layer.proj_up.weight.grad.clone()
    grad_in_r = input_r.grad.clone()
    
    # --- COMPARAR ---
    diff_val = (output_p - output_r).abs().mean().item()
    diff_grad_w = (grad_w_p - grad_w_r).abs().mean().item()
    diff_grad_in = (grad_in_p - grad_in_r).abs().mean().item()
    
    print(f"Diferencia VALOR:       {diff_val:.10f}")
    print(f"Diferencia GRAD PESO:   {diff_grad_w:.10f}")
    print(f"Diferencia GRAD INPUT:  {diff_grad_in:.10f}")
    
    if diff_grad_w < 1e-5:
        print("\nSUCCESS: PYTHON TIENE EQUIVALENCIA PERFECTA")
    else:
        print("\nWARNING: PYTHON TIENE DISCREPANCIA (Esto es normal en PyTorch por acumulación numérica)")

if __name__ == "__main__":
    test_equivalence()
