import torch
import torch.nn as nn
from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig
import torch.optim as optim

def run_equivalence():
    device = torch.device("cpu")
    # Port: Usar backend="vanilla" para evitar errores de CUDA
    config = sLSTMLayerConfig(
        embedding_dim=16, 
        num_heads=4, 
        backend="vanilla",
        conv1d_kernel_size=4
    )
    slstm = sLSTMLayer(config).to(device)
    
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, 16, device=device)
    
    # Parallel forward
    out_parallel, state_dict_parallel = slstm(x, return_last_state=True)
    
    # Recurrent forward (step by step)
    # En Python xLSTM, el estado no tiene un oficial "get_empty_state" en el Layer, 
    # asique lo pasamos como None para que se inicialice internamente o lo manejamos como dict.
    state = {"conv_state": None, "slstm_state": None}
    outputs = []
    for t in range(seq_len):
        x_t = x[:, t:t+1, :] # [B, 1, D]
        y_t, state = slstm.step(x_t, **state)
        outputs.append(y_t)
    out_recurrent = torch.cat(outputs, dim=1)
    
    diff = (out_parallel - out_recurrent).abs().mean().item()
    print(f"Equivalence - Output diff: {diff:.2e}")
    
    # Comparamos estados finales
    if state_dict_parallel["slstm_state"] is not None and state["slstm_state"] is not None:
        s_diff = (state_dict_parallel["slstm_state"] - state["slstm_state"]).abs().mean().item()
        print(f"Equivalence - State diff: {s_diff:.2e}")

def run_stability():
    device = torch.device("cpu")
    config = sLSTMLayerConfig(embedding_dim=16, num_heads=4, backend="vanilla")
    slstm = sLSTMLayer(config).to(device)
    x = torch.randn(2, 20, 16, device=device) * 10.0
    out = slstm(x)
    print(f"Estabilidad: |h|={out.abs().mean().item():.4f}")

def run_monotonic():
    device = torch.device("cpu")
    config = sLSTMLayerConfig(embedding_dim=8, num_heads=4, backend="vanilla")
    slstm = sLSTMLayer(config).to(device)
    x = torch.ones(1, 20, 8, device=device)
    out = slstm(x)
    prev = 0.0
    non_decrease = 0
    for t in range(20):
        val = out[0, t, :].abs().mean().item()
        if val >= prev: non_decrease += 1
        prev = val
    print(f"Monotonicidad: {non_decrease}/20")

def run_compare_lstm():
    device = torch.device("cpu")
    hidden_size = 16
    x = torch.randn(1, 20, hidden_size, device=device)
    
    config = sLSTMLayerConfig(embedding_dim=hidden_size, num_heads=4, backend="vanilla")
    slstm = sLSTMLayer(config).to(device)
    out_s = slstm(x)
    
    lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device)
    out_l, _ = lstm(x)
    print(f"Average |h|: sLSTM={out_s.abs().mean().item():.4f}, LSTM={out_l.abs().mean().item():.4f}")

def run_grad_input():
    device = torch.device("cpu")
    hidden_size = 16
    x = torch.randn(1, 10, hidden_size, device=device, requires_grad=True)
    config = sLSTMLayerConfig(embedding_dim=hidden_size, num_heads=4, backend="vanilla")
    slstm = sLSTMLayer(config).to(device)
    out = slstm(x)
    loss = out[:, -1, :].sum()
    loss.backward()
    print(f"Grad mean |dL/dx|: {x.grad.abs().mean().item():.6f}")

def run_copy_count():
    device = torch.device("cpu")
    hidden_size = 16
    config = sLSTMLayerConfig(embedding_dim=hidden_size, num_heads=4, backend="vanilla")
    slstm = sLSTMLayer(config).to(device)
    linear = nn.Linear(hidden_size, 2).to(device)
    optimizer = optim.Adam(list(slstm.parameters()) + list(linear.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        batch_size = 64
        seq_len = 12
        xs = torch.randint(0, 2, (batch_size, seq_len, 1), device=device).float()
        ys = xs[:, 0, 0].long()
        x = xs.repeat(1, 1, hidden_size)
        
        optimizer.zero_grad()
        out = slstm(x)
        logits = linear(out[:, -1, :])
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"copy_count epoch {epoch+1}: loss={loss.item():.4f}")

if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) <= 1:
        print("Modo por defecto: ejecutar TODOS los tests de sLSTM")
        run_equivalence()
        run_stability()
        run_monotonic()
        run_compare_lstm()
        run_grad_input()
        run_copy_count()
    else:
        mode = args[1]
        if mode == "equiv": run_equivalence()
        elif mode == "stability": run_stability()
        elif mode == "monotonic": run_monotonic()
        elif mode == "compare_lstm": run_compare_lstm()
        elif mode == "grad": run_grad_input()
        elif mode == "copy_count": run_copy_count()
        else: print(f"Modo inválido: {mode}")
