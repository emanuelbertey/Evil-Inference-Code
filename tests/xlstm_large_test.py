import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# --- MOCKING mlstm_kernels BEFORE IMPORTING THE MODEL ---
# Esto evita tocar model.py y permite que el test corra con el backend nativo.
class MockKernels:
    @dataclass
    class mLSTMBackendConfig:
        chunkwise_kernel: str = "parallel"
        sequence_kernel: str = "native"
        step_kernel: str = "native"
        mode: str = "train"
        chunk_size: int = 64
        return_last_states: bool = True
        autocast_kernel_dtype: str = "float32"
        eps: float = 1e-6
        inference_state_dtype: str = "float32"

    class mLSTMBackend(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            from xlstm.blocks.mlstm import backends as native_backends
            self.native = native_backends

        def forward(self, q, k, v, i, f, c_initial=None, n_initial=None, m_initial=None):
            if i.ndim == 3: i = i.unsqueeze(-1)
            if f.ndim == 3: f = f.unsqueeze(-1)
            B, NH, S, DH = q.shape
            
            if S > 1 and c_initial is None:
                # Parallel path
                return self.native.parallel_stabilized_simple(
                    queries=q, keys=k, values=v, igate_preact=i, fgate_preact=f, eps=self.config.eps
                ), None
            else:
                # Recurrent path
                if c_initial is None:
                    dh_qk, dh_v = q.shape[3], v.shape[3]
                    c_initial = torch.zeros((B, NH, dh_qk, dh_v), device=q.device, dtype=q.dtype)
                    n_initial = torch.zeros((B, NH, dh_qk, 1), device=q.device, dtype=q.dtype)
                    m_initial = torch.zeros((B, NH, 1, 1), device=q.device, dtype=q.dtype)
                
                curr_c, curr_n, curr_m = c_initial, n_initial, m_initial
                h_list = []
                for t in range(S):
                    h_t, (curr_c, curr_n, curr_m) = self.native.recurrent_step_stabilized_simple(
                        c_state=curr_c, n_state=curr_n, m_state=curr_m,
                        q=q[:, :, t:t+1], k=k[:, :, t:t+1], v=v[:, :, t:t+1],
                        igate_preact=i[:, :, t:t+1], fgate_preact=f[:, :, t:t+1], eps=self.config.eps
                    )
                    h_list.append(h_t)
                return torch.cat(h_list, dim=2), (curr_c, curr_n, curr_m)

# Inyectamos el mock en sys.modules para que el import en model.py funcione
mock_mod = type(sys)('mlstm_kernels')
mock_mod_torch = type(sys)('mlstm_kernels.torch')
mock_mod_backend = type(sys)('mlstm_kernels.torch.backend_module')
sys.modules['mlstm_kernels'] = mock_mod
sys.modules['mlstm_kernels.torch'] = mock_mod_torch
sys.modules['mlstm_kernels.torch.backend_module'] = mock_mod_backend

mock_mod_backend.mLSTMBackendConfig = MockKernels.mLSTMBackendConfig
mock_mod_backend.mLSTMBackend = MockKernels.mLSTMBackend
mock_mod_backend.ChunkwiseKernelType = str
mock_mod_backend.SequenceKernelType = str
mock_mod_backend.StepKernelType = str
mock_mod_backend.DtypeType = str
mock_mod_backend.BackendModeType = str

# --- NOW WE CAN IMPORT THE ACTUAL MODEL ---
from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

def test_xlstm_large_comprehensive():
    print("=== xLSTM Large COMPREHENSIVE TEST (PYTHON - NATIVE FALLBACK) ===")
    device = torch.device("cpu")
    
    batch_size, seq_len, embedding_dim, vocab_size = 1, 16, 32, 64
    
    config = xLSTMLargeConfig(
        embedding_dim=embedding_dim, num_heads=4, num_blocks=2, vocab_size=vocab_size,
        use_bias=True, return_last_states=True
    )
        
    model = xLSTMLarge(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    fixed_x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print("\n--- Phase 1: Training on Copy Task (100 steps) ---")
    for i in range(1, 101):
        optimizer.zero_grad()
        logits, _ = model(fixed_x)
        loss = loss_fn(logits.view(-1, vocab_size), fixed_x.view(-1))
        loss.backward()
        optimizer.step()
        if i % 20 == 0 or i == 1:
            print(f"Step {i:3}: Loss = {loss.item():.8f}")
            
    print("\n--- Phase 2: Equivalence (Parallel vs Recurrent) ---")
    model.eval()
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits_p, _ = model(test_input)
        recurrent_logits, state = [], None
        for t in range(seq_len):
            y_t, state = model(test_input[:, t:t+1], state=state)
            recurrent_logits.append(y_t)
        logits_r = torch.cat(recurrent_logits, dim=1)
        diff = torch.abs(logits_p - logits_r)
        print(f"Max Diff: {diff.max().item():.12f} | Mean Diff: {diff.mean().item():.12f}")
        if diff.mean().item() < 1e-4: print("DONE: EQUIVALENCE PASSED")

    print("\n--- Phase 3: Gradient Norms (256 steps) ---")
    model.train()
    long_input = torch.randint(0, vocab_size, (batch_size, 256), device=device)
    logits_long, _ = model(long_input)
    loss_long = (logits_long**2).mean()
    print(f"Loss (256 steps): {loss_long.item():.8f}")
    optimizer.zero_grad()
    loss_long.backward()

    # Imprimimos gradientes
    print(f"LM Head Grad Norm: {model.lm_head.weight.grad.norm().item():.10f}")
    print(f"Embedding Grad Norm: {model.embedding.weight.grad.norm().item():.10f}")
    print("DONE: Gradients computed successfully.")

if __name__ == "__main__":
    test_xlstm_large_comprehensive()
