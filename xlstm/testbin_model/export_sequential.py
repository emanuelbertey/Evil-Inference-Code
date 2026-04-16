import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import struct
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- MOCKING mlstm_kernels (Necesario para CPU) ---
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
                return self.native.parallel_stabilized_simple(
                    queries=q, keys=k, values=v, igate_preact=i, fgate_preact=f, eps=self.config.eps
                ), None
            else:
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

from tokenizers import Tokenizer
from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

def export_sequential():
    device = torch.device('cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoint.pt")
    tokenizer_path = os.path.abspath(os.path.join(current_dir, "..", "..", "rust", "tokenizer.json"))
    bin_path = os.path.join(current_dir, "sequential_data.bin")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    config = xLSTMLargeConfig(
        embedding_dim=128, num_heads=2, num_blocks=2, vocab_size=vocab_size,
        use_bias=False, weight_mode="single", chunk_size=16,
        return_last_states=True # <--- CRITICO para que devuelva el state
    )

    model = xLSTMLarge(config).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # --- PRUEBA SECUENCIAL: 50 PASOS PARA VER DEGRADACIÓN ---
    steps = 50
    # Generamos 50 tokens aleatorios del vocabulario real del modelo
    tokens_input = [random.randint(0, vocab_size - 1) for _ in range(steps)] 
    
    all_step_logits = []
    state = None
    
    print(f"Ejecutando inferencia secuencial de {steps} pasos en Python...")
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([[tokens_input[i]]], dtype=torch.long, device=device)
            # Como return_last_states=True, ahora SI devuelve la tupla (logits, state)
            logits, state = model(t, state)
            all_step_logits.append(logits) # shape [1, 1, Vocab]

    # Guardamos en binario (Weights + Secuencia X + Lista de Y)
    state_dict = model.state_dict()
    with open(bin_path, 'wb') as f:
        # 1. Metadatos de la prueba
        f.write(struct.pack('<I', steps)) # Num pasos
        
        # 2. X de entrada (los 10 tokens)
        f.write(struct.pack('<I', len(tokens_input)))
        f.write(np.array(tokens_input, dtype=np.int32).tobytes())

        # 3. Lista de Logits Y (uno por cada paso)
        for logit in all_step_logits:
            data = logit.float().cpu().numpy().tobytes()
            f.write(struct.pack('<I', len(data)))
            f.write(data)

        # 4. State Dict completo
        f.write(struct.pack('<I', len(state_dict)))
        for name, tensor in state_dict.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(tensor.shape)))
            for s in tensor.shape: f.write(struct.pack('<I', s))
            f.write(struct.pack('<I', len(tensor.float().cpu().numpy().tobytes())))
            f.write(tensor.float().cpu().numpy().tobytes())

    print(f"¡Binario Secuencial exportado en {bin_path}!")

if __name__ == "__main__":
    export_sequential()
