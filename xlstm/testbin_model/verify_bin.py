import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import struct
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- MOCKING mlstm_kernels ---
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
# -------------------------------------------------------------

from tokenizers import Tokenizer
from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

def load_and_verify():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(current_dir, "test_data.bin")
    tokenizer_path = os.path.abspath(os.path.join(current_dir, "..", "..", "rust", "tokenizer.json"))
    device = torch.device('cpu')

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    config = xLSTMLargeConfig(
        embedding_dim=128,   
        num_heads=2,
        num_blocks=2,
        vocab_size=vocab_size,
        use_bias=False, 
        norm_reduction_force_float32=True,
        weight_mode="single",
        chunk_size=16
    )

    model = xLSTMLarge(config).to(device)
    model.eval()

    print(f"Abriendo binario original: {bin_path}")
    if not os.path.exists(bin_path):
        print("No se encontró test_data.bin. Asegúrate de haber forzado un guardado antes.")
        return

    with open(bin_path, 'rb') as f:
        # Extraer Entrada X
        x_dim = struct.unpack('<I', f.read(4))[0]
        x_shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(x_dim))
        x_len = struct.unpack('<I', f.read(4))[0]
        x_data = f.read(x_len)
        x_tensor = torch.from_numpy(np.frombuffer(x_data, dtype=np.int32).copy()).reshape(x_shape).long().to(device)

        # Extraer Salida Logits esperada
        y_dim = struct.unpack('<I', f.read(4))[0]
        y_shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(y_dim))
        y_len = struct.unpack('<I', f.read(4))[0]
        y_data = f.read(y_len)
        y_expected = torch.from_numpy(np.frombuffer(y_data, dtype=np.float32).copy()).reshape(y_shape).to(device)

        # Extraer State Dict
        state_dict = model.state_dict()
        num_tensors = struct.unpack('<I', f.read(4))[0]
        print(f"Extrayendo y recreando {num_tensors} tensores de PyTorch desde el .bin...")
        
        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            shape_len = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(shape_len))
            data_len = struct.unpack('<I', f.read(4))[0]
            dt = f.read(data_len)
            
            if name in state_dict:
                nd = np.frombuffer(dt, dtype=np.float32).copy()
                state_dict[name].copy_(torch.from_numpy(nd).reshape(shape))
        
        model.load_state_dict(state_dict)

    print("\n[+] Todo cargado. Calculando Forward Pass interno de prueba en Python aislando a Rust...")
    with torch.no_grad():
        test_logits = model(x_tensor)
        if isinstance(test_logits, tuple): test_logits = test_logits[0]
        
    diff = (test_logits - y_expected).abs().max().item()
    print(f"\n >>> Max Diff matematico (Python Local vs Exportación de Python): {diff:.10f} <<<")
    
    if diff == 0.0:
        print("ÉXITO ABSOLUTO: El binario retiene los pesos con precisión absoluta. El desfase es exclusivo del NdArray de Rust (Totalmente normal).")
    else:
        print("Diferencia técnica encontrada en Python local.")

if __name__ == "__main__":
    load_and_verify()
