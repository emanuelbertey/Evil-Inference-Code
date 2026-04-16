import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- MOCKING mlstm_kernels BEFORE IMPORTING THE MODEL ---
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
import struct

def generate_text(model, tokenizer, prompt, max_new_tokens, device):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            if isinstance(logits, tuple): logits = logits[0]
            
            last_logits = logits[:, -1, :] 
            
            temperature = 0.8
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    end_time = time.perf_counter()
    model.train()
    
    out_ids = input_ids[0].tolist()
    output_str = tokenizer.decode(out_ids, True)
    
    delta = end_time - start_time
    tps = max_new_tokens / delta if delta > 0 else 0.0
    return f"{output_str}\n\n[--- Rendimiento Python ---]\n[Velocidad: {tps:.2f} tokens/segundo | Tiempo: {delta:.2f}s]"

def export_bin(model, device, bin_path, tokens, seq_len):
    print(f"\n[Guardando Binario Exportable para Rust en {bin_path} ...]")
    model.eval()
    # Ahora generamos un X / Y fijos iterativos en orden para la prueba de equivalencia
    x_test = []
    for i in range(2):
        start = i * seq_len
        x_test.append(tokens[start : start + seq_len])
    x = torch.tensor(x_test, dtype=torch.long, device=device)

    with torch.no_grad():
        test_logits = model(x)
        if isinstance(test_logits, tuple): test_logits = test_logits[0]

    state_dict = model.state_dict()
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', len(x.shape)))
        for s in x.shape: f.write(struct.pack('<I', s))
        x_data = x.int().cpu().numpy().tobytes()  
        f.write(struct.pack('<I', len(x_data)))
        f.write(x_data)

        f.write(struct.pack('<I', len(test_logits.shape)))
        for s in test_logits.shape: f.write(struct.pack('<I', s))
        y_data = test_logits.float().cpu().numpy().tobytes() 
        f.write(struct.pack('<I', len(y_data)))
        f.write(y_data)

        f.write(struct.pack('<I', len(state_dict)))
        for name, tensor in state_dict.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(tensor.shape)))
            for s in tensor.shape: f.write(struct.pack('<I', s))
            tensor_data = tensor.float().cpu().numpy().tobytes()
            f.write(struct.pack('<I', len(tensor_data)))
            f.write(tensor_data)
    model.train()


def load_from_bin(bin_path, model, device):
    import numpy as np
    print(f"\n[+] Restaurando estado anterior desde el BINARIO: {bin_path}...")
    with open(bin_path, 'rb') as f:
        x_dim = struct.unpack('<I', f.read(4))[0]
        for _ in range(x_dim): f.read(4)
        x_len = struct.unpack('<I', f.read(4))[0]
        f.read(x_len)

        y_dim = struct.unpack('<I', f.read(4))[0]
        for _ in range(y_dim): f.read(4)
        y_len = struct.unpack('<I', f.read(4))[0]
        f.read(y_len)

        state_dict = model.state_dict()
        num_tensors = struct.unpack('<I', f.read(4))[0]
        
        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            shape_len = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(shape_len))
            
            data_len = struct.unpack('<I', f.read(4))[0]
            data_bytes = f.read(data_len)
            
            if name in state_dict:
                numpy_array = np.frombuffer(data_bytes, dtype=np.float32).copy()
                tensor = torch.from_numpy(numpy_array).reshape(shape).to(device)
                state_dict[name].copy_(tensor)
                
    model.load_state_dict(state_dict)

def train_and_export_mini_large():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cpu') 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(current_dir, "..", "..", "rust", "input.txt"))
    tokenizer_path = os.path.abspath(os.path.join(current_dir, "..", "..", "rust", "tokenizer.json"))
    bin_path = os.path.join(current_dir, "test_data.bin")
    checkpoint_path = os.path.join(current_dir, "checkpoint.pt")

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        print(f"Error BPE: {e}")
        return

    vocab_size = tokenizer.get_vocab_size()
    
    config = xLSTMLargeConfig(
        embedding_dim=128,   
        num_heads=2,
        num_blocks=2,
        vocab_size=vocab_size,
        use_bias=True, 
        norm_reduction_force_float32=True,
        weight_mode="single",
        gate_soft_cap=15.0,
        chunk_size=16
    )

    model = xLSTMLarge(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    if os.path.exists(checkpoint_path):
        print(f"\n[+] Encontré un entrenamiento nativo! Restaurando {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    elif os.path.exists(bin_path):
        # Si no hay checkpoint.pt pero SI hay test_data.bin (tu exportacion de ayer)
        load_from_bin(bin_path, model, device)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read() 
    encoded = tokenizer.encode(text)
    tokens = encoded.ids 
    
    seq_len = 32
    batch_size = 4

    mode = "ENTRENAR"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "chat": mode = "CHAT"
        elif arg == "test": mode = "TEST"
        # Si es 'train' o cualquier otra cosa, se queda en 'ENTRENAR'

    print("\n-------------------------------------------")
    print(f"Modo de ejecución: {mode}")
    
    if mode == "CHAT":
        print("\n--- ¡MODO CHAT! (Pulsa Ctrl+C para salir) ---")
        while True:
            try:
                user_msg = input("Usuario: ")
                if not user_msg: continue
                out = generate_text(model, tokenizer, user_msg, max_new_tokens=200, device=device)
                print(f"xlstm_Large: {out}\n")
            except KeyboardInterrupt:
                break
        return

    # Si entramos en modo entrenamiento:
    num_epochs = 2
    if mode == "TEST":
        print(f"\n[!] MODO TEST ACTIVADO: Usando solo 50k tokens y 3 épocas.")
        tokens = tokens[:50000]
        num_epochs = 3
    
    steps_per_epoch = len(tokens) // (seq_len * batch_size)
    print(f"\nEmpezando el entrenamiento ({num_epochs} épocas, {steps_per_epoch} batch/época)...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                start = random.randint(0, len(tokens) - seq_len - 1)
                x_batch.append(tokens[start : start + seq_len])
                y_batch.append(tokens[start + 1 : start + seq_len + 1])
                
            x_train = torch.tensor(x_batch, dtype=torch.long, device=device)
            y_train = torch.tensor(y_batch, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(x_train)
            if isinstance(logits, tuple): logits = logits[0]
            
            loss = loss_fn(logits.view(-1, config.vocab_size), y_train.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Época {epoch+1:2d} | Step {step+1:4d} | Loss: {loss.item():.4f}")
            
            # --- CADA 100 STEPS: PROBAMOS GENERACIÓN INTERACTIVA ---
            if (step + 1) % 100 == 0:
                print("\n[--- TEST DE INFERENCIA EN VIVO ---]")
                out_txt = generate_text(model, tokenizer, "The", max_new_tokens=50, device=device)
                print(f"Generado: {out_txt}")
                print("[----------------------------------]\n")
                
                # Guardamos checkpoint frecuentemente cada 100 steps
                torch.save(model.state_dict(), checkpoint_path)

        avg_loss = epoch_loss / steps_per_epoch
        print(f" >>> FIN ÉPOCA {epoch+1:2d}/{num_epochs} | Loss Promedio: {avg_loss:.4f} <<<")
        torch.save(model.state_dict(), checkpoint_path)
        export_bin(model, device, bin_path, tokens, seq_len)

if __name__ == "__main__":
    train_and_export_mini_large()
