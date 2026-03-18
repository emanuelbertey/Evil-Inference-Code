import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# --- MODELO MinLSTM (Fiel al Jupyter y estable) ---

class MinLSTM(nn.Module):
    def __init__(self, input_size, expansion_factor=2):
        super().__init__()
        self.hidden_size = input_size * expansion_factor
        self.input_size = input_size
        
        # Capas lineales sin bias para paridad con Rust (minGRU style)
        self.W_f = nn.Linear(input_size, self.hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, self.hidden_size, bias=False)
        self.W_h = nn.Linear(input_size, self.hidden_size, bias=False)
        self.fc = nn.Linear(self.hidden_size, input_size, bias=False)
        
        self.init_weights()

    def init_weights(self):
        # Inicialización idéntica a Rust/PyTorch default (Uniform -k, k)
        k = math.sqrt(1.0 / self.input_size)
        nn.init.uniform_(self.W_f.weight, -k, k)
        nn.init.uniform_(self.W_i.weight, -k, k)
        nn.init.uniform_(self.W_h.weight, -k, k)
        
        k_h = math.sqrt(1.0 / self.hidden_size)
        nn.init.uniform_(self.fc.weight, -k_h, k_h)

    def log_cumsum_exp(self, x):
        # Emulamos el centrado (max+min)/2 de Rust para paridad f32
        # Aunque torch.logcumsumexp ya es estable, esto garantiza paridad bit-a-bit
        m_max = torch.max(x, dim=1, keepdim=True)[0]
        m_min = torch.min(x, dim=1, keepdim=True)[0]
        m = (m_max + m_min) / 2.0
        return torch.logcumsumexp(x - m, dim=1) + m

    def parallel_scan_log(self, log_coeffs, log_values):
        # a_star = cumsum(log_f)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        
        # log_h = a_star + logcumsumexp(log_values - a_star)
        log_h = a_star + self.log_cumsum_exp(log_values - a_star)
        return torch.exp(log_h)[:, 1:]

    def log_g(self, x):
        # Estabilización de activación g
        return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))

    def g(self, x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    def forward(self, x, h_0=None):
        b, s, d = x.shape
        if h_0 is None:
            h_0 = torch.zeros(b, 1, self.hidden_size).to(x.device)
            
        # log(sigm(x)) = -softplus(-x)
        log_f_raw = -F.softplus(-self.W_f(x))
        log_i_raw = -F.softplus(-self.W_i(x))
        
        # MinGRU Stability Trick: Forzamos diff a ser negativo
        # Esto limita log_f a [-log(2), 0]
        diff_raw = log_i_raw - log_f_raw
        diff = -F.softplus(diff_raw)
        
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        
        log_h_0 = self.log_g(h_0)
        log_tilde_h = self.log_g(self.W_h(x))
        
        # Valores a acumular
        log_values = torch.cat([log_h_0, log_i + log_tilde_h], dim=1)
        h = self.parallel_scan_log(log_f, log_values)
        
        return self.fc(h), h[:, -1:]

    def sequential_forward(self, x_t, h_prev):
        # Para paridad con el modo paralelo, debemos usar la misma lógica de gating
        # log_f = -softplus(diff) -> f = sigmoid(-diff)
        log_f_raw = -F.softplus(-self.W_f(x_t))
        log_i_raw = -F.softplus(-self.W_i(x_t))
        
        diff = -F.softplus(log_i_raw - log_f_raw)
        
        # Correcto: f = exp(-softplus(diff)) = sigmoid(-diff)
        f_prime_t = torch.sigmoid(-diff)
        i_prime_t = torch.sigmoid(diff)
        
        tilde_h_t = self.g(self.W_h(x_t))
        h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t
        
        return self.fc(h_t), h_t

# --- TESTS ---

def test_gradients():
    print("--- TEST 1: Gradient Flow (S=250) [PYTHON MinLstm] ---")
    device = "cpu"
    seq_len = 250
    b, d = 1, 16
    
    model = MinLSTM(d).to(device)
    x = torch.randn(b, seq_len, d, requires_grad=True)
    
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    
    grad = x.grad
    checkpoints = list(range(0, seq_len, 50)) + [seq_len-1]
    
    all_valid = True
    for t in checkpoints:
        g_t = grad[:, t, :].abs().mean().item()
        print(f"  t={t:3}  |grad|={g_t:.10f}")
        if math.isnan(g_t) or g_t == 0.0:
            all_valid = False
            
    if all_valid:
        print("SUCCESS: Todos los gradientes son válidos")
    else:
        print("FAILURE: Hay gradientes NaN o zero")

def test_sequential_equivalence():
    print("\n--- TEST 2: Parallel vs Sequential Equivalence (S=8) ---")
    b, d, s = 2, 16, 8
    model = MinLSTM(d)
    x = torch.randn(b, s, d)
    h0 = torch.zeros(b, 1, d*2)
    
    # Modo Paralelo
    out_par, _ = model(x, h0)
    
    # Modo Secuencial
    h_prev = model.g(h0) # Inicialización consistente
    seq_outs = []
    for t in range(s):
        x_t = x[:, t:t+1, :]
        out_t, h_next = model.sequential_forward(x_t, h_prev)
        seq_outs.append(out_t)
        h_prev = h_next
    out_seq = torch.cat(seq_outs, dim=1)
    
    diff = (out_par - out_seq).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max  |parallel - sequential| = {max_diff:.10f}")
    print(f"  Mean |parallel - sequential| = {mean_diff:.10f}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Parallel ~= Sequential")
    else:
        print("WARNING: Diferencia significativa")

def test_copy_task():
    print("\n--- TEST 3: LM Copy Task (S=32, B=4) [PYTHON MinLstm] ---")
    vocab_size = 10
    d = 16
    seq_len = 32
    batch_size = 4
    
    # Generar patrón repetitivo
    pattern = torch.arange(vocab_size).repeat(200)
    
    model = nn.ModuleDict({
        'embed': nn.Embedding(vocab_size, d),
        'minlstm': MinLSTM(d),
        'to_logits': nn.Linear(d, vocab_size, bias=False)
    })
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    num_steps = 200
    
    for step in range(1, num_steps + 1):
        # Preparar batch (simplificado como en Rust)
        # Tomamos chunks del patrón
        x_indices = []
        y_indices = []
        for b in range(batch_size):
            start = (step + b * 10) % (len(pattern) - seq_len - 1)
            x_indices.append(pattern[start:start+seq_len])
            y_indices.append(pattern[start+1:start+seq_len+1])
            
        x_batch = torch.stack(x_indices)
        y_batch = torch.stack(y_indices).view(-1)
        
        # Forward
        emb = model['embed'](x_batch)
        out, _ = model['minlstm'](emb)
        logits = model['to_logits'](out).view(-1, vocab_size)
        
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if step == 1 or step % 50 == 0:
            print(f"  Step {step:3}  loss={loss.item():.4f}")
            
    if losses[-1] < losses[0] * 0.5:
        print("SUCCESS: Copy Task converge!")
    else:
        print(f"FAILURE: Copy Task no converge ({losses[0]:.4f} -> {losses[-1]:.4f})")

def test_h0_persistence():
    print("\n--- TEST 4: h_0 Persistence Effect ---")
    b, d, s = 1, 16, 16
    model = MinLSTM(d)
    x = torch.randn(b, s, d)
    
    # h0 zeros
    h0_zeros = torch.zeros(b, 1, d*2)
    out_zeros, states_zeros = model(x, h0_zeros)
    
    # h0 persistente
    h0_prev = states_zeros
    out_with_state, _ = model(x, h0_prev)
    
    diff = (out_zeros - out_with_state).abs().max().item()
    print(f"  Max |out(h0=0) - out(h0=prev)| = {diff:.10f}")
    
    if diff > 1e-6:
        print("SUCCESS: h_0 persistente cambia la salida")
    else:
        print("WARNING: h_0 no tiene efecto")

if __name__ == "__main__":
    print("============================================")
    print("  TEST SUITE: MinLSTM Python Implementation")
    print("============================================\n")
    
    test_gradients()
    test_sequential_equivalence()
    test_copy_task()
    test_h0_persistence()
    
    print("\n============================================")
    print("  ALL TESTS COMPLETE")
    print("============================================")
