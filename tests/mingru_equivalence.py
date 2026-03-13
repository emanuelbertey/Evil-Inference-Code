import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- EXACTAMENTE del Jupyter ---
def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

def parallel_scan_log(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

class minGRU(nn.Module):
    def __init__(self, input_features, expansion_factor=2):
        super().__init__()
        self.linear_z = nn.Linear(input_features, input_features * expansion_factor, bias=False)
        self.linear_h = nn.Linear(input_features, input_features * expansion_factor, bias=False)
        self.output_projection = nn.Linear(input_features * expansion_factor, input_features, bias=False)

    def forward(self, x, h_0):
        update_gate = self.linear_z(x)
        hidden_state = self.linear_h(x)
        k = -F.softplus(update_gate)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(hidden_state)
        log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        output = parallel_scan_log(log_coeffs, log_values)
        output = output[:, -x.shape[1]:]
        latest_hidden = output[:, -1:]
        output = self.output_projection(output)
        return output, latest_hidden

def test_gradients():
    torch.manual_seed(42)
    seq_len    = 250
    batch_size = 1
    input_dim  = 16
    expansion  = 2

    model = minGRU(input_dim, expansion)
    x     = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    h0    = torch.zeros(batch_size, 1, input_dim * expansion)

    output, _ = model(x, h0)
    output.sum().backward()

    grad = x.grad
    print(f"--- TEST 1: Gradient Flow (S={seq_len}, loss=sum_all) ---")
    for t in list(range(0, seq_len, 50)) + [seq_len - 1]:
        g_t = grad[0, t, :].abs().mean().item()
        print(f"  t={t:3d}  |grad|={g_t:.10f}")

    g_first = grad[0, 0, :].abs().mean().item()
    g_last  = grad[0, seq_len-1, :].abs().mean().item()
    ratio   = g_first / (g_last + 1e-30)
    print(f"  ratio t=0/t={seq_len-1}: {ratio:.4f}")
    if g_first > 1e-10:
        print("SUCCESS: Gradiente llega al inicio!\n")
    else:
        print("FAILURE: Desvanecimiento detectado.\n")

def get_batches(data, batch_size, seq_length):
    total_length = data.size(0)
    num_batches = (total_length - 1) // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = data.view(batch_size, -1)
    for i in range(0, data.size(1) - seq_length, seq_length):
        x = data[:, i:i + seq_length]
        y = data[:, i + 1:i + seq_length + 1]
        yield x, y

def test_copy_task():
    torch.manual_seed(7)
    pattern = list(range(10)) * 200
    data    = torch.tensor(pattern)
    vocab_size   = 10
    input_features = 16
    expansion    = 2
    seq_len      = 32
    batch_size   = 4

    embed  = nn.Embedding(vocab_size, input_features)
    model  = minGRU(input_features, expansion)
    to_logits = nn.Linear(input_features, vocab_size, bias=False)
    h_0    = torch.zeros(batch_size, 1, input_features * expansion)

    optimizer = torch.optim.Adam(
        list(embed.parameters()) + list(model.parameters()) + list(to_logits.parameters()),
        lr=2e-3
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"--- TEST 2: LM Copy Task (S={seq_len}, B={batch_size}, vocab={vocab_size}) ---")
    losses = []
    for step in range(1, 201):
        step_loss = 0
        n = 0
        for x_ids, y_ids in get_batches(data, batch_size, seq_len):
            x_emb  = embed(x_ids)
            out, _ = model(x_emb, h_0)
            logits = to_logits(out)
            loss   = loss_fn(logits.reshape(-1, vocab_size), y_ids.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
            n += 1
        avg = step_loss / n
        if step == 1 or step % 50 == 0:
            print(f"  Step {step:3d}  loss={avg:.4f}")
        losses.append(avg)

    if losses[-1] < losses[0] * 0.5:
        print("SUCCESS: Copy Task converge!\n")
    else:
        print(f"FAILURE: Copy Task no converge ({losses[0]:.4f} -> {losses[-1]:.4f}).\n")

if __name__ == "__main__":
    test_gradients()
    test_copy_task()
