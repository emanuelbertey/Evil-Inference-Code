import torch
import torch.nn as nn
import torch.nn.functional as F

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

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

    def forward(self, x, h0):
        seq_len = x.shape[1]
        update_gate = self.linear_z(x)
        hidden_state = self.linear_h(x)
        k = -F.softplus(update_gate)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h0)
        log_tilde_h = log_g(hidden_state)        
        log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        output = parallel_scan_log(log_coeffs, log_values)
        output = output[:, -x.shape[1]:]  # this slices [:, -T:]
        return output

    def sequential_mode(self, x_t, h_prev):
        # Python seq mode
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = self.linear_h(x_t)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

    def sequential_mode_rust(self, x_t, h_prev):
        # Rust seq mode
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = self.linear_h(x_t)
        h_t = (1 - z_t) * h_prev + z_t * g(h_tilde)
        return h_t
        
    def sequential_mode_exact(self, x_t, h_prev):
        # mathematical exact corresponding to parallel forward
        update_gate = self.linear_z(x_t)
        k = -F.softplus(update_gate)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        # exponentiate coefficients
        coeff = torch.exp(log_coeffs)
        z = torch.exp(log_z)
        h_tilde = self.linear_h(x_t)
        # recursive update: h_t = coeff * h_{t-1} + z * g(h_tilde)
        h_t = coeff * h_prev + z * g(h_tilde)
        return h_t

torch.manual_seed(42)
B, T, D = 2, 5, 4
x = torch.randn(B, T, D)
h0 = torch.zeros(B, 1, D * 2)

model = minGRU(D, 2)
out_par = model(x, h0)

# Run Python sequential
h_seq_python = h0[:, 0, :]
h_seq_python_outs = []
for t in range(T):
    h_seq_python = model.sequential_mode(x[:, t, :], h_seq_python)
    h_seq_python_outs.append(h_seq_python)

# Run Rust sequential
h_seq_rust = h0[:, 0, :]
h_seq_rust_outs = []
for t in range(T):
    h_seq_rust = model.sequential_mode_rust(x[:, t, :], h_seq_rust)
    h_seq_rust_outs.append(h_seq_rust)

# Run Exact sequential
h_seq_exact = torch.exp(log_g(h0[:, 0, :]))
h_seq_exact_outs = []
for t in range(T):
    h_seq_exact = model.sequential_mode_exact(x[:, t, :], h_seq_exact)
    h_seq_exact_outs.append(h_seq_exact)

out_seq_python = torch.stack(h_seq_python_outs, dim=1)
out_seq_rust = torch.stack(h_seq_rust_outs, dim=1)
out_seq_exact = torch.stack(h_seq_exact_outs, dim=1)

print("Parallel vs Python Seq Diff:", torch.max(torch.abs(out_par - out_seq_python)).item())
print("Parallel vs Rust Seq Diff:", torch.max(torch.abs(out_par - out_seq_rust)).item())
print("Parallel vs Exact Seq Diff:", torch.max(torch.abs(out_par - out_seq_exact)).item())
