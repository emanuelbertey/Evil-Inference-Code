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
        output = output[:, -x.shape[1]:]
        latest_hidden = output[:, -1:]
        output = self.output_projection(output)
        return output, latest_hidden

    def sequential_mode(self, x_t, h_prev):
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = self.linear_h(x_t)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return self.output_projection(h_t), h_t
        
    def sequential_mode_fixed(self, x_t, h_prev):
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = self.linear_h(x_t)
        h_t = (1 - z_t) * h_prev + z_t * g(h_tilde)
        return self.output_projection(h_t), h_t

# Test equivalence
torch.manual_seed(42)
B, T, D = 2, 5, 4
x = torch.randn(B, T, D)
h0 = torch.zeros(B, 1, D * 2)

model = minGRU(D, 2)
out_par, h_par = model(x, h0)

h_seq = h0[:, 0, :]
for t in range(T):
    _, h_seq = model.sequential_mode(x[:, t:t+1, :], h_seq.unsqueeze(1))
h_seq_fixed = h0[:, 0, :]
for t in range(T):
    _, h_seq_fixed = model.sequential_mode_fixed(x[:, t:t+1, :], h_seq_fixed.unsqueeze(1))

print("Parallel vs Python Sequential max diff:", torch.max(torch.abs(h_par[:, -1, :] - h_seq.squeeze(1))).item())
print("Parallel vs Rust Sequential (fixed) max diff:", torch.max(torch.abs(h_par[:, -1, :] - h_seq_fixed.squeeze(1))).item())
