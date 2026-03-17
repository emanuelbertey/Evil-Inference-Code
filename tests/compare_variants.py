import torch
import torch.nn.functional as F

def log_g(x):
    mask = (x >= 0).float()
    pos = torch.log(F.relu(x) + 0.5)
    neg = -F.softplus(-x)
    return mask * pos + (1 - mask) * neg

def test_full_parity():
    device = torch.device("cpu")
    print("--- PYTHON: FULL PARITY TEST ---")

    # 1. GRADIENTES PRIMITIVAS
    x_vals = torch.tensor([-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0], device=device, requires_grad=True)
    
    # Softplus
    res_sp = F.softplus(x_vals)
    res_sp.sum().backward()
    grad_sp = x_vals.grad.clone()
    x_vals.grad.zero_()
    print("\n[1] Softplus:")
    print(f"Values: {res_sp}")
    print(f"Grads:  {grad_sp}")

    # Log-g
    res_lg = log_g(x_vals)
    res_lg.sum().backward()
    grad_lg = x_vals.grad.clone()
    print("\n[2] Log-g:")
    print(f"Values: {res_lg}")
    print(f"Grads:  {grad_lg}")

    # 2. LOGCUMSUMEXP 1D
    seq_1d = torch.linspace(-50, 50, 250, device=device, requires_grad=True)
    res_1d = seq_1d.exp().cumsum(0).log()
    res_1d.sum().backward()
    print("\n[3] Logcumsumexp 1D:")
    print(f"Val Max:  {res_1d.max().item():.6f}")
    print(f"Grad Max: {seq_1d.grad.max().item():.6f}")

    # 3. LOGCUMSUMEXP 3D
    seq_3d = torch.linspace(-50, 50, 250, device=device).repeat(8).view(2, 250, 4).requires_grad_(True)
    res_3d = seq_3d.exp().cumsum(1).log()
    res_3d.sum().backward()
    print("\n[4] Logcumsumexp 3D:")
    print(f"Val Max:  {res_3d.max().item():.6f}")
    print(f"Grad Max: {seq_3d.grad.max().item():.6f}")

if __name__ == "__main__":
    test_full_parity()
