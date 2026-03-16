import torch
import torch.nn.functional as F

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

def stable_logcumsumexp(x):
    result = []
    current = x[0]
    result.append(current)
    for i in range(1, len(x)):
        current = torch.logaddexp(current, x[i])
        result.append(current)
    return torch.stack(result)

def test_all():
    # 1. Primitivas
    x = torch.tensor([-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0])
    print("\n--- PYTHON: Primitivas ---")
    print(f"Softplus: {F.softplus(x).tolist()}")
    print(f"Log-g:    {log_g(x).tolist()}")

    # 2. Gradientes (S=250)
    seq = torch.linspace(-50, 50, 250, requires_grad=True)
    
    # Nativo
    res_native = torch.logcumsumexp(seq, dim=0)
    res_native.sum().backward()
    grad_native = seq.grad.clone()
    seq.grad.zero_()
    
    # Estable (Ratio 1.95)
    res_stable = stable_logcumsumexp(seq)
    res_stable.sum().backward()
    grad_stable = seq.grad.clone()

    print("\n--- PYTHON: logcumsumexp (S=250) ---")
    print(f"NATIVO - Grad Max: {grad_native.max().item():.7f}")
    print(f"NATIVO - Grad Min: {grad_native.min().item():.7f}")
    print(f"STABLE - Grad Max: {grad_stable.max().item():.7f}")
    print(f"STABLE - Grad Min: {grad_stable.min().item():.7f}")

if __name__ == "__main__":
    test_all()
