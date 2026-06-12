import torch
import math

def test_cumsum_isolated():
    # Tensor 4D deterministico: (B=1, NH=1, S=4, DH=1)
    fgate_preact = torch.tensor([[[[0.1], [-0.5], [1.2], [-0.1]]]], dtype=torch.float32)
    
    B, NH, S, _ = fgate_preact.shape
    
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)
    print("log_fgates:\n", log_fgates.squeeze())
    
    ltr = torch.tril(torch.ones((S, S), dtype=torch.bool))
    
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=torch.float32),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )
    print("\nlog_fgates_cumsum:\n", log_fgates_cumsum.squeeze())
    
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)
    
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)
    
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))
    
    print("\nFINAL log_fg_matrix (Python):")
    print(log_fg_matrix.squeeze())

if __name__ == "__main__":
    test_cumsum_isolated()
