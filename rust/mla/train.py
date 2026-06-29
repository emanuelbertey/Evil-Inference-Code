import sys, os, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import TransformerLM
from tokenizers import Tokenizer

def main():
    txt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    # Model hyperparams
    d_model = 128
    num_layers = 3
    num_heads = 12
    num_kv_groups = 4
    head_dim = d_model // num_heads  # 64
    # Training hyperparams
    seq_len = 512
    batch_size = 8
    grad_accum = 8
    lr = 3e-4
    num_epochs = 200000
    warmup_steps = 50
    rotary_pct = 0.25

    model = TransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups, head_dim=head_dim, use_swiglu=True, use_x0=True,
        max_seq_len=seq_len, residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, mla_block_size=128).to(device)
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: dim={d_model} layers={num_layers} heads={num_heads} kv={num_kv_groups} seq={seq_len} bs={batch_size} grad_acc={grad_accum} lr={lr} epochs={num_epochs}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    with open(txt_path, "r", encoding="utf-8") as f:
        tokens = tok.encode(f.read()).ids
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    n = len(tokens)
    steps = (n - seq_len) // (batch_size * seq_len)
    print(f"Tokens: {n:,} | Steps/epoch: {steps}")

    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(num_epochs):
        for i in range(0, n - seq_len - 1, batch_size * seq_len):
            if step >= steps * num_epochs:
                break
            if step < warmup_steps:
                lr_curr = lr * (step + 1) / max(warmup_steps, 1)
            else:
                t = (step - warmup_steps) / max(steps * num_epochs - warmup_steps, 1)
                lr_curr = lr * (0.2 + 0.8 * (1.0 + torch.tensor(3.14159 * t).cos().item()) / 2.0)
            for pg in opt.param_groups:
                pg["lr"] = lr_curr

            opt.zero_grad()
            loss_acc = 0.0
            for b in range(batch_size):
                start = i + b * seq_len
                if start + seq_len + 1 >= n:
                    break
                x = tokens[start:start + seq_len].unsqueeze(0)
                y = tokens[start + 1:start + seq_len + 1].unsqueeze(0)
                logits = model.forward_train_partial_rope(x, rotary_pct=rotary_pct)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                (loss / grad_accum).backward()
                loss_acc += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 10 == 0:
                print(f"step {step} loss {loss_acc/batch_size:.4f} lr {lr_curr:.6f} {batch_size*seq_len*10/(time.time()-t0+1e-9):.0f}t/s")
                t0 = time.time()
            step += 1

    print(f"Done! {step} steps")

if __name__ == "__main__":
    main()
