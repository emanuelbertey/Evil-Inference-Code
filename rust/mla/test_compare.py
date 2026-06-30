import sys, os, math, torch, torch.nn as nn, torch.nn.functional as F
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "LLM_D3-main"))

from model import TransformerLM
from LLM_2 import GPT, GPTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

txt = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(_DIR, "..", "input.txt"))
with open(txt, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
ctoi = {c:i for i,c in enumerate(chars)}

def encode(s):
    return [ctoi[c] for c in s]

tokens_1d = encode(text)
seq_len = 32
batch_size = 4

def make_batch(tokens, seq_len, batch_size):
    n = len(tokens)
    xs, ys = [], []
    for i in range(batch_size):
        offset = (i * seq_len) % (n - seq_len - 1)
        xs.append(tokens[offset:offset+seq_len])
        ys.append(tokens[offset+1:offset+seq_len+1])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y

x, y = make_batch(tokens_1d, seq_len, batch_size)

d_model = 256
n_layer = 3
n_head = 4
head_dim = d_model // n_head
d_c = 32; d_c1 = 32; d_rotate = 16
num_kv_groups = 4

model_ours = TransformerLM(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
).to(device)

cfg = GPTConfig(
    block_size=seq_len, vocab_size=vocab_size, n_layer=n_layer,
    n_head=n_head, n_embd=d_model, dropout=0.0, ffn_dim=d_model * 4,
    bias=False, d_c=d_c, d_c1=d_c1, d_rotate=d_rotate, theta=10000.0,
    n_exp=1, top_k=1, expert_dim=d_model * 2, stride=3,
    use_aux_loss=False, use_router_z_loss=False, use_noisy_top_k=False,
    aux_loss_weight=0.0, router_z_loss_weight=0.0,
    train_capacity=1.0, eval_capacity=1.0, min_capacity=1,
    use_switch_tfm_init=False, router_use_full_prec=False,
)
model_llm = GPT(cfg).to(device)

ours_params = sum(p.numel() for p in model_ours.parameters())
llm_params = sum(p.numel() for p in model_llm.parameters())
print(f"Params nuestro: {ours_params:,}  |  LLM_D3: {llm_params:,}  |  diff: {abs(ours_params-llm_params):,}")

opt_ours = torch.optim.AdamW(model_ours.parameters(), lr=3e-4, weight_decay=0.01)
opt_llm = torch.optim.AdamW(model_llm.parameters(), lr=3e-4, weight_decay=0.01)

print(f"{'step':>5}  {'nuestro':>10}  {'llm_d3':>10}  {'diff':>10}")
for step in range(10):
    opt_ours.zero_grad()
    l = F.cross_entropy(model_ours(x).reshape(-1, vocab_size), y.reshape(-1))
    l.backward(); torch.nn.utils.clip_grad_norm_(model_ours.parameters(), 1.0); opt_ours.step()
    l_ours = l.item()

    opt_llm.zero_grad()
    out = model_llm(x, labels=y)
    l = F.cross_entropy(out.logits[:, :-1].reshape(-1, vocab_size), y[:, 1:].reshape(-1))
    l.backward(); torch.nn.utils.clip_grad_norm_(model_llm.parameters(), 1.0); opt_llm.step()
    l_llm = l.item()

    print(f"{step+1:5d}  {l_ours:10.6f}  {l_llm:10.6f}  {l_ours-l_llm:10.6f}")
