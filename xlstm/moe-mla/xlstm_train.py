"""xLSTM MoE Training — mLSTM + MoE (uses transformer tokenizer, same revision)."""
import sys, os, time, math, torch, json
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))

from xlstm_model import xLSTMMoEModel
from dataset import download_wikipedia_50mb, StreamingDataset
from huggingface import HFManager, PeriodicPusher
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from plot import PlotManager

class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=100):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=1.0, top_k=50)
    model.train()
    return tokenizer.decode(out[0].tolist())

def get_lr(step, total, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (0.2 + 0.8 * (1.0 + math.cos(math.pi * t)) / 2.0)

# ─── Config ──────────────────────────────────────────────────────────────
d_model = 512
num_layers = 8
num_heads = 4
seq_len = 512
batch_size = 4
grad_accum = 4
lr = 3e-4
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 32000
tok_path = os.path.join(_DIR, "tokenizer.json")  # same tokenizer as transformer
train_json = os.path.join(_DIR, "xlstm_train_state.json")
ckpt_path = os.path.join(_DIR, "xlstm_checkpoint.pt")
plot_dir = _DIR
test_mode = False
n_experts = 4
top_k = 1
n_shared = 1
expert_dim = d_model * 4
capacity_factor = 1.0
z_loss_gamma = 0.001
bias_decay = 0.1
noise_std = 0.01

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_id = "ScortexIA/laurelia"
    revision = "moe-mla"  # same branch as transformer
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=200,
                     json_prefix="xlstm_")

    prec = input("Precision (n=f32, f=f16, b=bf16): ").strip().lower()
    dtype = {"b": torch.bfloat16, "f": torch.float16}.get(prec, torch.float32)
    print(f"  Compute: {dtype}")

    # Tokenizer (same as transformer, no xlstm prefix)
    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    elif hf and hf.tokenizer_exists(filename="tokenizer.json"):
        local_tok = hf.download_tokenizer(tok_path, remote_filename="tokenizer.json")
        tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
    else:
        wiki = download_wikipedia_50mb()
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=bpe_vocab, special_tokens=["eos_token"])
        with open(wiki, "r", encoding="utf-8") as f:
            tok.train_from_iterator([f.read()], trainer=trainer)
        tok.save(tok_path)
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    print(f"Vocab: {tokenizer.vocab_size}")

    # Model
    model = xLSTMMoEModel(
        vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, use_moe=True, n_experts=n_experts, top_k=top_k,
        n_shared=n_shared, expert_dim=expert_dim, capacity_factor=capacity_factor,
        z_loss_gamma=z_loss_gamma, bias_decay=bias_decay, noise_std=noise_std,
        max_seq_len=seq_len,
    ).to(device).to(dtype=dtype)

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Load checkpoint (local → HF → fresh)
    step = 0
    epoch = 0
    loaded = False
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint: step {step} epoch {epoch}")
        del ckpt
        torch.cuda.empty_cache()
        loaded = True
    if not loaded and hf:
        if hf.download_checkpoint(ckpt_path, filename="xlstm_checkpoint.pt"):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            print(f"Loaded HF checkpoint: step {step} epoch {epoch}")
            del ckpt
            torch.cuda.empty_cache()
    if not loaded:
        print("No checkpoint found, starting fresh")

    # Data
    sd = StreamingDataset(block_idx=step // 2000)
    tokens = sd.get_tokens() if step > 0 else None

    total_steps = 100000
    print(f"Tokens per step: {batch_size * seq_len}")
    print(f"Total steps: {total_steps}")

    # Train
    model.train()
    t0 = time.time()

    while True:
        if tokens is None:
            sd.load_tokens(tokenizer)
            tokens = sd.get_tokens()
        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            sd.next_block()
            tokens = sd.get_tokens()
            continue

        for batch_start in range(0, n_seq, batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x); y_list.append(y)
            x = torch.cat(x_list, dim=0); y = torch.cat(y_list, dim=0)

            lr_curr = get_lr(step, total_steps, warmup_steps, lr)
            for param_group in opt.param_groups:
                param_group["lr"] = lr_curr

            logits, aux_loss = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            (loss + aux_loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if step % 10 == 0:
                dt = time.time() - t0
                tps = batch_size * seq_len / max(dt, 1e-6)
                t0 = time.time()
                e = epoch + batch_start // (n_seq // max(num_epochs, 1) + 1)
                balance_strs = []
                for blk in model.blocks:
                    if hasattr(blk.moe, "balance_str"):
                        s = blk.moe.balance_str()
                        if s:
                            balance_strs.append(f"L{blk._layer_idx}:{s}")
                bal = " | ".join(balance_strs[:5])
                print(f"e{e} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s")
                if bal:
                    print(f"  MoE balance: {bal}")
                pm.log(step, loss.item(), lr_curr, tps, aux_loss.item(),
                       grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            if step > 0 and step % 50 == 0:
                t_gen = time.time()
                sample = generate_sample(model, tokenizer, device)
                gen_tps = 100 / (time.time() - t_gen)
                print(f"  >>> {sample}  [{gen_tps:.0f} tok/s]")

            if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                state = model.state_dict()
                ckpt = {"step": step, "epoch": epoch, "model": state}
                torch.save(ckpt, ckpt_path)
                pusher.maybe_push(ckpt_path, None, None, step)

            step += 1

        sd.next_block()
        tokens = sd.get_tokens()
        epoch += 1

if __name__ == "__main__":
    main()
