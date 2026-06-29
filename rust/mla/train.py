import sys, os, time, math, torch
import torch.nn.functional as F
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))
from mla.model import TransformerLM
from dataset import download_wikipedia_50mb, StreamingDataset
from huggingface import HFManager, PeriodicPusher
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=50):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=1.0, top_k=50, top_p=0.95,
                         use_partial_rope=True, rotary_pct=0.25)
    model.train()
    return tokenizer.decode(out[0].tolist())

def train_tokenizer_from_wiki(vocab_size, output_path):
    wiki = download_wikipedia_50mb()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["eos_token"])
    with open(wiki, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    return output_path

def get_lr(step, total, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (0.2 + 0.8 * (1.0 + math.cos(math.pi * t)) / 2.0)

# ─── Config ──────────────────────────────────────────────────────────────────
d_model = 128
num_layers = 3
num_heads = 12
num_kv_groups = 4
head_dim = d_model // num_heads
seq_len = 256
batch_size = 8
grad_accum = 8
lr = 3e-4
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 16000
rotary_pct = 0.25
tok_path = os.path.join(_DIR, "tokenizer.json")

def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1].endswith(".txt")
    txt_path = sys.argv[1] if test_mode else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── HF ──────────────────────────────────────────────────────────────────
    repo_id = "ScortexIA/laurelia"
    revision = "gens0mla"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=10)
        prec = input("Precision (n=f32, f=f16): ").strip().lower()
        dtype = torch.float16 if prec == "f" else torch.float32
    else:
        dtype = torch.float32
    print(f"Using {dtype}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = None
    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    elif hf and hf.tokenizer_exists():
        try:
            local_tok = hf.download_tokenizer(tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
        except:
            pass
    if tokenizer is None:
        if hf:
            train_tokenizer_from_wiki(bpe_vocab, tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
            hf.upload_tokenizer(tok_path, os.path.join(_DIR, "tokenizer_config.json"))
        else:
            sys.exit("No tokenizer found")
    print(f"Vocab: {tokenizer.vocab_size}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups, head_dim=head_dim,
        use_swiglu=True, use_x0=True, max_seq_len=seq_len,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, mla_block_size=128,
    ).to(device).to(dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # ── Checkpoint ─────────────────────────────────────────────────────────
    step = 0
    epoch = 0
    ckpt_block = 0
    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    safe_path = os.path.join(_DIR, "model_test.safetensors")

    if not test_mode:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            print(f"Loaded checkpoint: step {step} epoch {epoch} block {ckpt_block}")
        elif hf and hf.download_checkpoint(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            print(f"Loaded HF checkpoint: step {step} epoch {epoch} block {ckpt_block}")

    # ── Data ────────────────────────────────────────────────────────────────
    if test_mode:
        with open(txt_path, "r", encoding="utf-8") as f:
            all_tokens = tokenizer.encode(f.read())
        tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
        n = len(tokens)
        epochs_do = 10
        total_steps = ((n - seq_len - 1) // (batch_size * seq_len)) * epochs_do
        stream_next = lambda: None
        stream_block = 0
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = StreamingDataset(block_mb=3.0, block_idx=block_idx)
        sd.load_tokens(tokenizer)
        all_tokens = sd.get_tokens()
        tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
        n = len(tokens)
        tokens_per_epoch = (n - seq_len - 1) // seq_len
        epochs_do = num_epochs
        total_steps = (tokens_per_epoch // batch_size) * num_epochs
        stream_next = lambda: None  # simplified for now

    # ── Stats ──────────────────────────────────────────────────────────────
    emb_p = model.embedding.weight.numel()
    layer_p = sum(p.numel() for l in model.transformer.layers for p in l.parameters())
    norm_p = model.transformer.final_norm.weight.numel()
    x0_p = model.x0_lambdas.numel() if model.x0_lambdas is not None else 0
    total_p = emb_p + layer_p + norm_p + x0_p
    print(f"Params: emb={emb_p:,} + {num_layers}capas={layer_p:,} + norm={norm_p} + x0={x0_p} = {total_p:,}")
    print(f"dim={d_model} lay={num_layers} heads={num_heads} kv={num_kv_groups} seq={seq_len} bs={batch_size} ga={grad_accum} lr={lr}")
    print(f"Tokens: {n:,} | Steps total: {total_steps}")

    # ── Train loop ──────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    last_rpt = t0

    n_seq = (n - seq_len - 1) // seq_len
    for epoch in range(epochs_do):
        micro = 0
        for batch_start in range(0, n_seq, batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + batch_size, n_seq)
            x = torch.stack([tokens[i*seq_len:i*seq_len+seq_len] for i in range(batch_start, batch_end)])
            y = torch.stack([tokens[i*seq_len+1:i*seq_len+seq_len+1] for i in range(batch_start, batch_end)])

            if micro == 0:
                lr_curr = get_lr(step, total_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr_curr
                opt.zero_grad()

            logits = model.forward_train_partial_rope(x, rotary_pct=rotary_pct)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            (loss / grad_accum).backward()
            micro += 1

            if micro >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                micro = 0

                if step % 10 == 0:
                    now = time.time()
                    tps = batch_size * grad_accum * seq_len / max(now - last_rpt, 0.001)
                    print(f"e{epoch} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s")
                    last_rpt = now

                if not test_mode and step % 50 == 0:
                    sample = generate_sample(model, tokenizer, device)
                    print(f"  >>> {sample}")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    ckpt = {"step": step, "epoch": epoch, "block": ckpt_block, "model": model.state_dict()}
                    torch.save(ckpt, ckpt_path)
                    model.state_dict_to_safetensors(safe_path)
                    pusher.maybe_push(ckpt_path, safe_path, tok_path, step)

        if micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        print(f"── Epoch {epoch} done: {step} steps ──")
        if test_mode and epoch >= epochs_do - 1:
            break

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": ckpt_block, "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, safe_path, tok_path, step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
