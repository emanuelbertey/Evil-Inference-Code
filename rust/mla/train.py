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
d_model = 768
num_layers = 24
num_heads = 12
num_kv_groups = 4
head_dim = d_model // num_heads
seq_len = 512
batch_size = 8
grad_accum = 8
lr = 3e-4
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 32000
rotary_pct = 0.25
use_mla = True
mla_d_c = 32      # 16=ahorroMaximo 32=balance(default) 64=calidadMax. Comprime K y V.
mla_d_c1 = None      # Comprime Q. NO afecta cache, solo parametros. Dejalo igual que d_c.
mla_d_rotate = None  # RoPE. Info de posicion del token. 16 default. No tocar.
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
        pusher = PeriodicPusher(hf, interval_minutes=15)

    # ── Precision ──────────────────────────────────────────────────────────
    amp = False
    if test_mode:
        dtype = torch.float32
    else:
        prec = input("Precision (n=f32, f=f16, b=bf16, a=amp(f16+master-f32)): ").strip().lower()
        if prec == "b":
            dtype = torch.bfloat16
        elif prec == "f":
            dtype = torch.float16
        elif prec == "a":
            dtype = torch.float16
            amp = True
        else:
            dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=amp)
    master = "f32 (master)" if amp else str(dtype)
    print(f"  Compute: {dtype}  |  Weights: {master}  |  AMP: {amp}  |  Scaler: {scaler.get_scale() if amp else 'off'}")

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
        mla_d_c=mla_d_c, mla_d_c1=mla_d_c1, mla_d_rotate=mla_d_rotate,
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
        n = len(all_tokens)
        epochs_do = 10
        total_steps = ((n - seq_len - 1) // (batch_size * seq_len)) * epochs_do
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = StreamingDataset(block_mb=3.0, block_idx=block_idx)
        sd.load_tokens(tokenizer)
        n = len(sd.get_tokens())
        tokens_per_epoch = (n - seq_len - 1) // seq_len
        total_steps = (tokens_per_epoch // batch_size) * num_epochs
        epochs_do = num_epochs

    # ── Stats ──────────────────────────────────────────────────────────────
    emb_p = model.embedding.weight.numel()
    layer_p = sum(p.numel() for l in model.transformer.layers for p in l.parameters())
    norm_p = model.transformer.final_norm.weight.numel()
    x0_p = model.x0_lambdas.numel() if model.x0_lambdas is not None else 0
    total_p = emb_p + layer_p + norm_p + x0_p
    print(f"Params: emb={emb_p:,} + {num_layers}capas={layer_p:,} + norm={norm_p} + x0={x0_p} = {total_p:,}")
    print(f"dim={d_model} lay={num_layers} heads={num_heads} kv={num_kv_groups} seq={seq_len} bs={batch_size} ga={grad_accum} lr={lr}")
    gqa_cpt = 2 * num_kv_groups * head_dim
    if use_mla:
        a = model.transformer.layers[0].attention.qkv
        d_c_real = a.d_c; d_rot_real = a.d_rotate; d_c1_real = a.d_c1
        cpt = d_c_real + d_rot_real
        pct = 100 * (1 - cpt / gqa_cpt)
        print(f"MLA: d_c={d_c_real} d_c1={d_c1_real} d_rot={d_rot_real} | cache: {gqa_cpt}→{cpt}B/tok ({pct:.0f}%)")
    else:
        cpt = gqa_cpt
    print(f"Tokens: {n:,} | Steps total: {total_steps}")

    # ── Train loop ──────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0

    epoch = 0
    while True:
        if test_mode:
            tokens = all_tokens
        else:
            sd.load_tokens(tokenizer)
            tokens = sd.get_tokens()

        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            print(f"Block too small ({len(tokens)} tokens), skipping epoch {epoch}")
            epoch += 1
            if epoch >= epochs_do:
                break
            continue

        micro = 0
        for batch_start in range(0, n_seq, batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x)
                y_list.append(y)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

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
                    tok = (step - last_rpt_step) * batch_size * grad_accum * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    print(f"e{epoch} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s")
                    last_rpt_time = now
                    last_rpt_step = step

                if not test_mode and step % 50 == 0:
                    sample = generate_sample(model, tokenizer, device)
                    print(f"  >>> {sample}")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx if not test_mode else 0, "model": model.state_dict()}
                    torch.save(ckpt, ckpt_path)
                    model.state_dict_to_safetensors(safe_path)
                    pusher.maybe_push(ckpt_path, safe_path, tok_path, step)

        if micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        epoch += 1
        print(f"── Epoch {epoch} done: {step} steps ──")
        if epoch >= epochs_do:
            break

        if not test_mode:
            tokens = None
            sd.next_block()
            total_tokens = len(sd.get_tokens())
            n_seq = (total_tokens - seq_len - 1) // seq_len
            total_steps = (n_seq // batch_size) * (epochs_do - epoch)

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx, "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, safe_path, tok_path, step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
