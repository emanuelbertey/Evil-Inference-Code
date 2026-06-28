"""Train transformer with Wikipedia ES streaming, HF tokenizer & checkpoint push."""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.model import TransformerLM
from dataset import download_wikipedia_50mb, StreamingDataset
from huggingface import HFManager, PeriodicPusher


class BPEWrapper:
    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=50):
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=1.0, top_k=50, top_p=0.95,
                         use_partial_rope=True, rotary_pct=0.25)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


def create_batch(tokens, start_idx, batch_size, seq_len, stride):
    x_indices = []
    y_indices = []
    for i in range(batch_size):
        current_start = start_idx + i * stride
        for j in range(seq_len):
            x_indices.append(tokens[current_start + j])
            y_indices.append(tokens[current_start + j + 1])
    return (torch.tensor(x_indices, dtype=torch.long).view(batch_size, seq_len),
            torch.tensor(y_indices, dtype=torch.long).view(batch_size, seq_len))


def train_tokenizer_from_wiki(vocab_size: int, output_path: str) -> str:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    wiki_path = download_wikipedia_50mb()
    print(f"Training BPE tokenizer (vocab={vocab_size}) from {wiki_path}...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["eos_token"])
    with open(wiki_path, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    return output_path


def main():
    repo_id = "ScortexIA/laurelia"
    revision = "gens0x"

    hf = HFManager(repo_id=repo_id, revision=revision)
    hf._get_token()  # prompt upfront, not mid-training
    pusher = PeriodicPusher(hf, interval_minutes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prec = input("Precision (n=normal f32, f=f16): ").strip().lower()
    use_f16 = prec == "f"
    dtype = torch.float16 if use_f16 else torch.float32
    print(f"Using {dtype}")

    d_model = 768
    num_layers = 24
    num_heads = 12
    num_kv_groups = 4
    seq_len = 320
    batch_size = 4
    grad_accum = 4
    lr = 3e-4
    num_epochs = 20
    warmup_steps = 50
    bpe_vocab = 16000

    _DIR = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(_DIR, "tokenizer.json")
    tokenizer = None

    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        hf_tok = Tokenizer.from_file(tok_path)
        tokenizer = BPEWrapper(hf_tok)
        print(f"Loaded local tokenizer -> {tok_path}")
    else:
        tokenizer_loaded = False
        if hf.tokenizer_exists():
            try:
                print(f"Downloading tokenizer from {repo_id}@{revision}...")
                local_tok = hf.download_tokenizer(tok_path)
                from tokenizers import Tokenizer
                tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
                print(f"Loaded tokenizer from HF")
                tokenizer_loaded = True
            except Exception as e:
                print(f"Failed to download tokenizer: {e}")
        if not tokenizer_loaded:
            print("No tokenizer found. Training from Wikipedia 50MB...")
            train_tokenizer_from_wiki(bpe_vocab, tok_path)
            from tokenizers import Tokenizer
            hf_tok = Tokenizer.from_file(tok_path)
            tokenizer = BPEWrapper(hf_tok)
            hf.upload_tokenizer(tok_path, os.path.join(_DIR, "tokenizer_config.json"))
            print("Tokenizer trained and uploaded to HF")

    print(f"Vocab size: {tokenizer.vocab_size}")

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_groups=num_kv_groups,
        use_swiglu=True,
        use_x0=True,
        max_seq_len=seq_len,
        residual_dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    ).to(device).to(dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    global_step = 0
    epoch = 0
    ckpt_block = 0
    checkpoint_path = os.path.join(_DIR, "checkpoint.pt")
    safetensors_path = os.path.join(_DIR, "model_test.safetensors")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)
        ckpt_block = ckpt.get("block_idx", 0)
        print(f"Local checkpoint (step {global_step}, epoch {epoch}, block {ckpt_block})")
    elif hf.download_checkpoint(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)
        ckpt_block = ckpt.get("block_idx", 0)
        print(f"HF checkpoint (step {global_step}, epoch {epoch}, block {ckpt_block})")

    block_input = input(f"Block [{ckpt_block}]: ").strip()
    block_idx = int(block_input) if block_input else ckpt_block

    stream_data = StreamingDataset(block_mb=3.0, block_idx=block_idx)
    stream_data.load_tokens(tokenizer)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"dim={d_model} layers={num_layers} heads={num_heads} kv_groups={num_kv_groups} seq={seq_len} vocab={tokenizer.vocab_size}")
    print(f"batch={batch_size} grad_accum={grad_accum} lr={lr} warmup={warmup_steps} epochs={num_epochs}")
    print(f"model: {num_params:,} params ({num_params/1e6:.2f}M)")

    total_tokens = len(stream_data.get_tokens())
    tokens_per_epoch = (total_tokens - seq_len - 1) // seq_len
    total_steps = (tokens_per_epoch // batch_size) * num_epochs
    print(f"Tokens/block: {total_tokens} | Sequences/epoch: {tokens_per_epoch} | Steps/epoch: {tokens_per_epoch // batch_size} | Total steps: {total_steps}")
    print(f"LR: {lr} | Warmup: {warmup_steps} | Grad accum: {grad_accum}")

    model.train()
    start_time = time.time()
    last_report_time = start_time
    last_report_step = 0

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * (0.2 + 0.8 * (1.0 + torch.tensor(3.14159 * t).cos().item()) / 2.0)

    while True:
        tokens = stream_data.get_tokens()
        n_seq = (len(tokens) - seq_len - 1) // seq_len
        micro_step = 0
        for batch_start in range(0, n_seq, batch_size):
            if global_step >= total_steps:
                break

            batch_end = min(batch_start + batch_size, n_seq)
            actual_batch = batch_end - batch_start

            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                start_idx = i * seq_len
                x, y = create_batch(tokens, start_idx, 1, seq_len, seq_len)
                x_list.append(x)
                y_list.append(y)

            x = torch.cat(x_list, dim=0).to(device)
            y = torch.cat(y_list, dim=0).to(device)

            if micro_step == 0:
                current_lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                optimizer.zero_grad()

            logits = model.forward_train_partial_rope(x, rotary_pct=0.25)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.div(grad_accum).backward()

            micro_step += 1

            if micro_step >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                micro_step = 0

                if global_step % 10 == 0:
                    now = time.time()
                    tok = (global_step - last_report_step) * batch_size * grad_accum * seq_len
                    tps = tok / max(now - last_report_time, 0.001)
                    print(f"step {global_step} loss {loss.item():.4f} lr {current_lr:.6f} {tps:.0f}t/s")
                    last_report_time = now
                    last_report_step = global_step

                if global_step % 50 == 0:
                    sample = generate_sample(model, tokenizer, device, prompt="desde", max_new=30)
                    print(f"  >>> {sample}")

                if time.time() - pusher.last_push >= pusher.interval:
                    ckpt = {"global_step": global_step, "epoch": epoch, "block_idx": stream_data.block_idx,
                            "model": model.state_dict()}
                    torch.save(ckpt, checkpoint_path)
                    model.state_dict_to_safetensors(safetensors_path)
                    pusher.maybe_push(checkpoint_path, safetensors_path, tok_path, global_step)

        if micro_step > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        epoch += 1
        if epoch >= num_epochs:
            break

        stream_data.next_block()
        total_tokens = len(stream_data.get_tokens())
        n_seq = (total_tokens - seq_len - 1) // seq_len
        total_steps = (n_seq // batch_size) * (num_epochs - epoch)
        print(f"Epoch {epoch} done. Loaded next block: {total_tokens} tokens")

    # Final push
    hf.upload_checkpoint(checkpoint_path, safetensors_path, tok_path, global_step)
    print(f"Done! {global_step} steps in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()