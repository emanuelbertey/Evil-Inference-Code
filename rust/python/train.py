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
    pusher = PeriodicPusher(hf, interval_minutes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    d_model = 768
    num_layers = 24
    num_heads = 12
    num_kv_groups = 4
    seq_len = 128
    batch_size = 16
    grad_accum = 2
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

    stream_data = StreamingDataset(block_mb=3.0, block_idx=0)
    stream_data.load_tokens(tokenizer)

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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    global_step = 0
    checkpoint_path = os.path.join(_DIR, "checkpoint.pt")
    safetensors_path = os.path.join(_DIR, "model_test.safetensors")

    if hf.download_checkpoint(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from HF checkpoint (step {global_step})")
    elif os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from local checkpoint (step {global_step})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"dim={d_model} layers={num_layers} heads={num_heads} kv_groups={num_kv_groups} seq={seq_len} vocab={tokenizer.vocab_size}")
    print(f"batch={batch_size} grad_accum={grad_accum} lr={lr} warmup={warmup_steps} epochs={num_epochs}")
    print(f"model: {num_params:,} params ({num_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_batches_per_block = (len(stream_data.get_tokens()) - seq_len) // seq_len
    print(f"Tokens/block: {len(stream_data.get_tokens())} | Batches/block: {total_batches_per_block}")
    print(f"LR: {lr} | Warmup: {warmup_steps} | Grad accum: {grad_accum}")

    torch.save({"global_step": global_step, "model": model.state_dict()}, checkpoint_path)

    model.train()
    start_time = time.time()
    epoch = 0

    while True:
        tokens = stream_data.get_tokens()
        for batch_idx in range(0, total_batches_per_block, batch_size):
            if global_step >= num_epochs * total_batches_per_block:
                break

            if global_step < warmup_steps:
                current_lr = lr * global_step / max(warmup_steps, 1)
            else:
                t = (global_step - warmup_steps) / max(num_epochs * total_batches_per_block - warmup_steps, 1)
                current_lr = lr * (0.2 + 0.8 * (1.0 + torch.tensor(3.14159 * t).cos().item()) / 2.0)

            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            micro_loss = 0.0
            micro_count = 0
            optimizer.zero_grad()

            for micro in range(batch_size):
                start_idx = (batch_idx + micro) * seq_len
                if start_idx + seq_len + 1 >= len(tokens):
                    break

                x, y = create_batch(tokens, start_idx, 1, seq_len, seq_len)
                x, y = x.to(device), y.to(device)

                logits = model.forward_train_partial_rope(x, rotary_pct=0.25)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

                (loss / grad_accum).backward()
                micro_loss += loss.item()
                micro_count += 1

                if (micro + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            global_step += 1

            if global_step % 10 == 0:
                avg_loss = micro_loss / max(micro_count, 1)
                print(f"step {global_step} loss {avg_loss:.4f} lr {current_lr:.6f}")
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                tps = micro_count * seq_len / max(elapsed, 0.001)
                print(f"step {global_step} | {tps:.0f} tok/s")

            if pusher.maybe_push(checkpoint_path, safetensors_path, tok_path, global_step):
                torch.save({"global_step": global_step, "model": model.state_dict()}, checkpoint_path)
                model.state_dict_to_safetensors(safetensors_path)

        epoch += 1
        if epoch >= num_epochs:
            break

        stream_data.next_block()
        total_batches_per_block = (len(stream_data.get_tokens()) - seq_len) // seq_len
        print(f"Epoch {epoch} done. Loaded next block: {len(stream_data.get_tokens())} tokens")

    # Final push
    hf.upload_checkpoint(checkpoint_path, safetensors_path, tok_path, global_step)
    print(f"Done! {global_step} steps in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()