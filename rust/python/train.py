"""Quick test: train the Python transformer on input.txt with BPE tokenizer.

Usage:
    python train.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.model import TransformerLM


def train_bpe_tokenizer(text, vocab_size=8000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special = "eos_token"
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[special],
    )

    temp_file = "temp_train_bpe.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    tokenizer.train(files=[temp_file], trainer=trainer)
    os.remove(temp_file)

    return tokenizer


class BPEWrapper:
    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        enc = self.tokenizer.encode(text)
        return enc.ids

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
    x = torch.tensor(x_indices, dtype=torch.long).view(batch_size, seq_len)
    y = torch.tensor(y_indices, dtype=torch.long).view(batch_size, seq_len)
    return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    d_model = 256
    num_layers = 6
    num_heads = 8
    num_kv_groups = 4
    seq_len = 128
    batch_size = 16
    grad_accum = 2
    lr = 3e-4
    num_epochs = 20
    warmup_steps = 50
    bpe_vocab = 16000

    text_path = os.path.join(os.path.dirname(__file__), "..", "xorIA", "input.txt")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} chars from input.txt")

    tok_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        hf_tok = Tokenizer.from_file(tok_path)
        print(f"Loaded existing tokenizer from {tok_path}")
    else:
        print(f"Training BPE tokenizer (vocab={bpe_vocab})...")
        hf_tok = train_bpe_tokenizer(text, vocab_size=bpe_vocab)
        hf_tok.save(tok_path)
        print(f"Tokenizer saved to {tok_path}")
        cfg_path = os.path.join(os.path.dirname(__file__), "tokenizer_config.json")
        with open(cfg_path, "w") as f:
            f.write('{"tokenizer_class": "BPE", "eos_token": "eos_token", "model_max_length": 2048}\n')
        print(f"Config saved to {cfg_path}")
    tokenizer = BPEWrapper(hf_tok)
    print(f"Vocab size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")

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

    start_epoch = 0
    global_step = 0
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    safetensors_path = os.path.join(os.path.dirname(__file__), "model_test.safetensors")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path)
        model.load_state_dict(state, strict=False)
        del state
        print(f"Loaded from safetensors: {safetensors_path}")
    else:
        print("No checkpoint found, starting from scratch.")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} params ({num_params/1e6:.2f}M)")

    total_batches = (len(tokens) - seq_len) // seq_len
    total_steps = total_batches * num_epochs
    print(f"Batches/epoch: {total_batches} | Total steps: {total_steps}")
    print(f"LR: {lr} | Warmup: {warmup_steps} steps | Grad accum: {grad_accum}")
    print(f"\nStarting training from epoch {start_epoch + 1}...\n")

    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        for batch_idx in range(0, total_batches, batch_size):
            if global_step < warmup_steps:
                current_lr = lr * global_step / max(warmup_steps, 1)
            elif global_step < total_steps:
                t = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                current_lr = lr * (0.2 + 0.8 * (1.0 + torch.tensor(3.14159 * t).cos().item()) / 2.0)
            else:
                current_lr = lr * 0.2

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

                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

                (loss / grad_accum).backward()
                micro_loss += loss.item()
                micro_count += 1

                if (micro + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += micro_loss
            epoch_tokens += micro_count * seq_len
            global_step += 1

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / max(epoch_tokens, 1) * seq_len
                tps = epoch_tokens / max(elapsed, 0.001)
                print(f"  Epoch {epoch+1}/{num_epochs} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | {tps:.0f} tok/s")

        epoch_elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_tokens, 1) * seq_len
        print(f"Epoch {epoch+1}/{num_epochs} done | Loss: {avg_loss:.4f} | {epoch_elapsed:.1f}s")

        model.eval()
        seed = "First Citizen:"
        seed_ids = tokenizer.encode(seed)
        input_tensor = torch.tensor([seed_ids], dtype=torch.long).to(device)

        gen_start = time.time()
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
            )
        gen_elapsed = time.time() - gen_start
        gen_tokens = output.shape[1] - len(seed_ids)
        gen_tps = gen_tokens / max(gen_elapsed, 0.001)

        generated = tokenizer.decode(output[0].tolist())
        print(f"\n--- Sample (epoch {epoch+1}) ---")
        print(generated)
        print(f"--- Generated {gen_tokens} tokens in {gen_elapsed:.1f}s ({gen_tps:.1f} tok/s) ---\n")
        model.train()

        save_path = os.path.join(os.path.dirname(__file__), f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        safetensors_path = os.path.join(os.path.dirname(__file__), "model_test.safetensors")
        mapping_path = os.path.join(os.path.dirname(__file__), "model_test_mapping.json")
        model.export_for_rust(safetensors_path, mapping_path)

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
        }, checkpoint_path)

    print("Done!")


if __name__ == "__main__":
    main()