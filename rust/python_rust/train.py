#!/usr/bin/env python3
"""Streaming TPU training → HuggingFace (ScortexIA/laurelia @ gens0x).

Flujo:
  1. Toma MY_KEY de Colab secret (token HF)
  2. Descarga primeros 60MB de Wikipedia si no hay tokenizer en HF
  3. Entrena BPE tokenizer → sube a ScortexIA/laurelia
  4. Entrena en streaming (flujo constante)
  5. Push a HF cada 10 minutos con tag gens0x
"""

import math
import os
import json
import time
import io

import torch
import torch.nn.functional as F

from .model import TransformerLM
from .tokenizer import BPEWrapper
from . import CONFIG


# ─── Streaming file reader (como Rust FileFragmentIterator) ─────────────────

class FileFragmentStream:
    def __init__(self, path: str, buffer_size_mb: int = 2):
        self.path = path
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "rb")
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def read_fragment(self) -> bytes | None:
        buffer = self._file.read(self.buffer_size)
        if not buffer:
            return None
        while buffer and buffer[-1] & 0xC0 == 0x80:
            buffer = buffer[:-1]
        if buffer and buffer[-1] & 0x80 and not (buffer[-1] & 0x40):
            buffer = buffer[:-1]
        return buffer if buffer else None

    def __iter__(self):
        return self

    def __next__(self) -> str:
        chunk = self.read_fragment()
        if chunk is None:
            raise StopIteration
        return chunk.decode("utf-8", errors="replace")


# ─── Data source: auto-download Wikipedia ES ──────────────────────────────

DATA_FILE = "/content/evil_inference_bit2/rust/xorIA/input.txt"
WIKI_ES = ("wikimedia/wikipedia", "20231101.es")

def _download_wiki(output_file: str, max_bytes: int):
    from datasets import load_dataset
    ds = load_dataset(*WIKI_ES, split="train", streaming=True)
    with open(output_file, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            texto = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(texto.encode('utf-8'))
            if written + tam > max_bytes:
                break
            f.write(texto)
            written += tam
    return written

def download_training_data(max_mb: float = 6.0):
    out = DATA_FILE
    out_dir = os.path.dirname(out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024)
    if os.path.exists(out) and os.path.getsize(out) >= max_bytes:
        print(f"Training data already at {out} ({os.path.getsize(out)} bytes)")
        return out
    print(f"Downloading Wikipedia ES ({max_mb}MB block) to {out}...")
    written = _download_wiki(out, max_bytes)
    print(f"Written {written} bytes to {out}")
    return out

def download_wiki_tokenizer(vocab_size: int):
    path = "/tmp/wiki_tokenizer.txt"
    if os.path.exists(path) and os.path.getsize(path) >= 50_000_000:
        return path
    print("Downloading Wikipedia ES for tokenizer training...")
    written = _download_wiki(path, 60_000_000)
    print(f"Written {written} bytes to {path}")
    return path


# ─── HF helpers (token explícito, nada interactivo) ──────────────────────

def hf_token() -> str:
    return os.environ.get("MY_KEY", "")

def hf_api() -> "HfApi":
    from huggingface_hub import HfApi
    return HfApi(token=hf_token())

def train_tokenizer_from_wiki(vocab_size: int, output_path: str = "tokenizer.json"):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    wiki_path = download_wiki_tokenizer(vocab_size)
    print(f"Training BPE tokenizer (vocab={vocab_size}) from {wiki_path}...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["eos_token"])
    with open(wiki_path, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    return output_path

def push_to_hf(repo_id: str, local_files: list, revision: str = "gens0x",
               commit_message: str = ""):
    api = hf_api()
    try:
        from huggingface_hub import create_repo
        create_repo(repo_id=repo_id, exist_ok=True, private=False, token=hf_token())
    except Exception:
        pass
    for p in local_files:
        fname = os.path.basename(p)
        api.upload_file(path_or_fileobj=p, path_in_repo=fname,
                        repo_id=repo_id, revision=revision,
                        commit_message=commit_message or f"Update {fname}")
    print(f"Pushed {len(local_files)} files to {repo_id} @ {revision}")


# ─── Batch builder ──────────────────────────────────────────────────────────

def create_batch(tokens, start_idx, batch_size, seq_len, stride):
    x_indices, y_indices = [], []
    for i in range(batch_size):
        s = start_idx + i * stride
        for j in range(seq_len):
            x_indices.append(tokens[s + j])
            y_indices.append(tokens[s + j + 1])
    B, S = batch_size, seq_len
    return (torch.tensor(x_indices, dtype=torch.long).view(B, S),
            torch.tensor(y_indices, dtype=torch.long).view(B, S))


# ─── Main ───────────────────────────────────────────────────────────────────

def train():
    IS_TPU = False
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"TPU device: {device}")
        IS_TPU = True
    except (ImportError, RuntimeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} (no TPU)")

    cfg = CONFIG
    print("Config:", json.dumps(cfg, indent=2))

    # ── HF token ──
    hf_logged_in = bool(hf_token())
    repo_id = cfg["hf_repo"]
    tag = cfg["hf_tag"]

    # ── Tokenizer: local > HF download > Wikipedia ──
    tok_path = "tokenizer.json"
    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(tok_path)
        print(f"Loaded tokenizer → {tok_path}")
    else:
        downloaded = False
        if hf_logged_in:
            try:
                from huggingface_hub import hf_hub_download
                tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json",
                                           revision=tag)
                tokenizer = BPEWrapper(tok_path)
                print(f"Downloaded tokenizer from {repo_id} @ {tag}")
                downloaded = True
            except Exception:
                pass
        if not downloaded:
            train_tokenizer_from_wiki(cfg["vocab_size"], tok_path)
            tokenizer = BPEWrapper(tok_path)
            if hf_logged_in:
                with open("tokenizer_config.json", "w") as f:
                    json.dump({"tokenizer_class": "BPE", "eos_token": "eos_token"}, f)
                push_to_hf(repo_id, [tok_path, "tokenizer_config.json"],
                           revision=tag, commit_message="Add tokenizer")

    # ── Model ──
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_kv_groups=cfg["num_kv_groups"],
        max_seq_len=cfg["max_seq_len"],
        norm_eps=cfg["norm_eps"],
        ffn_expansion=cfg["ffn_expansion"],
        ffn_round_to=cfg["ffn_round_to"],
        attn_dropout=cfg["attn_dropout"],
        ffn_dropout=cfg["ffn_dropout"],
        residual_dropout=cfg["residual_dropout"],
    ).to(device)

    safetensors_path = "model_tpu.safetensors"
    checkpoint_path = "checkpoint_tpu.pt"
    start_step = 0
    # Try downloading latest from HF first
    if hf_logged_in:
        try:
            from huggingface_hub import hf_hub_download
            ckpt_hf = hf_hub_download(repo_id=repo_id, filename="checkpoint_tpu.pt",
                                      revision=tag)
            ckpt = torch.load(ckpt_hf, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)
            print(f"Resumed from HF checkpoint (step {start_step})")
        except Exception:
            pass
    if start_step == 0:
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)
            print(f"Resumed from local checkpoint (step {start_step})")
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
            model.load_state_dict(state, strict=False)
            del state
            print(f"Loaded from {safetensors_path}")

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {nparams:,} params ({nparams/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = cfg["total_steps"]

    # ── Data source (auto-download Wikipedia ES) ──
    data_path = download_training_data()

    # ── Train (flujo constante) ──
    model.train()
    global_step = start_step
    start_time = time.time()
    last_push_time = time.time()
    push_interval = cfg["push_every_minutes"] * 60
    print(f"\nTraining from step {global_step}. Push every {cfg['push_every_minutes']}min.\n")

    ring = []
    ring_max = cfg["batch_size"] * cfg["stride"] + cfg["max_seq_len"] + 1

    while global_step < total_steps:
        with FileFragmentStream(data_path, buffer_size_mb=2) as stream:
            for fragment in stream:
                if global_step >= total_steps:
                    break

                fragment_ids = tokenizer.encode(fragment)
                ring.extend(fragment_ids)

                while len(ring) >= ring_max:
                    batch_loss = 0.0
                    batch_count = 0
                    optimizer.zero_grad()

                    for micro in range(cfg["batch_size"]):
                        start_idx = micro * cfg["stride"]
                        if start_idx + cfg["stride"] + 1 >= len(ring):
                            break
                        x, y = create_batch(ring, start_idx, 1,
                                            cfg["max_seq_len"], cfg["stride"])
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, tokenizer.vocab_size), y.view(-1)
                        )
                        (loss / cfg["batch_size"]).backward()
                        batch_loss += loss.item()
                        batch_count += 1

                    ring = ring[cfg["batch_size"] * cfg["stride"]:]

                    # LR
                    if global_step < cfg["warmup_steps"]:
                        current_lr = cfg["lr"] * global_step / max(cfg["warmup_steps"], 1)
                    else:
                        t = (global_step - cfg["warmup_steps"]) / max(
                            total_steps - cfg["warmup_steps"], 1)
                        current_lr = cfg["lr"] * (
                            cfg["lr_min_ratio"]
                            + (1.0 - cfg["lr_min_ratio"])
                            * (1.0 + math.cos(math.pi * t)) / 2.0
                        )
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr

                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_norm"])
                    if IS_TPU:
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    global_step += 1

                    if global_step % 50 == 0:
                        elapsed = time.time() - start_time
                        avg_loss = batch_loss / max(batch_count, 1)
                        tps = (batch_count * cfg["max_seq_len"]
                               / max(time.time() - start_time, 0.001))
                        print(
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.6f} | "
                            f"{tps:.0f} tok/s"
                        )

                    # Push cada 10 minutos
                    now = time.time()
                    if hf_logged_in and (now - last_push_time) >= push_interval:
                        torch.save({"step": global_step, "model": model.state_dict()},
                                   checkpoint_path)
                        model.save_safetensors(safetensors_path)
                        push_to_hf(
                            repo_id,
                            [checkpoint_path, safetensors_path,
                             "tokenizer.json", "tokenizer_config.json"],
                            revision=tag,
                            commit_message=f"Step {global_step} @ {time.strftime('%Y-%m-%d %H:%M UTC')}",
                        )
                        last_push_time = now
                        print(f"  Pushed to {repo_id} @ {tag} (step {global_step})")

    # Final push
    torch.save({"step": global_step, "model": model.state_dict()}, checkpoint_path)
    model.save_safetensors(safetensors_path)
    if hf_logged_in:
        push_to_hf(
            repo_id,
            [checkpoint_path, safetensors_path,
             "tokenizer.json", "tokenizer_config.json"],
            revision=tag,
            commit_message=f"Final step {global_step}",
        )

    elapsed = time.time() - start_time
    print(f"\nDone! {global_step} steps in {elapsed:.1f}s")

    # Sample
    model.eval()
    seed = "First Citizen:"
    seed_ids = tokenizer.encode(seed)
    input_tensor = torch.tensor([seed_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(200):
            logits = model(input_tensor)
            probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    print(f"\n--- Sample ---\n{tokenizer.decode(input_tensor[0].tolist())}\n--------------")


if __name__ == "__main__":
    train()
