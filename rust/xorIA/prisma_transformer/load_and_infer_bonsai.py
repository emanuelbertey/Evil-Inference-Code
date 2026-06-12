import argparse
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from config import PrismaConfig
from transformer import PrismaTransformer

GGUF_ALIGN = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_Q1_0 = 42
GGML_TYPE_PRISMA_TQ2_0 = 42

VALUE_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}

@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    tensor_type: int
    offset: int

def align(pos: int, alignment: int = GGUF_ALIGN) -> int:
    return (pos + alignment - 1) // alignment * alignment

class GGUFLoader:
    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.tensors_info: dict[str, TensorInfo] = {}
        self.data_offset = 0
        self._parse()

    def _read_string(self, f) -> str:
        size = struct.unpack("<Q", f.read(8))[0]
        return f.read(size).decode("utf-8", errors="replace")

    def _skip_value(self, f, value_type: int):
        if value_type == 8:
            self._read_string(f)
        elif value_type == 9:
            item_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            if item_type == 8:
                for _ in range(count):
                    self._read_string(f)
            elif item_type in VALUE_SIZES:
                f.seek(VALUE_SIZES[item_type] * count, os.SEEK_CUR)
            else:
                raise ValueError(f"Tipo de array GGUF no soportado: {item_type}")
        elif value_type in VALUE_SIZES:
            f.seek(VALUE_SIZES[value_type], os.SEEK_CUR)
        else:
            raise ValueError(f"Tipo de metadata GGUF no soportado: {value_type}")

    def _parse_value(self, f, value_type: int):
        if value_type == 8:
            return self._read_string(f)
        if value_type == 9:
            item_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            if item_type == 8:
                return [self._read_string(f) for _ in range(count)]
            if item_type == 4:
                return list(struct.unpack(f"<{count}I", f.read(count * 4)))
            if item_type == 5:
                return list(struct.unpack(f"<{count}i", f.read(count * 4)))
            if item_type == 6:
                return list(struct.unpack(f"<{count}f", f.read(count * 4)))
            if item_type == 7:
                return list(struct.unpack(f"<?", f.read(1))[0] for _ in range(count))
            self._skip_value(f, item_type)
            return None
        if value_type == 4:
            return struct.unpack("<I", f.read(4))[0]
        if value_type == 5:
            return struct.unpack("<i", f.read(4))[0]
        if value_type == 6:
            return struct.unpack("<f", f.read(4))[0]
        if value_type == 7:
            return bool(struct.unpack("<?", f.read(1))[0])
        if value_type == 10:
            return struct.unpack("<Q", f.read(8))[0]
        if value_type == 11:
            return struct.unpack("<q", f.read(8))[0]
        if value_type == 12:
            return struct.unpack("<d", f.read(8))[0]
        self._skip_value(f, value_type)
        return None

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"No parece GGUF: magic={magic!r}")
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(metadata_count):
                key = self._read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                self.metadata[key] = self._parse_value(f, value_type)

            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                tensor_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                self.tensors_info[name] = TensorInfo(name, shape, tensor_type, offset)

            alignment = int(self.metadata.get("general.alignment") or GGUF_ALIGN)
            self.data_offset = align(f.tell(), alignment)
            print(
                f">>> [GGUF] version={version} tensors={tensor_count} "
                f"metadata={metadata_count} data_offset={self.data_offset}",
                flush=True,
            )

def _copy_norm(f, target, info: TensorInfo):
    count = target.weight.numel()
    if info.tensor_type == GGML_TYPE_F32:
        data = np.frombuffer(f.read(count * 4), dtype="<f4")
    elif info.tensor_type == GGML_TYPE_F16:
        data = np.frombuffer(f.read(count * 2), dtype="<f2").astype(np.float32)
    else:
        raise ValueError(f"{info.name} deberia ser F16/F32, tipo={info.tensor_type}")
    if data.size != count:
        raise ValueError(f"Lectura corta en {info.name}: {data.size} != {count}")
    target.weight.data.copy_(torch.from_numpy(data.copy()).view_as(target.weight))

def _copy_tq2(f, target, info: TensorInfo):
    rows = int(info.shape[1]) if len(info.shape) > 1 else 1
    blocks_per_row = int(info.shape[0]) // target.QK
    block_stride = 34
    row_bytes = blocks_per_row * block_stride
    stride = row_bytes
    raw = f.read(rows * stride)
    if len(raw) != rows * stride:
        raise ValueError(f"Lectura corta en {info.name}: {len(raw)} != {rows * stride}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(target.num_blocks, block_stride)
    d = blocks[:, :2].copy().view(dtype=np.float16).astype(np.float32).reshape(-1)
    target.blocks_d.copy_(torch.from_numpy(d))
    target.blocks_qs.copy_(torch.from_numpy(blocks[:, 2:34].copy()))

def _copy_q1(f, target, info: TensorInfo):
    expected = target.num_blocks * 6
    raw = f.read(expected)
    if len(raw) != expected:
        raise ValueError(f"Lectura corta en {info.name}: {len(raw)} != {expected}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(target.num_blocks, 6)
    d = arr[:, :2].copy().view(dtype=np.float16).astype(np.float32).reshape(-1)
    target.blocks_d.copy_(torch.from_numpy(d))
    target.blocks_qs.copy_(torch.from_numpy(arr[:, 2:6].copy()))

def load_weights_into_model(model: PrismaTransformer, loader: GGUFLoader):
    print(">>> [CARGA] Inyectando pesos GGUF...", flush=True)
    mapping = {
        "token_embd.weight": model.tok_embeddings,
        "output_norm.weight": model.norm,
    }

    for i, layer in enumerate(model.layers):
        mapping.update({
            f"blk.{i}.attn_q.weight": layer.attention.wq,
            f"blk.{i}.attn_q_norm.weight": layer.attention.q_norm,
            f"blk.{i}.attn_k.weight": layer.attention.wk,
            f"blk.{i}.attn_k_norm.weight": layer.attention.k_norm,
            f"blk.{i}.attn_v.weight": layer.attention.wv,
            f"blk.{i}.attn_output.weight": layer.attention.wo,
            f"blk.{i}.ffn_gate.weight": layer.feed_forward.w1,
            f"blk.{i}.ffn_up.weight": layer.feed_forward.w3,
            f"blk.{i}.ffn_down.weight": layer.feed_forward.w2,
            f"blk.{i}.attn_norm.weight": layer.attention_norm,
            f"blk.{i}.attn_sub_norm.weight": layer.attention.attn_sub_norm,
            f"blk.{i}.ffn_norm.weight": layer.ffn_norm,
            f"blk.{i}.ffn_sub_norm.weight": layer.feed_forward.ffn_sub_norm,
        })

    loaded = 0
    with open(loader.path, "rb") as f:
        for name, target in mapping.items():
            info = loader.tensors_info.get(name)
            if info is None:
                print(f"AVISO: falta {name}", flush=True)
                continue
            f.seek(loader.data_offset + info.offset)
            if hasattr(target, "blocks_qs"):
                if info.tensor_type in {GGML_TYPE_TQ2_0, GGML_TYPE_PRISMA_TQ2_0}:
                    _copy_tq2(f, target, info)
                elif info.tensor_type == GGML_TYPE_Q1_0 and target.blocks_qs.shape[1] == 4:
                    _copy_q1(f, target, info)
                else:
                    raise ValueError(f"{name} deberia ser TQ2_0 tipo 35 o Q1_0 tipo 42, tipo={info.tensor_type}")
            else:
                _copy_norm(f, target, info)
            if name.endswith(".attn_sub_norm.weight"):
                layer_idx = int(name.split(".")[1])
                model.layers[layer_idx].attention.use_attn_sub_norm = True
            elif name.endswith(".ffn_sub_norm.weight"):
                layer_idx = int(name.split(".")[1])
                model.layers[layer_idx].feed_forward.use_ffn_sub_norm = True
            loaded += 1
            print(f"OK: {name}", flush=True)
    print(f">>> [CARGA] {loaded}/{len(mapping)} tensores cargados.", flush=True)

def tokenizer_from_gguf(loader: GGUFLoader) -> PreTrainedTokenizerFast:
    tokens = loader.metadata.get("tokenizer.ggml.tokens")
    merges = loader.metadata.get("tokenizer.ggml.merges")
    token_types = loader.metadata.get("tokenizer.ggml.token_type") or []
    if not tokens or not merges:
        raise ValueError("El GGUF no contiene tokenizer.ggml.tokens/merges")

    vocab = {token: idx for idx, token in enumerate(tokens)}
    bpe_merges = [tuple(merge.split(" ", 1)) for merge in merges]
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=bpe_merges, fuse_unk=False))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    added_tokens = [
        token
        for token, token_type in zip(tokens, token_types)
        if token_type in {3, 4}
    ]
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        chat_template=loader.metadata.get("tokenizer.chat_template"),
        eos_token=tokens[int(loader.metadata["tokenizer.ggml.eos_token_id"])],
        pad_token=tokens[int(loader.metadata["tokenizer.ggml.padding_token_id"])],
        bos_token=None,
        add_bos_token=bool(loader.metadata.get("tokenizer.ggml.add_bos_token")),
    )
    fast.add_special_tokens({"additional_special_tokens": added_tokens})
    return fast

def config_from_gguf(loader: GGUFLoader) -> PrismaConfig:
    emb = loader.tensors_info["token_embd.weight"]
    dim = int(emb.shape[0])
    vocab_size = int(emb.shape[1])
    layer_ids = [
        int(name.split(".")[1])
        for name in loader.tensors_info
        if name.startswith("blk.") and name.endswith(".attn_q.weight")
    ]
    n_layers = max(layer_ids) + 1
    ffn = loader.tensors_info.get("blk.0.ffn_gate.weight")
    hidden_dim = int(ffn.shape[1]) if ffn is not None and len(ffn.shape) > 1 else 6144
    return PrismaConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=16,
        n_heads_kv=8,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        quant_mode="tq2_0",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"D:\Ternary-Bonsai-1.7B-Q2_0.gguf")
    parser.add_argument("--tokenizer", default="gguf")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    loader = GGUFLoader(args.model)
    config = config_from_gguf(loader)
    print(f">>> [CONFIG] {config}", flush=True)
    model = PrismaTransformer(config, lazy=True).eval()
    if args.tokenizer.lower() == "gguf":
        tokenizer = tokenizer_from_gguf(loader)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    load_weights_into_model(model, loader)
    print("\n>>> [CHAT] Bonsai 1.7B listo.", flush=True)

    while True:
        try:
            user = input("\nUsuario > ")
            messages = [{"role": "user", "content": user}]
            try:
                toks = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
                if hasattr(toks, "input_ids"):
                    toks = toks.input_ids
                elif isinstance(toks, dict) and "input_ids" in toks:
                    toks = toks["input_ids"]
                elif isinstance(toks, str):
                    toks = tokenizer.encode(toks, return_tensors="pt")
                elif not isinstance(toks, torch.Tensor):
                    toks = torch.tensor([toks], dtype=torch.long)
            except Exception:
                prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                toks = tokenizer.encode(prompt, return_tensors="pt")
            kv, offset = None, 0
            print("Bonsai > ", end="", flush=True)
            for _ in range(args.max_new_tokens):
                logits, kv = model(toks, kv_caches=kv, offset=offset)
                next_t = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                text = tokenizer.decode(next_t[0], skip_special_tokens=False)
                print(text, end="", flush=True)
                offset += toks.shape[1]
                toks = next_t
                if next_t.item() in {tokenizer.eos_token_id, 151645}:
                    break
            print()
        except KeyboardInterrupt:
            print()
            break

if __name__ == "__main__":
    main()
