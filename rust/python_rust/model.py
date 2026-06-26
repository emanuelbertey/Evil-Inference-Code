import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .rope import apply_rope_partial
from .attention import repeat_kv


class TransformerLM(nn.Module):
    """Full LM: Embedding → N×(PreNorm GQA+RoPE + PreNorm SwiGLU) → FinalNorm → Linear.

    Matches Rust TransformerBitLinearLM / TransformerLM forward pass exactly.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, num_kv_groups: int, max_seq_len: int = 2048,
                 norm_eps: float = 1e-5, ffn_expansion: float = 4.0,
                 ffn_round_to: int = 64, attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0, residual_dropout: float = 0.0,
                 attn_logit_cap: float | None = None, causal: bool = True):
        super().__init__()
        intermediate_dim = ((int(ffn_expansion * d_model * 2.0 / 3.0) + ffn_round_to - 1)
                            // ffn_round_to * ffn_round_to)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, num_kv_groups, intermediate_dim,
                attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout, attn_logit_cap=attn_logit_cap,
                causal=causal, norm_eps=norm_eps,
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward (training, no cache)."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, 0)
        x = self.final_norm(x)
        return self.head(x)

    def forward_train_partial_rope(self, input_ids: torch.Tensor,
                                   rotary_pct: float = 0.5) -> torch.Tensor:
        """Matches Rust forward_train_partial_rope."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            residual = x
            h = layer.attn_norm(x)
            attn = layer.attention
            B, S, _ = h.shape

            q = attn.q_proj(h).reshape(B, S, attn.num_heads, attn.head_dim)
            k = attn.k_proj(h).reshape(B, S, attn.num_kv_groups, attn.head_dim)
            v = attn.v_proj(h).reshape(B, S, attn.num_kv_groups, attn.head_dim)

            q, k = apply_rope_partial(q, k, 0, rotary_pct)

            k = repeat_kv(k, attn.num_heads, attn.num_kv_groups)
            v = repeat_kv(v, attn.num_heads, attn.num_kv_groups)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scale = math.sqrt(attn.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale

            if S > 1:
                mask = torch.triu(
                    torch.full((S, S), float("-inf"), device=x.device), diagonal=1,
                )
                scores = scores + mask.unsqueeze(0).unsqueeze(0)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
            h_attn = attn.o_proj(attn_output)

            x = residual + h_attn
            residual = x
            h = layer.ffn_norm(x)
            h_ffn = layer.ffn(h)
            x = residual + h_ffn

        x = self.final_norm(x)
        return self.head(x)

    # ─── Weight I/O ──────────────────────────────────────────────────────

    def save_safetensors(self, path: str):
        from safetensors.torch import save_file
        save_file(self.state_dict(), path)

    @staticmethod
    def load_safetensors(path: str, **kwargs):
        from safetensors.torch import load_file
        state = load_file(path)
        model = TransformerLM(**kwargs)
        model.load_state_dict(state)
        return model

    def export_for_rust(self, path: str, mapping_path: str | None = None):
        from safetensors.torch import save_file
        state = self.state_dict()
        save_file(state, path)
        if mapping_path is None:
            mapping_path = path.replace(".safetensors", "_mapping.json")
        mapping = {"parameters": {}}
        for name, tensor in state.items():
            mapping["parameters"][name] = {
                "shape": list(tensor.shape), "dtype": "f32", "burn_name": name,
            }
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Exported {len(state)} tensors → {path}")
        print(f"Mapping → {mapping_path}")
