"""xLSTM MoE Model: mLSTM blocks with MoE FFN."""
import sys, os, math
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", ".."))

import torch
from torch import nn
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_moe_block import xLSTMMoEBlock

class xLSTMMoEModel(nn.Module):
    """LM model: embedding → N xLSTMMoEBlock → lm_head."""

    def __init__(self, vocab_size, d_model=512, num_layers=8, num_heads=4, use_moe=True,
                 n_experts=4, top_k=1, n_shared=1, expert_dim=None, capacity_factor=1.0,
                 z_loss_gamma=0.001, bias_decay=0.1, noise_std=0.0, max_seq_len=1024):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_drop = nn.Dropout(0.0)

        # mLSTM config
        mlstm_cfg = mLSTMLayerConfig(
            embedding_dim=d_model,
            num_heads=num_heads,
            context_length=max_seq_len,
            dropout=0.0,
            bias=False,
        )

        # MoE config
        moe_cfg = {
            "expert_dim": expert_dim or d_model * 4,
            "n_experts": n_experts if use_moe else 1,
            "top_k": top_k if use_moe else 1,
            "n_shared": n_shared if use_moe else 0,
            "capacity_factor": capacity_factor,
            "z_loss_gamma": z_loss_gamma,
            "bias_decay": bias_decay,
            "noise_std": noise_std,
        }

        self.blocks = nn.ModuleList([
            xLSTMMoEBlock(d_model, mlstm_cfg, moe_cfg, layer_idx=i)
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        x = self.embed_drop(x)
        aux_loss = 0.0
        for block in self.blocks:
            x, loss = block(x)
            aux_loss = aux_loss + loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss

    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx[:, -1024:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=-1)
        return idx
