"""xLSTM Large MoE block: xLSTMBlock with mLSTM + MoE FFN."""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", ".."))

import torch
from torch import nn
from xlstm.blocks.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from moe import MoELayer

class xLSTMMoEBlock(nn.Module):
    """xLSTMBlock (mLSTM) with MoE replacing the standard FeedForward."""

    def __init__(self, d_model, mlstm_cfg, moe_cfg, layer_idx=0):
        super().__init__()
        # xLSTMBlock without FFN
        block_cfg = xLSTMBlockConfig(
            mlstm=mlstm_cfg,
            slstm=None,
            feedforward=None,
            _num_blocks=1,
            _block_idx=layer_idx,
        )
        self.block = xLSTMBlock(config=block_cfg)

        # MoE in place of FFN
        self.moe_norm = nn.LayerNorm(d_model)  # xLSTMBlock's pre-FFN norm is inside block, use another
        self.moe = MoELayer(
            d_model=d_model,
            expert_dim=moe_cfg["expert_dim"],
            n_experts=moe_cfg["n_experts"],
            top_k=moe_cfg["top_k"],
            n_shared=moe_cfg["n_shared"],
            capacity_factor=moe_cfg.get("capacity_factor", 1.0),
            z_loss_gamma=moe_cfg.get("z_loss_gamma", 0.001),
            bias_decay=moe_cfg.get("bias_decay", 0.1),
            noise_std=moe_cfg.get("noise_std", 0.0),
        )

    def forward(self, x, **kwargs):
        x = self.block(x, **kwargs)
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, aux_loss

    def step(self, x, mlstm_state=None, conv_state=None):
        x_xlstm, state = self.block.xlstm.step(self.block.xlstm_norm(x), mlstm_state=mlstm_state, conv_state=conv_state)
        x = x + x_xlstm
        moe_out, _ = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, state
