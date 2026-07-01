"""xLSTM + MoE block: replaces FFN with MoE experts."""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", ".."))

import torch
from torch import nn
from xlstm.blocks.mlstm.layer import mLSTMLayer
from xlstm.components.ln import LayerNorm
from moe import MoELayer

class xLSTMMoEBlock(nn.Module):
    """xLSTM block with MoE FFN: norm → mLSTM → residual → norm → MoE → residual."""

    def __init__(self, d_model, mlstm_cfg, moe_cfg, layer_idx=0):
        super().__init__()
        self.mlstm_norm = LayerNorm(ndim=d_model, weight=True, bias=False)
        self.mlstm = mLSTMLayer(config=mlstm_cfg)

        self.moe_norm = LayerNorm(ndim=d_model, weight=True, bias=False)
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
        self._layer_idx = layer_idx

    def forward(self, x, **kwargs):
        x = x + self.mlstm(self.mlstm_norm(x))
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, aux_loss

    def step(self, x, mlstm_state=None, conv_state=None):
        x_mlstm, state = self.mlstm.step(self.mlstm_norm(x), mlstm_state=mlstm_state, conv_state=conv_state)
        x = x + x_mlstm
        moe_out, _ = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, state

    def reset_parameters(self):
        self.mlstm.reset_parameters()
        self.mlstm_norm.reset_parameters()
        self.moe_norm.reset_parameters()
        # MoE params initialized in MoELayer.__init__
