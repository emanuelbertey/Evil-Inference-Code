import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PrismaConfig
from layers import Q1_0_Embedding, TernaryEmbedding, RMSNorm

class PrismaTransformer(nn.Module):
    def __init__(self, config: PrismaConfig, lazy=True):
        super().__init__()
        self.config = config
        if config.quant_mode == "q1_0":
            self.tok_embeddings = Q1_0_Embedding(config.vocab_size, config.dim)
        else:
            self.tok_embeddings = TernaryEmbedding(config.vocab_size, config.dim)
        
        # Precomputar RoPE
        head_dim = config.dim // config.n_heads
        t = torch.arange(config.max_seq_len)
        freqs = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        freqs = torch.outer(t, freqs).float()
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.layers = nn.ModuleList([
            PrismaBlock(config, self.freqs_cis, lazy=lazy) 
            for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = None

    def forward(self, tokens, kv_caches=None, offset=0):
        h = self.tok_embeddings(tokens)
        if kv_caches is None: kv_caches = [None] * len(self.layers)
            
        for i, layer in enumerate(self.layers):
            h, kv_caches[i] = layer(h, kv_cache=kv_caches[i], offset=offset)
            
        h = self.norm(h)
        logits = self.tok_embeddings.linear(h)
        return logits, kv_caches

class PrismaBlock(nn.Module):
    def __init__(self, config, freqs_cis, lazy=True):
        super().__init__()
        from attention import Attention
        from layers import FeedForward
        self.attention = Attention(config, freqs_cis, lazy=lazy)
        self.feed_forward = FeedForward(config.dim, config.hidden_dim, quant_mode=config.quant_mode, lazy=lazy)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, kv_cache=None, offset=0):
        h, new_kv = self.attention(self.attention_norm(x), kv_cache=kv_cache, offset=offset)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv
