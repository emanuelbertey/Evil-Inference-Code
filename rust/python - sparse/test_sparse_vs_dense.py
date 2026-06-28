"""Compara logits sparse vs denso (sin padding)."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TransformerLM
from sparse import SparseAttentionMio3
from safetensors.torch import load_file
from tokenizers import Tokenizer
from cache_kv import KVCache
from attention import repeat_kv
from rope import apply_rope_partial

device = torch.device("cpu")
_DIR = os.path.dirname(os.path.abspath(__file__))
_RUST = os.path.dirname(_DIR)
tok = Tokenizer.from_file(os.path.join(_RUST, "tokenizer.json"))
vocab = tok.get_vocab_size()

model = TransformerLM(vocab_size=vocab, d_model=768, num_layers=24, num_heads=12, num_kv_groups=4,
    use_swiglu=True, use_x0=True, max_seq_len=4096, use_sparse_attn=True, num_selected_blocks=16,
).to(device).eval()
state = load_file(os.path.join(_RUST, "model_test.safetensors"))
model.load_state_dict(state, strict=False)

# Forward denso
def fwd_dense(input_ids, offset, caches):
    x = model.embedding(input_ids); h = x; x0 = x.clone(); nc = []
    for i, (layer, cache) in enumerate(zip(model.transformer.layers, caches)):
        r = h; hn = layer.attn_norm(h)
        q, kn, vn = layer.attention.qkv(hn)
        q, kn = apply_rope_partial(q, kn, offset, 0.25, layer.attention.rope.inv_freq, layer.attention.rope.cos_cache, layer.attention.rope.sin_cache, layer.attention.head_dim, layer.attention.rope.max_seq_len)
        kf = torch.cat([cache.cached_k, kn], dim=1) if cache is not None else kn
        vf = torch.cat([cache.cached_v, vn], dim=1) if cache is not None else vn
        nc.append(KVCache(kf.clone(), vf.clone()))
        ke = repeat_kv(kf, layer.num_heads, layer.num_kv_groups); ve = repeat_kv(vf, layer.num_heads, layer.num_kv_groups)
        qt = q.transpose(1,2); kt = ke.transpose(1,2); vt = ve.transpose(1,2)
        s = torch.matmul(qt, kt.transpose(-2,-1)) / math.sqrt(layer.head_dim)
        if layer.attn_logit_cap is not None: s = torch.tanh(s / layer.attn_logit_cap) * layer.attn_logit_cap
        if s.shape[2] > 1: s = s + torch.triu(torch.full((s.shape[2], s.shape[3]), float("-inf"), device=h.device), diagonal=s.shape[3]-s.shape[2]+1).unsqueeze(0).unsqueeze(0)
        ao = torch.matmul(F.softmax(s, dim=-1), vt).transpose(1,2)
        h = r + layer.attention.o_proj(ao); r = h; hn = layer.ffn_norm(h); hf = layer.ffn(hn); h = r + hf
        if model.x0_lambdas is not None: h = h + model.x0_lambdas[0,i] * x0
    return model.head(model.transformer.final_norm(h)), nc

# Spars forward sin padding
def _sparse_nowpad(self, q, k, v):
    B, NH, S_q, HD = q.shape; NK = self.num_kv_groups; HPG = NH // NK; BSZ = self.block_size
    _, _, S_kv, _ = k.shape; NB = max(1, (S_kv+BSZ-1)//BSZ); K = min(self.num_selected_blocks, NB)
    k_nk = k[:,:NK]; v_nk = v[:,:NK]
    if NB <= K:
        ke = repeat_kv(k_nk, NH, NK); ve = repeat_kv(v_nk, NH, NK)
        s = torch.matmul(q, ke.transpose(-2,-1)) / math.sqrt(HD)
        if self.causal and S_q > 1: s = s + torch.triu(torch.full((S_q,S_kv),float("-inf"),device=q.device),diagonal=S_kv-S_q+1).unsqueeze(0).unsqueeze(0)
        return torch.matmul(F.softmax(s,dim=-1), ve)
    # NB > K: bloques sin padding
    k_blocks = [k_nk[:,:,blk*BSZ:min((blk+1)*BSZ,S_kv)] for blk in range(NB)]
    v_blocks = [v_nk[:,:,blk*BSZ:min((blk+1)*BSZ,S_kv)] for blk in range(NB)]
    k_comp = torch.stack([b.mean(dim=2) for b in k_blocks], dim=2)
    q_g = q.reshape(B,NK,HPG,S_q,HD).mean(dim=2)
    scores = (q_g @ k_comp.transpose(-2,-1)) * (HD**-0.5)
    if self.causal and S_q > 1:
        scores.masked_fill_((torch.arange(S_q,device=q.device)[:,None] < (torch.arange(NB,device=q.device)*BSZ)[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    always = [i for i in (0,NB-1) if i < NB]; n_a = len(always)
    s2 = scores.clone(); s2[...,always] = float('-inf')
    _, tr = s2.topk(K-n_a, dim=-1)
    topk = torch.cat([torch.tensor(always,device=q.device).view(1,1,1,-1).expand(B,NK,S_q,-1), tr], dim=-1)
    sel_k, sel_v = [], []
    for kk in range(K):
        blk = topk[0,0,0,kk].item()
        sel_k.append(repeat_kv(k_blocks[blk], NH, NK))
        sel_v.append(repeat_kv(v_blocks[blk], NH, NK))
    ks = torch.cat(sel_k, dim=2); vs = torch.cat(sel_v, dim=2)
    return F.scaled_dot_product_attention(q.reshape(B*NH*S_q,1,HD), ks.reshape(B*NH*S_q,ks.shape[2],HD), vs.reshape(B*NH*S_q,vs.shape[2],HD), is_causal=False).reshape(B,NH,S_q,HD)
SparseAttentionMio3._sparse_forward = _sparse_nowpad

def fwd_sparse(input_ids, offset, caches):
    x = model.embedding(input_ids); h = x; x0 = x.clone(); nc = []
    for i, (layer, cache) in enumerate(zip(model.transformer.layers, caches)):
        r = h; hn = layer.attn_norm(h)
        q, kn, vn = layer.attention.qkv(hn)
        q, kn = apply_rope_partial(q, kn, offset, 0.25, layer.attention.rope.inv_freq, layer.attention.rope.cos_cache, layer.attention.rope.sin_cache, layer.attention.head_dim, layer.attention.rope.max_seq_len)
        kf = torch.cat([cache.cached_k, kn], dim=1) if cache is not None else kn
        vf = torch.cat([cache.cached_v, vn], dim=1) if cache is not None else vn
        nc.append(KVCache(kf.clone(), vf.clone()))
        ke = repeat_kv(kf, layer.num_heads, layer.num_kv_groups); ve = repeat_kv(vf, layer.num_heads, layer.num_kv_groups)
        qt = q.transpose(1,2); kt = ke.transpose(1,2); vt = ve.transpose(1,2)
        ao = layer.attention._sparse_forward(qt, kt, vt).transpose(1,2)
        h = r + layer.attention.o_proj(ao); r = h; hn = layer.ffn_norm(h); hf = layer.ffn(hn); h = r + hf
        if model.x0_lambdas is not None: h = h + model.x0_lambdas[0,i] * x0
    return model.head(model.transformer.final_norm(h)), nc

prompt = "desde las"
ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
print(f"Prompt: '{prompt}' ({ids.shape[1]} tokens)")

ld, _ = fwd_dense(ids, 0, [None]*model.num_layers)
ls, _ = fwd_sparse(ids, 0, [None]*model.num_layers)
diff = (ld - ls).abs().max().item()
print(f"  1 bloque, NB<=K: diff = {diff:.10f}")
print(f"  Top-5 dense:  {ld[0,-1].topk(5).indices.tolist()}")
print(f"  Top-5 sparse: {ls[0,-1].topk(5).indices.tolist()}")
print(f"  IDÉNTICOS: {diff < 1e-6}")

# Test con cache > 2048 para sparse real
caches_d = [None]*model.num_layers; caches_s = [None]*model.num_layers
pretend = torch.randint(0, 100, (1, 2049), dtype=torch.long)
ld2, caches_d = fwd_dense(pretend, 0, caches_d)
ls2, caches_s = fwd_sparse(pretend, 0, caches_s)
print(f"\n  2049 tokens (17 bloques, NB>K): diff prefill = {(ld2-ls2).abs().max().item():.10f}")
nt = torch.tensor([[123]], dtype=torch.long)
ld3, _ = fwd_dense(nt, 2049, caches_d)
ls3, _ = fwd_sparse(nt, 2049, caches_s)
diff3 = (ld3 - ls3).abs().max().item()
print(f"  1 token cache 2049: diff = {diff3:.10f}")
print(f"  Top-5 dense:  {ld3[0,-1].topk(5).indices.tolist()}")
print(f"  Top-5 sparse: {ls3[0,-1].topk(5).indices.tolist()}")
