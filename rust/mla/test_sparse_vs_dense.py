"""Compara logits sparse vs denso (con attn_mask en lugar de padding)."""
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

def _sparse_mask(self, q, k, v):
    B, NH, S_q, HD = q.shape; NK = self.num_kv_groups; HPG = NH // NK; BSZ = self.block_size
    _, _, S_kv, _ = k.shape; NB = max(1, (S_kv+BSZ-1)//BSZ); K = min(self.num_selected_blocks, NB)
    if NB <= K:
        s = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(HD)
        if self.causal and S_q > 1: s = s + torch.triu(torch.full((S_q,S_kv),float("-inf"),device=q.device),diagonal=S_kv-S_q+1).unsqueeze(0).unsqueeze(0)
        return torch.matmul(F.softmax(s,dim=-1), v)
    pad = NB*BSZ - S_kv; last_real = S_kv - (NB-1)*BSZ
    kp = F.pad(k, (0,0,0,pad)); vp = F.pad(v, (0,0,0,pad))
    knk = kp[:,::HPG]; vnk = vp[:,::HPG]
    k_b = knk.reshape(B,NK,NB,BSZ,HD); v_b = vnk.reshape(B,NK,NB,BSZ,HD)
    k_comp = k_b.mean(dim=3)
    if last_real < BSZ:
        k_comp[:,:,-1,:] = k_b[:,:,-1,:last_real,:].sum(dim=2) / last_real
    q_g = q.reshape(B,NK,HPG,S_q,HD).mean(dim=2)
    scores = (q_g @ k_comp.transpose(-2,-1)) * (HD**-0.5)
    if self.causal and S_q > 1:
        scores.masked_fill_((torch.arange(S_q,device=q.device)[:,None] < (torch.arange(NB,device=q.device)*BSZ)[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    _, topk = scores.topk(K, dim=-1)
    topk_nh = topk.repeat_interleave(HPG, dim=1)
    k_b_nh = k_b.repeat_interleave(HPG, dim=1); v_b_nh = v_b.repeat_interleave(HPG, dim=1)
    idx_b = torch.arange(B,device=q.device)[:,None,None,None]; idx_h = torch.arange(NH,device=q.device)[None,:,None,None]
    k_sel = k_b_nh[idx_b,idx_h,topk_nh]; v_sel = v_b_nh[idx_b,idx_h,topk_nh]
    pos = torch.arange(BSZ,device=q.device).view(1,1,1,1,BSZ)
    amask = ((topk_nh.unsqueeze(-1)==NB-1)&(pos>=last_real)).float()*float('-inf')
    qf = q.reshape(B*NH*S_q,1,HD); kf = k_sel.reshape(B*NH*S_q,K*BSZ,HD); vf = v_sel.reshape(B*NH*S_q,K*BSZ,HD)
    return F.scaled_dot_product_attention(qf, kf, vf, attn_mask=amask.reshape(B*NH*S_q,1,K*BSZ)).reshape(B,NH,S_q,HD)
SparseAttentionMio3._sparse_forward = _sparse_mask

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
print(f"  2 tokens (NB<=K): diff = {diff:.10f}")
print(f"  Top-5 dense:  {ld[0,-1].topk(5).indices.tolist()}")
print(f"  Top-5 sparse: {ls[0,-1].topk(5).indices.tolist()}")
print(f"  IDÉNTICOS: {diff < 1e-6}")

prompt2 = "hola mundo que tal estas hoy"
ids2 = torch.tensor([tok.encode(prompt2).ids], dtype=torch.long)
print(f"\nPrompt: '{prompt2}' ({ids2.shape[1]} tokens)")
ld2, _ = fwd_dense(ids2, 0, [None]*model.num_layers)
ls2, _ = fwd_sparse(ids2, 0, [None]*model.num_layers)
diff2 = (ld2 - ls2).abs().max().item()
print(f"  7 tokens (NB<=K): diff = {diff2:.10f}")
print(f"  Top-5 dense:  {ld2[0,-1].topk(5).indices.tolist()}")
print(f"  Top-5 sparse: {ls2[0,-1].topk(5).indices.tolist()}")
print(f"  IDÉNTICOS: {diff2 < 1e-6}")
