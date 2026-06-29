# Cambios MLA vs Standard GQA (python/)

## Atención

| Componente | `python/` (Standard GQA) | `mla/` (MLA + GQA) |
|---|---|---|
| Q | `W_q`: `d_model → d_model` | `W_down(d_model → d_c1+NK*d_c+d_rot) + W_up_q(d_c1 → d_model+NH*d_rot)` |
| K | `W_k`: `d_model → NK*hd` | `W_down + W_up_kv(NK*d_c → 2*NK*hd)` |
| V | `W_v`: `d_model → NK*hd` | Compartido con K via `W_up_kv` |
| RoPE | `head_dim` completo | Solo `d_rotate` parcial |
| qk_dim | `head_dim` | `head_dim + d_rotate` |
| Params Q | `d_model²` | `d_model*(d_c1+NK*d_c+d_rot) + d_c1*(d_model+NH*d_rot)` |
| Params K+V | `2*d_model*NK*hd` | `(d_model*NK*d_c) + (NK*d_c)*(2*NK*hd)` |

## Latentes (MLA)

- `d_c = max(32, d_model//6)` — KV compression dim
- `d_c1 = max(32, d_model//6)` — Q compression dim
- `d_rotate = max(16, d_model//12)` — RoPE dim

## Archivos

| Archivo | `python/` | `mla/` |
|---|---|---|
| `attention.py` | `Attention` (GQA + RoPE) | — |
| `mla_attention.py` | — | `MultiHeadLatentAttentionGQA` + `QKVProjectionMLA` + `OutputProjectionMLA` |
| `block.py` | `TransformerLayer` usa `Attention` | `TransformerLayer` con `use_mla=True` crea `MultiHeadLatentAttentionGQA` |
| `model.py` | `TransformerLM` standard forward | `forward_train_partial_rope` con branch MLA |
| `rope.py` | `RoPE`, `apply_rope_partial` | Sin cambios (reutilizado) |
| `cache_kv.py` | `KVCache` | Sin cambios |
| `train.py` | Original con HF Hub + streaming | Copia con `revision="gens0mla"`, MLA por defecto |

## Forward `MultiHeadLatentAttentionGQA`

```
x → W_down → C_Q (d_c1) + C_KV (NK*d_c) + K_rotate (d_rot)
              ↓                  ↓
          W_up_q              W_up_kv
              ↓                  ↓
         Q_state (d_model)    K_state (NK*hd)
         Q_rotate (NH*d_rot)  V_state (NK*hd)

Q = cat(Q_state, Q_rotate).reshape(NH, hd+d_rot)
K = cat(K_state, expand(K_rotate, NK)).reshape(NK, hd+d_rot)
V = V_state.reshape(NK, hd)

K = repeat_kv(K, NH, NK)  # expand NK → NH
V = repeat_kv(V, NH, NK)

scores = Q @ K.T / sqrt(hd+d_rot)
out = softmax(scores, causal) @ V
out = o_proj(out.reshape(NH*hd))  # solo head_dim, no qk_dim
```

## OutputProjection

- `python/`: `nn.Linear(NH*hd, d_model)` (qk_dim = hd)
- `mla/`: **`nn.Linear(NH*hd, d_model)`** (usa `head_dim`, **NO** `qk_dim`)

Esto es porque `scaled_dot_product_attention(Q,K,V)` produce salida con la última dim de `V` = `head_dim`, no `qk_dim`.

## Weight tying

`head.weight = embedding.weight` — sharing entre embedding y lm_head.

## Train

- `test_mode` activo cuando se pasa `input.txt` como argumento
- En test mode: no pide HF token, no sube checkpoints, single pass
- `mla_block_size=128` pasado al constructor
- Sin flag `use_mla` — en `mla/` es el default
