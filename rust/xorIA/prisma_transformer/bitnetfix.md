# BitNetFix: Comparación Detallada — prisma_transformer (Python) vs llama.cpp BitNet (C++)

Comparación línea por línea entre la implementación en `prisma_transformer/` y el código real de `models/bitnet.cpp` + `ggml-cpu/quants.c` del repositorio `prism-ml-llama.cpp`.

---

## 1. Arquitectura del Bloque Transformer

### llama.cpp (`models/bitnet.cpp`, líneas 21-129)
```
Para cada capa:
  1. attn_norm (RMSNorm)
  2. Qcur = build_lora_mm(wq, cur, wq_s)    ← MatMul ternario con escala separada
  3. Kcur = build_lora_mm(wk, cur, wk_s)
  4. Vcur = build_lora_mm(wv, cur, wv_s)
  5. RoPE(Q), RoPE(K)
  6. build_attn(Q, K, V)                     ← Atención con KV Cache
  7. attn_sub_norm (RMSNorm)                 ← ⚠️ SUB-NORM EXTRA (BitNet específico)
  8. build_lora_mm(wo, cur, wo_s)
  9. Residual
  10. ffn_norm (RMSNorm)
  11. build_ffn(ffn_up, ffn_gate, SiLU)       ← SwiGLU
  12. ffn_sub_norm (RMSNorm)                  ← ⚠️ SUB-NORM EXTRA (BitNet específico)
  13. build_lora_mm(ffn_down, cur, ffn_down_s)
  14. Residual
```

### Mi Python (`transformer.py` + `attention.py` + `layers.py`)
```
Para cada capa:
  1. attention_norm (RMSNorm)                ✅ Igual
  2. wq = Q1_0_Linear / TQ2_0_Linear        ✅ MatMul ternario (empaquetado en bits)
  3. wk = Q1_0_Linear / TQ2_0_Linear        ✅
  4. wv = Q1_0_Linear / TQ2_0_Linear        ✅
  5. (Sin RoPE)                              ❌ FALTA — No implementamos RoPE
  6. torch.matmul(q, k.T) → softmax → matmul(attn, v)  ✅ Atención con KV Cache
  7. (Sin attn_sub_norm)                     ❌ FALTA — BitNet usa sub-norm después de atención
  8. wo = Q1_0_Linear / TQ2_0_Linear        ✅
  9. Residual                                ✅
  10. ffn_norm (RMSNorm)                     ✅
  11. SwiGLU: silu(w1(x)) * w3(x)            ✅ Igual
  12. (Sin ffn_sub_norm)                      ❌ FALTA — BitNet usa sub-norm después de FFN
  13. w2 = Q1_0_Linear / TQ2_0_Linear        ✅
  14. Residual                                ✅
```

### Diferencias Encontradas en Arquitectura

| Componente | llama.cpp BitNet | Mi Python | Estado |
|---|---|---|---|
| RMSNorm pre-atención | ✅ `attn_norm` | ✅ `attention_norm` | ✅ IGUAL |
| Proyecciones Q/K/V/O | ✅ Ternario con `wq_s` escala | ✅ Ternario con `blocks_d` escala | ✅ IGUAL |
| RoPE | ✅ `ggml_rope_ext` | ❌ No implementado | ❌ FALTA |
| Atención + KV Cache | ✅ `build_attn` + cache | ✅ `torch.matmul` + `torch.cat` | ✅ FUNCIONAL |
| **attn_sub_norm** | ✅ RMSNorm después de atención | ❌ No existe | ❌ FALTA |
| SwiGLU FFN | ✅ `ffn_gate` + `ffn_up` + SiLU | ✅ `w1` + `w3` + SiLU | ✅ IGUAL |
| **ffn_sub_norm** | ✅ RMSNorm después de SwiGLU | ❌ No existe | ❌ FALTA |
| Output head | ✅ Reutiliza `tok_embd` (tied) | ❌ `nn.Linear` separado | ❌ DIFERENTE |

---

## 2. Empaquetado de Pesos (Weight Quantization)

### llama.cpp (`ggml-common.h` + `ggml-cpu/quants.c`)

#### Estructura Q1_0 (Prisma/Khosravipasha — 1 bit)
```c
// ggml-common.h línea 180
#define QK1_0 32
typedef struct {
    ggml_half d;            // escala float16 (2 bytes)
    uint8_t qs[QK1_0 / 8]; // 32 pesos en 4 bytes (1 bit c/u)
} block_q1_0;
```

#### Estructura TQ2_0 (Ternario — 2 bits)
```c
// ggml-common.h línea 283
typedef struct {
    uint8_t qs[QK_K/4]; // 256 pesos en 64 bytes (2 bits c/u)
    ggml_half d;
} block_tq2_0;
```

### Mi Python (`layers.py`)

#### Q1_0_Linear (líneas 29-116)
```python
QK = 32  # Igual que QK1_0
# blocks_d → escala por bloque (equivale a ggml_half d)
# blocks_qs → uint8[QK // 8] = 4 bytes por bloque (IGUAL que qs[QK1_0/8])
```

#### TQ2_0_Linear (líneas 132-218)
```python
QK = 32  # ⚠️ DIFERENTE — llama.cpp usa QK_K=256
# blocks_d → escala por bloque
# blocks_qs → uint8[QK // 4] = 8 bytes por bloque de 32
```

### Comparación de Empaquetado

| Aspecto | llama.cpp | Mi Python | Estado |
|---|---|---|---|
| Q1_0 block size | 32 | 32 | ✅ IGUAL |
| Q1_0 bits/peso | 1 | 1 | ✅ IGUAL |
| Q1_0 escala | `ggml_half` (fp16, 2 bytes) | `float32` (4 bytes) | ⚠️ Más pesado |
| TQ2_0 block size | **256 (QK_K)** | **32** | ❌ DIFERENTE |
| TQ2_0 bits/peso | 2 | 2 | ✅ IGUAL |
| Encoding `{-1,0,+1}` | `(bits & 3) - 1` | `(bits & 3) - 1` | ✅ IDÉNTICO |

---

## 3. Producto Punto (vec_dot) — El Kernel Matemático

### llama.cpp: `ggml_vec_dot_q1_0_q8_0_generic` (quants.c líneas 127-166)
```c
for (int j = 0; j < QK1_0; j++) {
    const int byte_index = j / 8;
    const int bit_offset = j % 8;
    const int xi = ((x[i].qs[byte_index] >> bit_offset) & 1) ? 1 : -1;
    const int yi = y[i].qs[j];
    sumi += xi * yi;
}
sumf += d0 * d1 * sumi;
```

### Mi Python: `Q1_0_Linear.forward` (layers.py líneas 99-109)
```python
for j in range(self.QK):
    byte_index = j // 8
    bit_offset = j % 8
    xi = 1 if ((self.blocks_qs[idx, byte_index].item() >> bit_offset) & 1) else -1
    yi = act_q[j].item()
    sumi += xi * yi
sumf += d0 * d1 * sumi
```

**VEREDICTO: COPIA EXACTA.** Misma lógica bit a bit, mismas máscaras, mismo orden de operaciones.

### llama.cpp: `ggml_vec_dot_tq2_0_q8_K_generic` (quants.c líneas 524-554)
```c
for (size_t l = 0; l < 4; ++l) {
    for (size_t k = 0; k < 32; ++k) {
        sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
    }
}
```

### Mi Python: `TQ2_0_Linear.forward` (layers.py líneas 203-209)
```python
for byte_idx in range(self.QK // 4):
    for l in range(4):
        w = ((self.blocks_qs[idx, byte_idx].item() >> (l * 2)) & 3) - 1
        yi = act_q[byte_idx * 4 + l].item()
        sumi += yi * w
```

**VEREDICTO: COPIA EXACTA.** La extracción de 2 bits con `>> (l*2) & 3) - 1` es idéntica.

---

## 4. Cuantización de Activaciones (INT8)

### llama.cpp (Q8_0)
```c
// Cada bloque de activación se escala a int8:
// d = max(|x|) / 127
// q = round(x / d), clamped to [-127, 127]
```

### Mi Python (layers.py líneas 94-97)
```python
d1_val = act_blk.abs().max().clamp(min=1e-8)
act_q = torch.round(act_blk * (127.0 / d1_val)).clamp(-127, 127).to(torch.int8)
d1 = (d1_val / 127.0).item()
```

**VEREDICTO: IDÉNTICO.** Misma fórmula AbsMax con factor 127.

---

## 5. Embedding

### llama.cpp (`bitnet.cpp` línea 12)
```c
inpL = build_inp_embd(model.tok_embd);
// Esto es un ggml_get_rows() — lookup directo, NO cuantizado
```

### Mi Python (`transformer.py` línea 31)
```python
self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
# Lookup directo en float32
```

**VEREDICTO: IGUAL.** Ambos usan embeddings de alta precisión (no ternarios).

---

## 6. Output Head (lm_head)

### llama.cpp (`bitnet.cpp` líneas 140-142)
```c
// FIXME: do not use model.tok_embd directly, duplicate as model.output
cur = build_lora_mm(model.tok_embd, cur);
// ¡Reutiliza los pesos del embedding! (Tied Weights)
```

### Mi Python (`transformer.py` línea 37)
```python
self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
# Capa separada, NO tied
```

**VEREDICTO: DIFERENTE.** llama.cpp reutiliza el embedding como output head. Mi Python tiene pesos separados.

---

## 7. KV Cache

### llama.cpp (`llama-kv-cache.cpp` líneas 81-198)
```c
// type_k y type_v se pasan como parámetros
// Default: GGML_TYPE_F16 (línea 2905 de llama-context.cpp)
// Para BitNet óptimo: GGML_TYPE_Q8_0
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```

### Mi Python (`attention.py` líneas 33-36)
```python
# Cache se guarda como tensores PyTorch (float32 implícito)
k = torch.cat([k_cache, k], dim=2)
v = torch.cat([v_cache, v], dim=2)
```

### Especificación de Bits del KV Cache

| Componente | llama.cpp (Default) | llama.cpp (BitNet óptimo) | Mi Python |
|---|---|---|---|
| **K Cache** | F16 (16 bits) | Q8_0 (8 bits) | float32 (32 bits) |
| **V Cache** | F16 (16 bits) | Q8_0 (8 bits) | float32 (32 bits) |
| **Peso por token** | 4 bytes/elem | 1.06 bytes/elem | 8 bytes/elem |
| **Operación attn** | `ggml_mul_mat` | `ggml_mul_mat` | `torch.matmul` |

---

## 8. Lo Que FALTA en Mi Python Para Ser BitNet Completo

| # | Componente Faltante | Dónde en llama.cpp | Impacto |
|---|---|---|---|
| 1 | **attn_sub_norm** | `bitnet.cpp` línea 79-81 | RMSNorm entre atención y `wo`. Estabiliza la señal. |
| 2 | **ffn_sub_norm** | `bitnet.cpp` línea 113-115 | RMSNorm entre SwiGLU y `ffn_down`. Estabiliza la señal. |
| 3 | **RoPE** | `bitnet.cpp` línea 59-69 | Embeddings posicionales rotatorios. Sin esto, no hay noción de posición. |
| 4 | **Tied Weights** | `bitnet.cpp` línea 142 | Output head reutiliza embedding. Ahorra parámetros. |
| 5 | **KV Cache int8** | `llama-kv-cache.cpp` línea 197 | Cache debería ser Q8_0, no float32. |
| 6 | **Block size 256** | `ggml-common.h` (TQ2_0) | TQ2_0 usa bloques de 256, no 32. Menos overhead de escalas. |
| 7 | **Escalas fp16** | `ggml-common.h` | Escalas deberían ser `ggml_half` (2 bytes), no `float32` (4 bytes). |

---

## 9. Lo Que ESTÁ BIEN en Mi Python

| # | Componente | Estado |
|---|---|---|
| 1 | Empaquetado Q1_0 (1 bit por peso, 8 pesos/byte) | ✅ IDÉNTICO a `ggml-common.h` |
| 2 | Empaquetado TQ2_0 (2 bits por peso, 4 pesos/byte) | ✅ IDÉNTICO (salvo block size) |
| 3 | Bit-masking `(>> shift) & mask` | ✅ IDÉNTICO a `quants.c` |
| 4 | Activaciones int8 (AbsMax scaling) | ✅ IDÉNTICO a Q8_0 |
| 5 | SwiGLU FFN (gate + up + SiLU) | ✅ IDÉNTICO |
| 6 | RMSNorm pre-atención y pre-FFN | ✅ IDÉNTICO |
| 7 | Embedding en alta precisión | ✅ IDÉNTICO |
| 8 | KV Cache incremental (concat) | ✅ Funcional (pero en f32) |
| 9 | Factory pattern (`make_linear`) | ✅ Limpio y extensible |
