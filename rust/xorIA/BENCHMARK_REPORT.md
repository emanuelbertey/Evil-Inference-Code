# Benchmark Report — KV Cache & RoPE Optimization

## 1. KV Cache: 3 Estrategias

Config: `max_cache=4096, head_dim=128, groups=8, 24 layers, Flex<f32>` (CPU)

### seq_per_call=1 (inference token a token, 4096 appends)

| Estrategia | Append (4096 calls) | view24x (full) |
|---|---|---|
| **clone+slice_assign** (original) | 215 330 ms | **0.07 ms** |
| **Vec+cat** (actual) | **23.5 ms** | 512 ms |
| bucket64+take | 772 ms | 304 ms |

### seq_per_call=16 (prefill, 256 appends)

| Estrategia | Append (256 calls) | view24x (full) |
|---|---|---|
| clone+slice_assign | 12 991 ms | **0.04 ms** |
| **Vec+cat** (actual) | **16.0 ms** | 315 ms |
| bucket64+take | 57.7 ms | 378 ms |

### Análisis

- **clone+slice_assign** hace deep-copy del buffer completo [1,4096,8,128] = 16 MB en **cada append**. Para 4096 tokens: 64 GB copiados por capa. Catastrófico.
- **Vec+cat** append es O(1) (push a Vec). View hace cat de N tensores (cada append crea 1 chunk → 4096 chunks a max context). ~512 ms por 24 views = ~21 ms/view.
- **bucket64+take** pre-asigna buckets de 64 tokens, escribe con slice_assign. Append más lento que Vec+cat porque slice_assign copia el bucket completo (256 KB) en cada append. View más rápido porque son solo 64 buckets.

**Implementado**: Vec+cat (k_chunks: Vec<Tensor>). Es el mejor balance simplicidad/performance.

### Pendiente: chunked grande

Si el view de Vec+cat se vuelve cuello de botella a contextos muy grandes, se puede migrar a chunked con bucket_size=256. Esto reduce chunks de 4096 a 16, y view haría cat de solo 16 tensores. Append se mantiene O(1) con bucket pre-asignado.

## 2. RoPE: Fused Kernel (APLICADO)

Config: `batch=1, nheads=8, nkv=8, head_dim=128, Flex<f32>` (CPU)

### seq_len=1 (inference, 4096 iteraciones)

| Estrategia | ms/llamada | vs baseline |
|---|---|---|
| recompute theta (sin caché) | 0.041 ms | 1.0x |
| narrow+chunk (caché 2D) | 0.038 ms | 0.93x |
| narrow+slice (caché 4D, roto sin reshape) | 0.037 ms | 0.90x |
| **fused raw as_slice** (APLICADO) | **0.014 ms** | **0.34x** |

### seq_len=128 (prefill, 32 iteraciones)

| Estrategia | ms/llamada | vs baseline |
|---|---|---|
| recompute theta (sin caché) | 9.74 ms | 1.0x |
| narrow+chunk (caché 2D) | 9.17 ms | 0.94x |
| narrow+slice (caché 4D, **roto**) | 2002 ms | 205x |
| **fused raw as_slice** (APLICADO) | **3.57 ms** | **0.37x** |

### Análisis

**¿Por qué la caché de RoPE no ayuda apenas?**
La `narrow(0, offset, seq_len)` sobre la caché 4D copia `seq_len × head_dim/2` elementos:
- seq_len=1: 1 × 64 = 64 f32 = 256 bytes. Despreciable.
- seq_len=128: 128 × 64 = 8192 f32 = 32 KB. Despreciable.
- seq_len=4096: 4096 × 64 = 256K f32 = 1 MB. _Ahí empieza a doler._
- seq_len=16384: 16384 × 64 = 1M f32 = 4 MB. **Copia más cara que trig.**

Además, la caché 2D y 4D requieren reshape para broadcasting correcto, lo que añade overhead. El "ahorro" de no recalcular theta es marginal porque `cos/sin` sobre ~8K elementos (seq=128) es más rápido que copiar y reshapedear.

**Fused kernel**: elimina toda copia y reshape. Extrae datos crudos, opera in-place con trig directa, reconstruye el tensor. Es ~3x más rápido siempre, independiente del contexto.

### Estado actual en model.rs

- `apply_rope_fused` reemplaza a `apply_rope_cached`, `apply_rope`, `RoPECache`, `precompute_rope_cache`
- Sin parámetro `cos_sin` en ninguna función de forward
- Firma limpia: `fn forward(&self, x, offset)` sin cadenas de referencias heredadas

## 3. Recomendaciones

### KV Cache — APLICADO (Vec+cat)

| Antes | Después |
|---|---|
| `Option<Tensor>` base + cat en append | `Vec<Tensor>` chunks + cat en view |
| O(t) en append (copia incremental) | O(1) en append, O(n_chunks) en view |
| free view (Arc clone) | cat on view (1 copia total) |

### RoPE — APLICADO (fused kernel)

| Antes | Después |
|---|---|
| `RoPECache` + `precompute_rope_cache` + `cos_sin` en cadena | `apply_rope_fused(q, k, offset)` |
| narrow + slice + cat (3 copias por llamada) | raw as_slice + loop nativo (0 copies, 1 reconstrucción al final) |
| 0.041 ms/token (recompute) | 0.014 ms/token (3x más rápido) |
| 9.74 ms/prefill-128 | 3.57 ms/prefill-128 (2.7x más rápido) |

### Próximos pasos
1. Verificar tok/s con xoria post-optimizaciones.
2. Si el view de Vec+cat es cuello de botella a contextos extremos (>16K), migrar a chunked con buckets grandes (256+).
