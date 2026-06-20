# Benchmark Report — KV Cache & RoPE Optimization

## 1. KV Cache: 3 Estrategias

Config: `max_cache=4096, head_dim=128, groups=8, 24 layers, Flex<f32>`

### seq_per_call=1 (inference token a token, 4096 appends)

| Estrategia | Append (4096 calls) | view24x (full) |
|---|---|---|
| **clone+slice_assign** (actual) | **59 602 ms** | **0.07 ms** |
| **Vec+cat** | **25.6 ms** | 549 ms |
| bucket64+take | 1 098 ms | 357 ms |

### seq_per_call=16 (prefill, 256 appends)

| Estrategia | Append (256 calls) | view24x (full) |
|---|---|---|
| clone+slice_assign | 3 633 ms | 0.04 ms |
| **Vec+cat** | **20.0 ms** | 381 ms |
| bucket64+take | 60.8 ms | 369 ms |

### Análisis

- **clone+slice_assign** hace deep-copy del buffer completo [1,4096,8,128] = 16 MB en **cada append**. Para 4096 tokens: 64 GB copiados por capa. Catastrófico.
- **Vec+cat** append es O(1) (push a Vec). View hace cat de N tensores = O(N). A max context: ~23 ms por view.
- **bucket64+take** mejora clone pero sigue pagando slice_assign por bucket.

**Ganador**: Vec+cat (ya implementado en model.rs como cumulative cat).

## 2. RoPE: 4 Estrategias

Config: `batch=1, nheads=8, nkv=8, head_dim=128, Flex<f32>`

### seq_len=1 (inference, 4096 iteraciones)

| Estrategia | ms/llamada | vs baseline |
|---|---|---|
| recompute theta (sin caché) | 0.048 ms | 1.0x |
| narrow+chunk (OLD cached) | 0.039 ms | 0.81x |
| narrow+slice (NEW cached) | 0.036 ms | 0.75x |
| **fused raw as_slice** | **0.017 ms** | **0.35x** |

### seq_len=128 (prefill, 32 iteraciones)

| Estrategia | ms/llamada | vs baseline |
|---|---|---|
| recompute theta (sin caché) | 10.1 ms | 1.0x |
| narrow+chunk (OLD cached) | 9.5 ms | 0.94x |
| **narrow+slice (NEW cached)** | **2036 ms** | **ROTO** |
| **fused raw as_slice** | **4.4 ms** | **0.44x** |

### Análisis

- **narrow+reshape** es necesario para broadcasting correcto. Sin reshape, la dimensión 0 del cache (seq) queda desalineada con batch vs seq de q/k, causando broadcasting masivo (128x más elementos → 2036 ms).
- El reshape trabaja sobre un slice pequeño (seq_len × dim/2 = 128×64 = 32 KB), no sobre el cache completo. Su costo es despreciable.
- **fused kernel** (raw as_slice + loop nativo) es 2.3x más rápido que cualquier versión con Tensores Burn para seq=1, y 2.1x para seq=128.

## 3. Recomendaciones

### KV Cache — APLICADA
- **Usar Vec+cat cumulativo** (ya implementado): append = cat(base, new) con Option<Take>, view = clone(base).
- Elimina pre-asignación y deep-copy en cada append.
- keep_last = narrow sobre base (O(1), sin copia).

### RoPE — POR IMPLEMENTAR
- **Opción A (Burn puro, recomendada)**: Mantener el enfoque actual con narrow+reshape en el cached RoPE. La caché 4D [max_seq,1,1,dim/2] evita el reshape en precompute pero el narrow+reshape en caliente sigue siendo necesario para broadcasting correcto. Su costo es ~0.036 ms por token.
- **Opción B (fused kernel, máximo rendimiento)**: Implementar `apply_rope_fused` que extrae q/k como `as_slice::<f32>()`, aplica rotación in-place con loop nativo, y reconstruye Tensor. 2.3x más rápido. Recomendado si el cuello de botella está en RoPE.

### Próximos pasos
1. Verificar que la nueva KV Cache cumulativa compile y funcione.
2. Decidir si implementar fused RoPE o mantener Burn puro.
3. Medir tok/s real post-optimizaciones (esperado: >30 tok/s para contextos medianos).
