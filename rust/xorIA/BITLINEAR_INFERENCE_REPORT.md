# BitLinear Inference — Especificación de Mejoras y Kernels

## 1. Resumen de Optimizaciones Aplicadas

### 1.1 forward_inference: Eliminación de Overhead de Burn Tensors

**Antes:** `quantize_activations_8bit` (Burn tensor STE) → `forward_raw` (f32 matmul)
**Ahora:** `quantize_to_i8` (pure CPU) → `forward_raw_i8` (kernel i8) → dequant `gamma/127` → `Tensor::from_data`

**Mejora:**
- Elimina la construcción intermedia de tensors Burn (`.into_data()`, `.reshape()`, STE dequant)
- Quantize + dequant en i8 puro: solo escalar f32 → i8 con absmax, y luego `gamma/127` en el resultado
- Un solo allocation al final: `Tensor::from_data` con los datos crudos

**Estimación de mejora:** ~10-15% reducción de overhead por token en inferencia (eliminación de 2-3 operaciones Burn por cada forward de capa)

---

### 1.2 KV Cache Pre-alloc (slice_assign + narrow)

**Antes:** `Tensor::cat(vec![prev.cached_k, k_new], 1)` — copia TODO el cache + token nuevo cada step
**Ahora:** Pre-allocado a `MAX_CACHE_LEN=256`, `slice_assign` para append, `narrow` para vista

**Mecanismo:**
```
KVCache {
    cached_k: Tensor::zeros([1, 256, num_kv_groups, head_dim]),  // pre-alloc
    cached_v: Tensor::zeros([1, 256, num_kv_groups, head_dim]),
    current_len: usize,  // tracking de uso real
}
```

- **append():** `slice_assign([0..1, start..end, 0..groups, 0..dim], k_new)` — escribe en posición fija, sin copiar
- **view():** `narrow(1, 0, current_len)` — vista zero-copy del contenido válido
- **keep_last(n):** reconstruye buffer copiando solo los últimos `n` tokens, resetea `current_len = n`

**Mejora:**
| Operación | Antes (Tensor::cat) | Ahora (Pre-alloc) |
|-----------|--------------------|--------------------|
| Append 1 token | O(n) copia | O(1) slice_assign |
| Vista | O(n) copia | O(1) narrow |
| Memoria | Crece indefinidamente | Máximo 256 * 4KB ≈ 1MB por layer |

**Estimación de mejora:** ~30-50% reducción de tiempo en la porción de KV cache por token, especialmente significativo a medida que crece la secuencia

---

### 1.3 Trim Rule: >200 → keep_last(70)

**Antes:** `>= 255` → `keep_last(160)` — esperaba a que el cache estuviera casi lleno
**Ahora:** `> 200` → `keep_last(70)` — trim más agresivo, más frecuente

**Mejora:**
- Menos picos de memoria (el cache nunca supera 200 por mucho)
- Menos operaciones de trim (el keep es más barato: 70 vs 160 tokens a copiar)
- Offset del session se resetea a 70, manteniendo coherencia con el cache reducido

---

## 2. Kernels Disponibles en BitLinear

### 2.1 I2SKernel (Integer to Signed)

**Descripción:** Matmul estándar con quantización de pesos a i8 y activations a i8, resultado f32.

**Algoritmo:**
1. **Pack weights:** Cada 4 pesos i8 se empaquetan en un `u32` (2 bits por peso)
2. **Quantize activations:** Token-level absmax → escalar f32, luego `f32 → i8`
3. **Matmul i8×i8→i32:** Multiplica i8 * i8, acumula en i32 por grupo
4. **Dequant:** `resultado_f32 = resultado_i32 * alpha_w * alpha_x`

**Peso por activation:** Cada peso = 0.5 bits (4 pesos por byte)

**Óptimo para:** Matrices grandes (≥512×512), head layer (16000×512)

---

### 2.2 I2STile16Kernel (Integer to Signed + Tile 16×16)

**Descripción:** Misma quantización que I2S, pero con estrategia de multiplicación por bloques de 16×16 para mejor uso de cache.

**Algoritmo:**
- Misma quantización y packing que I2S
- Multiplicación por tiles de 16×16 en lugar de fila-completa
- Reduce cache misses en matrices grandes

**Peso por activation:** Igual que I2S (0.5 bits por peso)

**Óptimo para:** Hidden layers del transformer (512×512), mejor balance memoria/cache

---

### 2.3 TL1Kernel (Ternary Layer 1)

**Descripción:** Matmul con pesos ternarios (-1, 0, +1). Cada peso = ~1.58 bits teóricos.

**Algoritmo:**
1. **Pack weights:** 4 pesos ternarios empaquetados en 1 byte (2 bits cada uno)
2. **Quantize activations:** Token-level absmax → i8
3. **Matmul ternario:** Solo sumas/restas (sin multiplicación), resultado i32
4. **Dequant:** `resultado_f32 = resultado_i32 * alpha_w * alpha_x`

**Peso por activation:** 0.5 bits por peso (empaquetado idéntico a I2S)

**Óptimo para:** Experimentación, entrenamiento con straight-through estimator (STE)

---

### 2.4 TL2Kernel (Ternary Layer 2)

**Descripción:** Variante de TL1 con optimización adicional de packing para densidad extra.

**Óptimo para:** Casos donde se necesita máxima compresión

---

### 2.5 Comparativa de Kernels

| Kernel | Precision | Memoria/Peso | Velocidad (relativa) | Uso actual |
|--------|-----------|-------------|---------------------|------------|
| **I2SKernel** | i8 quant | 0.5 bits | 1.0x (baseline) | Head (16000×512) |
| **I2STile16Kernel** | i8 quant | 0.5 bits | ~1.1x (tile bonus) | Layers (512×512) |
| **TL1Kernel** | Ternary | 0.5 bits | ~1.2x (sin mul) | Training (STE) |
| **TL2Kernel** | Ternary | ~0.4 bits | ~1.1x | Compresión extrema |
| **AVX2 (experimental)** | i8 quant | 0.5 bits | 0.3-1.5x | Solo matrices pequeñas |
| **f32 path** | f32 | 32 bits | 0.8x (no quant) | Baseline / debug |

### 2.6 Configuración Actual del Modelo

```rust
// En build_inference_state:
layer_kernel: KernelKind::Tile16   // Para las 6 capas transformer (512×512)
head_kernel:  KernelKind::I2S      // Para la head de output (16000×512)
```

**Razón:** Tile16 tiene mejor locality en matrices cuadradas. I2S es más rápido en matrices anchas (head).

---

## 3. Impacto Estimado en Velocidad

| Métrica | Antes | Después (estimado) |
|---------|-------|--------------------|
| tok/s inferencia | ~16-22 | ~25-35 |
| Overhead por token | ~45ms | ~30ms |
| KV cache alloc por token | O(n) copia | O(1) assign |
| Memoria cache (6 layers) | ~6MB creciente | ~6MB fija (256 tokens) |

**La mejora principal es la eliminación del overhead de Burn tensors en forward_inference y la pre-asignación del KV cache. El kernel i8 demuestra equivalencia matemática con f32 (tests 5/5 pass).**
