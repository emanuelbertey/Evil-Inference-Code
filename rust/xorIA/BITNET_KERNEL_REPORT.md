# BitNet Kernel Architecture — Informe Técnico

## 1. compute_row_i8: Por qué retorna i32 y no f32

### Diseño original (ANTES)

```rust
fn compute_row_i8(
    x_i8: &[i8], x_off: usize,
    packed_w: &[u32], w_row: usize,
    scales: &[f32], in_features: usize,  // ← scales parámetro
) -> i32 {
    let mut sum = 0i32;
    // ... unpacking ternario × i8 → suma en i32 ...
    let _g = (w_row / GROUP_SIZE).min(scales.len() - 1);  // ← código muerto
    sum
}
```

**Problemas:**
- `scales` se pasaba como parámetro pero **nunca se usaba** dentro de la función
- `_g` se calculaba pero **nunca se leía** — era variable muerta
- El escalado por `scales[g]` siempre se hacía en el **caller**, no aquí

### Diseño actual (DESPUÉS)

```rust
fn compute_row_i8(
    x_i8: &[i8], x_off: usize,
    packed_w: &[u32], w_row: usize,
    in_features: usize,              // ← scales eliminado
) -> i32 {
    let mut sum = 0i32;
    // ... unpacking ternario × i8 → suma en i32 ...
    sum                              // ← sin _g, retorno directo
}
```

### Por qué retorna i32 (no f32)

La función **solo** calcula la suma de productos punto ternario × i8:

```
sum += (bit_ternario - 1) × x_i8
```

Donde `bit_ternario` ∈ {0b00, 0b01, 0b10} → (bit-1) ∈ {-1, 0, +1}

Esto es **entero puro**. No hay multiplicación de punto flotante. Retorna i32 porque:
1. **Performance**: i32 es más rápido que f32 en CPU (sin conversión)
2. **Precisión**: acumulación exacta sin rounding error
3. **Separación de responsabilidades**: esta función solo suma, el escalado es otra función

### Dónde ocurre el escalado (en forward_raw_i8)

```rust
// caller 1: camino secuencial (total < 16)
let raw = Self::compute_row_i8(x_i8, x_off, packed_w, o * in_features, in_features);
let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
*out = raw as f32 * scales[g];    // ← i32 → f32 + escalado por grupo

// caller 2: camino multi-threaded
let raw = Self::compute_row_i8(x_i8, b * in_features, packed_w, o * in_features, in_features);
let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
*chunk = raw as f32 * scales[g];   // ← misma lógica
```

**Flujo completo:**
```
packed_w[u32] → unpack ternario{-1,0,+1} → × x_i8[i8] → sum i32 → × scales[g] → f32
                  (compute_row_i8)                           (forward_raw_i8)
```

---

## 2. Estructura de los Kernels

### KernelKind: Estrategia de multiplicación

| Kernel | Estrategia | Uso óptimo |
|--------|-----------|------------|
| `I2SKernel` | Fila completa (16× unrolled) | Head (16000×512) |
| `I2STile16Kernel` | Bloques 16×16 | Layers (512×512) |

Ambos usan la **misma** representación de pesos: `packed_w: Vec<u32>` (16 pesos ternarios por u32).

### Flujo de datos en inferencia

```
┌─────────────────────────────────────────────────────────┐
│  forward_inference (layer.rs)                           │
│                                                         │
│  1. RMSNorm(x) → x_norm          [f32, pure CPU]       │
│  2. quantize_to_i8(x_norm)       [f32→i8, pure CPU]    │
│     └─ returns (x_i8: Vec<i8>, gammas: Vec<f32>)       │
│  3. forward_raw_i8(x_i8)         [i8 matmul, kernel]   │
│     └─ compute_row_i8 per row    [i32 sum]              │
│     └─ × scales[g] per group     [f32 dequant]         │
│  4. × (gammas[t] / 127)          [f32 rescale]         │
│  5. Tensor::from_data(result)    [f32 → Burn tensor]   │
└─────────────────────────────────────────────────────────┘
```

### Ternary encoding (pack_weights)

```rust
// 4 pesos empaquetados en 1 byte (2 bits cada uno)
let bits = if w < -0.5 { 0b00 }      // → -1
           else if w > 0.5 { 0b10 }  // → +1
           else { 0b01 };            // →  0

// Decode: (bits as i32) - 1
// 0b00 = 0 → 0-1 = -1 ✓
// 0b01 = 1 → 1-1 =  0 ✓
// 0b10 = 2 → 2-1 = +1 ✓
```

---

## 3. Cambios de Seguridad — KV Cache

### Bug 1: Buffer Overflow en append

**ANTES:** Sin control. Si `current_len + add_len > 256`, `slice_assign` escribía fuera de bounds.

**DESPUÉS:**
```rust
pub fn append(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) {
    let end = self.current_len + add_len;
    if end > MAX_CACHE_LEN {
        let safe_keep = MAX_CACHE_LEN.saturating_sub(add_len);
        self.keep_last(safe_keep);  // auto-trim antes de escribir
    }
    // ... slice_assign seguro ...
}
```

### Bug 2: keep_last creaba tensor nuevo innecesariamente

**ANTES:** `Tensor::zeros([1, max_cap, groups, dim])` + `slice_assign` → 2 allocs por tensor.

**DESPUÉS:**
```rust
pub fn keep_last(&mut self, keep: usize) {
    let k = self.cached_k.clone().narrow(1, start, keep);
    self.cached_k = self.cached_k.clone().slice_assign(
        [0..1, 0..keep, 0..groups, 0..dim], k
    );
    // ...
}
```
Reutiliza el mismo tensor como base. Datos stale en `keep..current_len` son invisibles porque `view()` usa `narrow(1, 0, current_len)`.

---

## 4. Resumen de Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `kernel.rs:629` | `compute_row_i8`: eliminado `scales` param y `_g` muerto |
| `kernel.rs:718` | Caller actualizado: sin `scales` en call |
| `kernel.rs:746` | Caller actualizado: sin `scales` en call |
| `model.rs:241` | `append()`: bounds check contra overflow |
| `model.rs:270` | `keep_last()`: eliminado `Tensor::zeros` innecesario |
| `bitnet_export.rs` | Nuevo módulo: export/load formato .bitnet |
| `main.rs:9` | `mod bitnet_export` |
| `main.rs:156` | CLI `--export` y `--compare` |
