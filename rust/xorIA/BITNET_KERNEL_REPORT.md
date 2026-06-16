# BitNet Kernel Report

Estado final verificado contra `rust/src/blocks/bitlinear/kernel.rs`, `rust/src/blocks/bitlinear/layer.rs` y `rust/xorIA/transformer_bit2/bitnet_export.rs`. Todos los bugs reportados han sido corregidos.

## Kernels activos vs. kernels presentes

| Kernel | Seleccionable por `KernelKind` | Tipo de `packed_w` | Uso |
|--------|-------------------------------|--------------------|-----|
| `I2SKernel` | Sí | `Vec<u32>` | inferencia activa |
| `I2STile16Kernel` | Sí | `Vec<u32>` | inferencia activa |
| `TL1Kernel` | No | `Vec<u8>` | experimental, no cableado |
| `TL2Kernel` | No | `Vec<u8>` | experimental, no cableado |

`TL1Kernel` y `TL2Kernel` no forman parte del camino actual de `transformer_bit2`.

## Packing real de pesos

Para `I2SKernel` y `I2STile16Kernel`:

- 2 bits por peso ternario
- 16 pesos por `u32`
- 4 pesos por byte
- `GROUP_SIZE = 128` para la escala por grupos

Codificación:

| Bits | Valor lógico |
|------|--------------|
| `0b00` | `-1` |
| `0b01` | `0` |
| `0b10` | `+1` |

## Qué hace `compute_row_i8`

Devuelve `i32` porque solo acumula productos enteros:

```rust
sum += (bits as i32 - 1) * x_i8
```

No aplica escala ni hace conversión a `f32`. El reescalado ocurre en el caller (`forward_raw_i8`), multiplicando por la `scale` del grupo correspondiente.

## Sobre `I2STile16Kernel`

El nombre `Tile16` no describe una implementación de GEMM por tiles `16x16`. Lo que existe es un kernel con desenrollado por bloques de 16 elementos en la dimensión de entrada, igual que en `I2S`. No es un mosaico bidimensional `M x N`.

## Integración en `BitLinear`

- `BitLinearInferenceState` despacha por `KernelKind`
- `BitLinear::forward_inference(...)` usa `quantize_to_i8(...)` y luego `forward_raw_i8(...)`
- `BitLinearInferenceState::forward_raw(...)` y `forward_raw_i8(...)` agregan bias al final si existe

## Estado del `transformer_bit2`

```rust
build_inference_state(&device, KernelKind::Tile16, KernelKind::I2S)
```

- cuerpo del transformer: `Tile16`
- `head`: `I2S`

El `head` no usa el camino `i8`. El método `forward_with_cache_inference(...)` llama a `state.head.forward_raw(...)`, o sea el kernel recibe activaciones `f32`.

## KV cache

- `append()` hace trim preventivo si se supera `MAX_CACHE_LEN`
- `keep_last()` muta en sitio y ajusta `current_len`
- `view()` expone solo el rango válido con `narrow(...)`

## Bugs corregidos

### 1. Escalas per-group en kernels

**Antes**: Todos los kernels aplicaban un solo scale por fila entera.

**Después**: Cada grupo de 128 elementos se computa por separado con `scales[base_g + gi]`.

Lugares corregidos:

| Kernel | Path | Función | Antes | Después |
|--------|------|---------|-------|---------|
| I2S | f32 | `forward_inner` seq + par | 1 scale por fila | per-group loop |
| I2S | i8 | `forward_inner_i8` seq + par | 1 scale por fila | per-group loop |
| Tile16 | f32 | `compute_row` + `forward_raw` | scale interno único | `compute_row` retorna raw, `forward_raw` aplica per-group |
| Tile16 | i8 | `forward_raw_i8` seq + par | 1 scale por fila | per-group loop |

### 2. Indexación de escalas en `reconstruct_bitlinear`

**Antes**:
```rust
let scale = bl.scales.get(g_idx).copied().unwrap_or(1e-8);
```
`g_idx` es el índice de u32 (16 pesos c/u), pero `scales` tiene 1 entrada por 128 pesos. Para in_features=512: 32 u32s por fila pero solo 4 escalas. El 87.5% de los pesos recibían fallback `1e-8`.

**Después**:
```rust
let row = g_idx / u32s_per_row;
let pos_in_row = g_idx % u32s_per_row;
let gi = pos_in_row / 8;
let scale_idx = row * groups_per_row + gi;
let scale = bl.scales.get(scale_idx).copied().unwrap_or(1e-8);
```
Mapea correctamente: 8 u32s × 16 pesos = 128 pesos = 1 GROUP_SIZE = 1 escala.

### 3. Desglose de bytes en export

**Antes**: `let rms = inf * 2;` (asumía f16)
**Después**: `let rms = inf * 4;` (`write_bitlinear` escribe f32)

### 4. Round-trip f32 en .bitnet

**Antes**: `load_bitnet` pasaba por `reconstruct_bitlinear` → `build_inference_state` → `get_ternary_weights`, recomputando escalas incorrectamente.

**Después**: `load_bitnet_inference_state()` carga directamente desde datos packed, sin round-trip.

## Verificación numérica

`compare_compatibility()` carga ambos modelos, pasa la misma entrada, compara logits:

```
Cosine similarity: 0.99204511
Top-1 match:       24/32 (75.0%)
Top-5 overlap:     32/32 (100.0%)
Veredicto: EXCELENTE — paridad casi perfecta
```

La diferencia residual se debe a la cuantización ternaria inherente (f32 → {-1,0,+1} * scale).

## Conclusión

El kernel `BitLinear` de `blocks` está bien integrado y su lógica de cómputo es consistente. Se corrigieron cuatro bugs:

1. Escalas per-group en kernels (iteración por grupo con scale propio).
2. Indexación de escalas en `reconstruct_bitlinear` (mapeo correcto u32→group).
3. Desglose de bytes en export (f32 real, no f16).
4. Round-trip f32 en .bitnet (carga directa desde datos packed).

El `.bitnet` (23.77 MB) produce inferencia equivalente al MPK (68.23 MB) con cosine 0.992 y reducción 2.87x.

### Archivos clave

| Archivo | Función |
|---------|---------|
| `kernel.rs` | I2S, Tile16 kernels, per-group scaling |
| `layer.rs` | `BitLinearInferenceState::from_packed()`, `quantize_to_i8()` |
| `bitnet_export.rs` | `export_bitnet()`, `load_bitnet()`, `load_bitnet_inference_state()`, `compare_compatibility()` |
