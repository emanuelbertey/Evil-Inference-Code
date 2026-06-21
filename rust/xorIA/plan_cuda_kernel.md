# Plan: Kernel CUDA I2S Optimizado

## Estado actual
`test_cuda_kernel.rs` tiene un kernel naive: cada thread lee 1 peso por iteración, sin coalescing, sin shared memory. Sirve para verificar corrección matemática, **no para producción**.

## Pipeline de optimización (6 niveles)

### Nivel 1 — Transponer pesos en host (ganancia ~6×)
```c
// Layout actual: weights[col][kp]  → acceso strideado
// Layout óptimo: weights[kp][col]  → acceso coalescente
```
- Hilos contiguos (col, col+1) leen posiciones adyacentes en VRAM → `128 bytes por transacción` vs `4 bytes`
- Se transpone en CPU antes de copiar a GPU (una sola vez)
- **Costo**: 1 pasada O(N × K_packed) al cargar modelo
- **Kernel**: `weights[kp * N + col]` en vez de `weights[col * K_packed + kp]`

### Nivel 2 — Shared memory tiling (ganancia ~8×)
```
Block: 16×16 = 256 threads
Tile:  input[16 × tile_K] → SRAM
      weights[tile_K × 16] → SRAM
```
- 1 tile de weights + 1 tile de input en shared memory (latencia 1 ciclo vs 200-400)
- Reducción en K hecha completamente en SRAM
- **Ocupación**: ~50-64 registers/thread, 0 bytes shared → usar `block(16,16)` o `(8,32)`

### Nivel 3 — Vectorized loads (ganancia ~2×)
```c
float4 in_vec = ((float4*)input)[base / 4];  // 128 bits de una
```
- Carga 4 floats por instrucción en vez de 1
- Se aplica tanto en input como en weights (usando `uint4`)
- **Requiere alineación** de las filas a múltiplo de 16 bytes

### Nivel 4 — Desenrollado manual + prefetch (ganancia ~1.5×)
```c
// En vez de loop por 16 elementos dentro del tile:
unsigned int w0 = __ldg(&weights[tile_w + 0]);
unsigned int w1 = __ldg(&weights[tile_w + 1]);
// ... desenrollado a mano
```
- Reduce overhead de control divergence
- `__ldg()` fuerza cache de texturas (solo lectura)

### Nivel 5 — Tensor Cores (ganancia ~4× si viable)
- I2S ternario no mapea directamente a tensor cores (requieren fp16/bf16/int8)
- Opción: convertir ternario a fp16 en shared memory → tensor core matmul
- **Tradeoff**: overhead de conversión vs throughput ×4
- Probablemente no vale la pena a menos que K ≥ 512

### Nivel 6 — Fusión con RoPE + KV Cache (ganancia ~2×)
- Fusionar el I2S matmul Q/K con RoPE + KV Cache en un solo kernel
- Elimina escrituras/lecturas intermedias a VRAM
- **Más complejo**: kernel unificado ~200 líneas

## Plan de implementación

| Paso | Archivo | Cambio |
|------|---------|--------|
| 0 | `plan_cuda_kernel.md` | Este plan |
| 1 | `xorIA/bin/kernels/i2s_v1_transposed.cu` | Kernel con pesos transpuestos |
| 2 | `xorIA/bin/kernels/i2s_v2_tiled.cu` | Kernel con shared memory tiling |
| 3 | `xorIA/bin/kernels/i2s_v3_vectorized.cu` | Kernel con vectorized loads |
| 4 | `xorIA/bin/kernels/i2s_v4_unrolled.cu` | Kernel con desenrollado |
| 5 | `test_cuda_kernel.rs` | Menú para seleccionar versión de kernel |
| 6 | `benches/i2s_bench.rs` | Benchmark de cada versión |
| 7 | `src/blocks/...` | Integrar kernel óptimo en modelo real |

## Estructura de archivos
```
xorIA/bin/kernels/
├── i2s_v0_naive.cu       # kernel actual (referencia)
├── i2s_v1_transposed.cu  # pesos transpuestos
├── i2s_v2_tiled.cu       # + shared memory
├── i2s_v3_vectorized.cu  # + vector loads
├── i2s_v4_unrolled.cu    # + desenrollado
├── i2s_v5_fused.cu       # + RoPE/KV fusion
└── common.h              # helpers compartidos

xorIA/plan_cuda_kernel.md  # este archivo
```
Opcional: usar `include_str!()` en vez de constantes rust para los `.cu`.

## Cómo medir
```bash
cargo bench -- i2s 2>&1
# o via nsys:
nsys profile -o i2s_profile cargo run --release --bin test_cuda_kernel
```
Métricas clave:
- Bandwidth utilization (GB/s) — DRAM vs SRAM
- Occupancy (active warps / max warps)
- Compute utilization (%)

## Meta final
- Kernel I2S CUDA para producción en el chasis d_model=768, layers=24, experts
- Integrado en `xoria_bit_cuda.rs` como backend de inferencia CUDA
- Benchmark público en BENCHMARK_REPORT.md
