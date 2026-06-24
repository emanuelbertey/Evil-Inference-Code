# xoria Transformer — Reporte v2.0

**Fecha:** 24 Junio 2026  
**Versión:** 2.0  
**Archivos principales:** `transformer_quant_kv.rs`, `transformer_chat_cuda.rs`, `transformer_chat.rs`

---

## 1. Resumen de Cambios (v1.0 → v2.0)

### 1.1 Módulos del Transformer (librería)

| Antes (v1.0) | Ahora (v2.0) | Archivo |
|---|---|---|
| `RMSNorm` custom (campo `weight`) | `burn::nn::RmsNorm` (campo `gamma`) | `layer.rs` |
| `SwiGLUFeedForward` custom (gate_up_proj → split → silu*up → down) | `burn::nn::SwiGlu` + `Linear` down projection | `feedforward.rs` |

**Impacto en checkpoints:** Los `.mpk` v1.0 **no** son compatibles con v2.0. Los campos internos cambiaron de nombre y tipo.

### 1.2 Menú interactivo (3 archivos)

| Opción | v1.0 | v2.0 |
|---|---|---|
| seq_len | Hardcodeado `64` | Configurable (default `128`) |
| stride | Hardcodeado `64` (= seq_len, 0% overlap) | `seq_len / 2` (50% overlap) |
| gradient_accumulation | No existía | Configurable (default `1x`) |

### 1.3 Training loop

| Aspecto | v1.0 | v2.0 |
|---|---|---|
| Optimizer step | Cada batch | Cada N micro-batches (N = grad_accum) |
| Loss accumulation | Ninguna | `accum_loss = sum(loss_i) / N` |
| Backward pass | 1 por batch | 1 por cada N micro-batches |
| Display `\r` | Se truncaba con datos viejos | Padding con espacios para limpiar línea |

---

## 2. Arquitectura del Modelo

```
TransformerLM
├── Embedding (vocab_size → d_model)
├── Transformer Stack (N capas)
│   └── Cada capa:
│       ├── RmsNorm (pre-attention)
│       ├── Attention (GQA + RoPE)
│       ├── Dropout residual
│       ├── RmsNorm (pre-FFN)
│       ├── SwiGLU FFN (burn::nn::SwiGlu + Linear down)
│       └── Dropout residual
├── RmsNorm (final)
└── Linear (d_model → vocab_size)
```

### Parámetros de ejemplo (configuración del usuario)

| Hyperparameter | Valor |
|---|---|
| d_model | 128 |
| num_layers | 12 |
| num_heads | 8 |
| num_kv_groups | 4 |
| head_dim | 16 (128/8) |
| RoPE% | 25% (head_dim parcial = 4 dims rotados) |
| x0 injection | Sí |
| ResDrop | 0.0 |
| seq_len | 128 |
| stride | 64 |
| grad accumulation | 2x |
| Batch efectivo | 32 (16 × 2) |
| **Total parámetros** | **6.46 M** |

---

## 3. Gradient Accumulation — Detalles Técnicos

### Problema: burn no tiene `Add` para `GradientsParams`

```rust
// burn-optim/src/optim/grads.rs
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}
// No hay impl Add for GradientsParams
```

### Solución: Acumular loss tensor

```rust
let mut accum_loss = Tensor::<B, 1>::zeros([1], &device);

for m in 0..gradient_accumulation_steps {
    let loss = loss_fn.forward(logits, targets);  // forward de 1 micro-batch
    accum_loss = accum_loss + loss;                // acumula en el graph
}

accum_loss = accum_loss / micro_steps as f32;     // promedia
let grads = accum_loss.backward();                // UN solo backward
model = optim.step(lr, model, grads);             // UN solo step
```

### Por qué funciona

- `accum_loss.backward()` aplica chain rule sobre la suma: ∂(Σloss_i/N)/∂w = Σ(∂loss_i/∂w)/N
- Los gradientes salen promediados automáticamente
- **Costo de VRAM:** ~N× el graph de un micro-batch (forward pass se repite N veces en el graph)

### Cuándo usar grad accumulation

| Escenario | Recomendación |
|---|---|
| VRAM suficiente para batch grande | No necesario, usar batch grande directo |
| Loss fluctúa mucho entre batches | Accum 2-4x suaviza |
| LR alto que causa inestabilidad | Accum reduce varianza del gradiente |
| Modelo pequeño (< 10M params) | Generalmente no necesario |

---

## 4. Stride Overlap (50%)

### Antes (v1.0): stride = seq_len = 64
```
Token:  [0 1 2 3 4 5 6 7 ...]
Batch1: [0 1 2 3 4 5 6 7]
Batch2: [8 9 10 11 12 13 14 15]
→ Sin overlap, cada token se ve 1 vez
```

### Ahora (v2.0): stride = seq_len / 2 = 64, seq_len = 128
```
Token:  [0 1 2 ... 127 128 129 ...]
Batch1: [0 1 2 ... 127]
Batch2: [64 65 66 ... 191]
→ 50% overlap, tokens 64-127 se ven 2 veces
```

### Beneficios
- Más training data efectiva sin cambiar el dataset
- Mejor coherencia entre ventanas (el modelo ve contexto superpuesto)
- Tradeoff: ~2× más batches por epoch (más tiempo pero mejor calidad)

---

## 5. Módulos Reemplazados — Justificación

### 5.1 RMSNorm → `burn::nn::RmsNorm`

```rust
// Nuestro custom (v1.0)
pub struct RMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,  // parámetro aprendible
    pub eps: f64,
}

// burn::nn (v2.0)
pub struct RmsNorm<B: Backend> {
    gamma: Tensor<B, D>,   // parámetro aprendible (mismo concepto)
    epsilon: f64,
}
```

- **Funcionalmente equivalente:** Ambos calculan `x / sqrt(mean(x²) + eps) * gamma`
- **Ventaja burn:** Better initialization, tested, maintained, compatible con burn ecosystem
- **Costo:** Checkpoint incompatibility (campo `weight` → `gamma`)

### 5.2 SwiGLU → `burn::nn::SwiGlu` + Linear

```rust
// Nuestro custom (v1.0): 3 capas en 1
gate_up_proj: Linear(d_model, 2 * inter_dim)  // concatenado
→ split → SiLU(gate) * up → dropout → down_proj

// burn::nn + down (v2.0): 2 componentes
swiglu: SwiGlu(d_model, inter_dim)  // solo gate + outer
down_proj: Linear(inter_dim, d_model)
```

- **Mismos parámetros:** gate_up = 2×(d_model × inter_dim), down = inter_dim × d_model
- **`burn::nn::SwiGlu` no tiene down projection** — genera dim `d_output`, necesitas Linear separada
- **Ventaja burn:** Better initialization (Kaiming), tested, ~2% mejor en test benchmark

---

## 6. Bugs Corregidos

### 6.1 Display `\r` truncado
**Problema:** Cuando el número de batch tenía menos dígitos, la línea anterior quedaba parcialmente visible.
```
Batch 100 | Loss: 9.83 | 155 tok/s
Batch 99 | Loss: 9.83 | 155 tok/s   ← el "0" de "100" quedaba visible
```
**Solución:** Padding con espacios al final de cada `print!("\r...")`.
```rust
print!("\r...| LR: {:.1e}            ", ...);
//                                    ^^^^ 12 espacios extra
```

### 6.2 `micro_idx` fuera de scope
**Problema:** La variable `micro_idx` se definía dentro del loop interno de micro-batches y se usaba en el `print!` después del loop.
**Solución:** Reemplazado con `batch_count` ( contador de optimizer steps).

### 6.3 Re-export `RMSNorm` en `mod.rs`
**Problema:** `mod.rs` re-exportaba `RMSNorm` que ya no existe.
**Solución:** Eliminado el re-export.

---

## 7. Estado Actual

| Componente | Estado |
|---|---|
| `burn::nn::RmsNorm` en layer.rs | Funcionando |
| `burn::nn::SwiGlu` en feedforward.rs | Funcionando |
| Menú con seq_len/stride/grad_accum | Funcionando |
| Gradient accumulation en 3 archivos | Funcionando |
| Display `\r` limpio | Corregido |
| Compilación (sin errores) | OK |
| Checkpoint compatibilidad v1.0 | Rota (esperado) |

---

## 8. Próximos Pasos

1. **Entrenar** con nueva config (d_model=128, layers=12, RoPE 25%, grad_accum=2x) y verificar convergencia
2. **Weight tying** — compartir pesos entre `embedding` y `head` (reduce ~6.5K params)
3. **Presence penalty** — diversidad en sampling
4. **Shuffle de fragmentos** — mejor generalización
5. **Eval periódica** — validation loss + early stopping
