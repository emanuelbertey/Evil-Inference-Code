# Plan de Implementación — Transformer Mejorado

## Features ordenados por prioridad

| Prioridad | Feature | Esfuerzo | Impacto | Status |
|---|---|---|---|---|
| P0 | Partial RoPE (porcentaje de dims rotadas) | ~30 líneas | Estabilidad training (Kimi, Phi-2) | ✅ Implementado |
| P0 | x0 injection (embedding inicial inyectado por capa) | ~15 líneas | Estabilidad, gradiente flow | ✅ Implementado |
| **P1** | **Weight tying** (`lm_head.weight = embed.weight`) | **~5 líneas** | **Reduce params ~30%, convergencia más rápida** | **❌ Pendiente** |
| **P2** | **Presence penalty** (penaliza aditivo tokens generados) | **~10 líneas** | **Mejor diversity en sampling** | **❌ Pendiente** |
| **P3** | **Residual lambdas** (escalar aprendido por capa) | **~15 líneas** | **Mejor gradiente flow en modelos profundos** | **❌ Pendiente** |
| P4 | U-Net skip connections (early→late) | ~50 líneas | Estabilidad modelos profundos | ❌ Pendiente |
| P5 | Chimera topology (bottom únicas + top loop) | Arquitectural | Escalar params sin crecer capas | ❌ Pendiente |
| P6 | Gated Delta Net + Differential Attention | Arquitectural | RNN híbrido, cancelación de ruido | ❌ Pendiente |

---

## P1 — Weight Tying

**Qué hace**: El `lm_head` (proyección final a logits) comparte los mismos pesos que el `Embedding`. Esto forcejea la matriz de embedding a ser tanto input como output, reduciendo parámetros ~30% y mejorando convergencia.

**Implementación** (Auron `model.py:257-258`):
```python
self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
self.lm_head.weight = self.embed.weight
```

**En Burn Rust**:
```rust
// En construcción del modelo:
head: LinearConfig::new(d_model, vocab_size).with_bias(false).init(&device),
// Después del init, copiar pesos del embedding al head:
// O mejor: usar embedding.weight como head.weight mediante Param compartido
```

**Archivos a modificar**: `transformer_quant_kv.rs` — solo la construcción del modelo.

---

## P2 — Presence Penalty

**Qué hace**: Además del repetition penalty (multiplicativo), añade un **penalizador aditivo** a todos los tokens que ya fueron generados, no solo los que están en el contexto. Qwen 3 y otros lo usan para mejorar diversidad.

**Implementación** (Auron `generate.py:127-130`):
```python
if presence_pen != 0:
    for tid in generated_ids:
        next_logits[tid] -= presence_pen
```

**En Rust**: Agregar un `HashSet<usize>` con los tokens generados en la sesión, y antes de samplear restar `presence_pen` a sus logits.

**Archivos a modificar**: `transformer_quant_kv.rs` — las funciones `generate_kuant_cached` y `generate_text_cached`.

---

## P3 — Residual Lambdas

**Qué hace**: Cada capa tiene un escalar aprendido que escala el residual antes de sumar:
```python
out = lambda_attn * x + attn(x)
out = lambda_mlp * x + mlp(x)
```
En lugar de `out = x + attn(x)`. Esto permite que el modelo aprenda qué tanto confiar en el residual vs la transformación capa por capa.

**Implementación** (Auron `model.py:188-193`):
```python
self.resid_attn = nn.Parameter(torch.full((), 1.1 ** 0.5))
self.resid_mlp = nn.Parameter(torch.full((), 1.1 ** 0.5))
```

**En Rust**: Similar a x0_lambdas — `Option<Param<Tensor<B, 1>>>` por capa. Inicializado a `sqrt(1.1)`.

**Archivos a modificar**: `transformer_quant_kv.rs` — struct + forward + constructor.

---

## P4 — U-Net Skip Connections

**Qué hace**: Conecta capas tempranas con capas tardías simétricamente (como U-Net). La capa i se conecta con la capa (N-1-i) mediante un skip ponderado.

**Implementación** (Auron `model.py:246-253`):
```python
if cfg.use_skip_connections and cfg.n_layers >= 4:
    n_skips = cfg.n_layers // 2
    self.skip_weights = nn.ParameterList([
        nn.Parameter(torch.ones(cfg.dim)) for _ in range(n_skips)
    ])
```

**Forward**: En la primera mitad del modelo, guardar `x` en una pila. En la segunda mitad, sumar `skip_weight * x_skip` al `x` actual.

---

## P5 — Chimera Topology

**Qué hace**: Divide el modelo en:
- **Bottom**: `N` capas únicas (feature extraction)
- **Top**: `M` capas físicas compartidas que se loopan `L` veces (razonamiento iterativo)

Esto da `N + M*L` capas virtuales con solo `N + M` capas físicas de parámetros.

**Implementación** (Auron `model.py:216-253`):
- `bottom_blocks: ModuleList` (únicas)
- `top_blocks: ModuleList` (compartidas, loopeadas)
- `top_layer_ids: Parameter` (embedding de posición por loop/layer)
- `x0_lambdas` y `skip_connections` operan sobre capas virtuales

---

## P6 — GDN + Differential Attention

**GDN (Gated Delta Net)**: Recurrente O(n), ideal para la mayoría de capas. Usa una compuerta delta para actualizar estado oculto.

**DiffAttention**: Pares de cabezas de atención que se restan para cancelar ruido. Cada par produce una sola salida:
```python
y = sub_norm(attn1 - lam * attn2)
```

Requiere `fla` (Flash Linear Attention) library.

---

## Resumen de implementación actual

| Feature | File | Detalle |
|---|---|---|
| Partial RoPE tensor ops | `ops.rs:apply_rope_partial` | Training, preserva autodiff |
| Partial RoPE fused | `ops.rs:apply_rope_fused_partial` | Inferencia TurboQuant |
| x0 injection | `transformer_quant_kv.rs:TransformerLM` | `Option<Param>`, carga retrocompatible |
| UI: RoPE% y x0 toggle | `transformer_quant_kv.rs` | Menú configuración |
