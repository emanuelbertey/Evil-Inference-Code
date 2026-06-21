# xoria_bit_cuda — Análisis y Mejoras

## Implementación Actual

**Archivo:** `xorIA/xoria_bit_cuda.rs` (428 líneas)
**Backend:** `Autodiff<burn_cuda::Cuda<f32>>` (Burn + CUDA)
**Modelo:** `TransformerBitLinearLM` con 3 capas `BitLinear` (pesos ternarios 2-bit, activaciones 8-bit)

### Componentes

| Componente | Detalle |
|---|---|
| **Training** | AdamW + weight decay + gradient clipping, CrossEntropyLoss |
| **Data** | `FileFragmentIterator` (fragmentos de 1MB) → tokenize → `create_batch` |
| **Inferencia** | I2S Kernel CPU con KVCache f32 (trim a 70 cuando llega a 200) |
| **Export** | `export_bitnet` después de cada epoch → `.bitnet` compatible CPU |
| **Config** | `config.toml` auto-detect, menú interactivo, o defaults hardcodeados |

### Pipeline de entrenamiento

```
Fragmento 1MB → Tokenize → split en ventanas de 64 tokens con stride 64
                          → batch de 8 ventanas → forward → loss → backward → step
```

---

## Problema Principal: Pocos Tokens por Segundo

### Causas

**1. Tokenización repetida por epoch**
```rust
for epoch in 0..num_epochs {
    let fragments = FileFragmentIterator::new(Path::new(&text_file), 1)?;
    for (frag_idx, fragment) in fragments.enumerate() {
        let tokens = tokenizer.encode(&fragment);  // ← se tokeniza CADA epoch
```
Cada epoch re-tokeniza todo el dataset desde cero. Para un dataset de 100MB con epoch=10, son 1GB de tokenización innecesaria.

**2. Fragmentos pequeños descartados**
```rust
let nb = tokens.len() / tpb;  // tpb = batch_size * seq_len = 8 * 64 = 512
if nb == 0 { continue; }  // fragmentos < 512 tokens → se descartan
```
Con fragmentos de 1MB, un fragmento con <512 tokens (~400 caracteres) se pierde. En texto real con párrafos cortos puede ser ~10-20% del dataset.

**3. Sin shuffling**
Los fragmentos se procesan en orden secuencial, igual que los batches dentro de cada fragmento. El modelo ve el texto siempre en el mismo orden, lo que perjudica la generalización.

**4. stride = seq_len = 64**
```rust
let seq_len = 64;
let stride = 64;  // sin overlapping
```
Stride igual a seq_len significa que no hay overlapping entre ventanas consecutivas. Cada token se ve exactamente una vez por epoch, pero se pierde contexto entre ventanas.

**5. batch relativamente chico**
```rust
let mut batch_size: usize = 8;  // 8 × 64 = 512 tokens por paso
```
En GPU esto es bajo. Una RTX puede manejar batch 32-64 sin problema para d_model=512.

### Estimación de throughput

Con defaults (d_model=512, layers=6, batch=8, seq=64):
- ~2.5M parámetros (6 capas × ~400K c/u)
- ~500 tok/s en GPU mid-range (RTX 3060)
- Dataset de 10MB → ~500k tokens → ~1000 batches → ~2 min por epoch
- La tokenización repetida puede duplicar o triplicar ese tiempo

---

## Mejoras Propuestas

### 1. Pre-tokenizar el dataset (ALTA prioridad)

```rust
// Una sola vez, antes del training loop
let tokens: Vec<usize> = {
    let text = std::fs::read_to_string(&text_file)?;
    tokenizer.encode(&text)
};
```

En vez de fragmentos de 1MB, cargar el texto completo y tokenizar UNA SOLA VEZ. Esto elimina la sobrecarga de `FileFragmentIterator` + tokenización repetida.

### 2. Aumentar batch size

| GPU | batch_size recomendado | tok/s estimado |
|---|---|---|
| GTX 1060 (6GB) | 8-16 | ~300-500 |
| RTX 3060 (12GB) | 16-32 | ~800-1200 |
| RTX 4090 (24GB) | 64-128 | ~3000-5000 |

Hacer configurable desde `config.toml` con default según VRAM detectada.

### 3. Overlapping windows (stride < seq_len)

```rust
let stride = seq_len / 2;  // 50% overlap
// Duplica los tokens procesados por epoch, mejora coherencia
```

### 4. Shuffle batches

```rust
use rand::seq::SliceRandom;
let mut indices: Vec<usize> = (0..total_batches).collect();
indices.shuffle(&mut rand::rng());
for idx in indices {
    let (x, y) = create_batch(&tokens, idx * tpb, ...);
    // ...
}
```

### 5. DataLoader estilo PyTorch (prefetch + threading)

Implementar un buffer circular que tokenice y prepare batches en un thread separado mientras la GPU entrena.

### 6. Evaluación periódica + early stopping

```rust
if batch_count % eval_interval == 0 {
    let val_loss = compute_validation_loss(&model, &val_tokens);
    if val_loss < best_loss {
        best_loss = val_loss;
        model.save_file(&best_model_file, &recorder)?;
    }
}
```

### 7. Gradient accumulation

Para batches efectivos grandes sin aumentar VRAM:
```rust
for _ in 0..gradient_accumulation_steps {
    let loss = compute_loss(batch);
    loss.backward();  // acumula gradientes
}
optim.step();  // un paso con gradiente acumulado
optim.reset_grad();
```

---

## Resumen de Impacto

| Mejora | Impacto en tok/s | Esfuerzo |
|---|---|---|
| Pre-tokenizar | 2-3× | Bajo (10 líneas) |
| Aumentar batch (16→32) | 1.5-2× | Bajo (cambiar default) |
| Overlapping (stride=32) | 2× más tokens/epoch | Bajo (cambiar constante) |
| Shuffle | Mejor convergencia | Bajo |
| Gradient accumulation | Permite batch efectivo grande | Medio |
| DataLoader prefetch | 1.2-1.5× | Medio-Alto |

**Total potencial: 5-10× más tokens procesados por segundo** combinando pre-tokenización + batch más grande + stride óptimo.
