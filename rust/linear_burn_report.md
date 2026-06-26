# Linear Burn Report — Convención de pesos en Burn 0.21 vs PyTorch

## Problema

Al cargar pesos de un modelo PyTorch (`.safetensors`) en un modelo Burn 0.21.0-pre.4, el matmul fallaba con:

```
matmul: inner dimensions must match
  left: 256
  right: 128
```

A pesar de que los shapes impresos con `dims()` mostraban dimensiones correctas (e.g. `k_proj.weight shape: [128, 256]`).

## Causa raíz

**Burn 0.21** almacena y usa los pesos `Linear` con forma `[d_input, d_output]` y la forward es `input @ weight` (producto matricial directo, SIN transponer el peso).

**PyTorch** almacena los pesos `Linear` con forma `[d_output, d_input]` (out_features, in_features), y la forward es `input @ weight^T`.

### Código fuente de Burn 0.21

En `burn-nn-0.21.0-pre.4/src/modules/linear.rs`:

```rust
pub struct LinearConfig {
    pub d_input: usize,   // ← primero
    pub d_output: usize,  // ← segundo
}

// Para LinearLayout::Row (default):
let shape = [self.d_input, self.d_output];  // ← [in, out]
```

Y la forward llama a `linear(input, weight, bias)` que en `burn-tensor` hace simplemente `input @ weight` (en la implementación del backend Flex).

### Stack trace que confirmó el origen

```
burn_flex::ops::matmul::matmul
burn_backend::backend::ops::modules::linear::linear    ← matmul dentro de linear
burn_nn::modules::linear::Linear<B>::forward
xlstm::blocks::trasformer::heads::QKVProjection<B>::forward  ← qkv.forward()
xlstm::blocks::trasformer::attention::Attention<B>::forward_with_cache
```

## Explicación del error numérico

Para `k_proj` con `d_model=256`, `num_kv_groups=4`, `head_dim=32`:
- **PyTorch**: `nn.Linear(256, 128)` → peso `[128, 256]`, forward: `input @ W^T` = `[1,1,256] @ [256,128]` ✓
- **Burn**: `LinearConfig::new(256, 128)` → peso `[256, 128]`, forward: `input @ W` = `[1,1,256] @ [256,128]` ✓

Al cargar el peso PyTorch `[128, 256]` directamente en el peso Burn (que espera `[256, 128]`):
- **Después de cargar**: peso `[128, 256]`
- **Forward Burn**: `input @ W` = `[1,1,256] @ [128,256]`
- **Matmul**: left (última dim de input) = 256, right (penúltima dim de weight) = 128
- **256 ≠ 128** → ¡Error!

## Solución

Transponer todos los pesos `Linear` al cargar desde PyTorch, porque:
```
PyTorch [out, in] → transpose() → Burn [in, out]
```

### Pesos afectados en nuestro modelo (d_model=256, num_kv_groups=4, head_dim=32, ffn_intermediate=704, vocab=16000)

| Peso | PyTorch shape | Burn shape (después de transpose) |
|------|---------------|-----------------------------------|
| q_proj.weight | [256, 256] | [256, 256] (cuadrada, igual) |
| k_proj.weight | [128, 256] | [256, 128] |
| v_proj.weight | [128, 256] | [256, 128] |
| o_proj.weight | [256, 256] | [256, 256] (cuadrada, igual) |
| gate_proj.weight (linear_inner) | [704, 256] | [256, 704] |
| up_proj.weight (linear_outer) | [704, 256] | [256, 704] |
| down_proj.weight | [256, 704] | [704, 256] |
| head.weight | [16000, 256] | [256, 16000] |
| embedding.weight | [16000, 256] | [16000, 256] (NO es Linear, no se transpone) |

### Cambio en código

```rust
// ANTES (roto):
model.layers[i].attention.qkv.k_proj.weight =
    Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device));

// DESPUÉS (arreglado):
model.layers[i].attention.qkv.k_proj.weight =
    Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
```

### Archivos modificados
- `rust/src/blocks/load_pytorch.rs` — todas las asignaciones de pesos Linear en `load_into_transformer`
- `rust/xorIA/transformer_quant_kv.rs` — asignación de `head.weight` en el bloque de carga Python

## Nota importante

Esta convención (`[in, out]` con forward `input @ weight`) es específica de **Burn 0.21**. En Burn 0.20 y anteriores, la convención era `[out, in]` con forward `input @ weight^T` (igual que PyTorch). El cambio ocurrió en la versión 0.21.
