# KuantGrad — Gradient Compression Specification

## Overview

KuantGrad compresses gradient tensors during training by grouping 8 consecutive f32 values and encoding them as 9 bytes:

- **1 × f32** (4 bytes): group intensity/relief (max absolute value)
- **8 × 5-bit** (5 bytes): per-element magnitude as fraction of intensity

**Total per group**: 9 bytes (vs 32 bytes raw) → **3.56× compression**

## Encoding

### Group layout (per 8 consecutive gradient elements)

```
Byte 0-3:  f32 scale (little-endian IEEE 754)
Byte 4-8:  40 bits packed (8 values × 5 bits each)

Bit packing (40 bits = 5 bytes, little-endian bit order):
  Bits  0-4:   value 0
  Bits  5-9:   value 1
  Bits 10-14:  value 2
  Bits 15-19:  value 3
  Bits 20-24:  value 4
  Bits 25-29:  value 5
  Bits 30-34:  value 6
  Bits 35-39:  value 7
```

### scale computation

```
scale = max(|g[0]|, |g[1]|, ..., |g[7]|)
```

If `scale == 0.0`, the 5-bit field is skipped (5 zero bytes written/read) and all 8 values decode to 0.0.

### Quantization

Each gradient `g[i]` is mapped to a 5-bit unsigned value `q ∈ [0, 31]`:

```
norm = clamp(g[i] / scale, -1.0, +1.0)
q    = round((norm + 1.0) * 15.5)
q    = clamp(q, 0, 31)
```

| norm | q | meaning |
|------|---|---------|
| -1.0 | 0 | negative full intensity |
| 0.0 | 15-16 | zero (center) |
| +1.0 | 31 | positive full intensity |

### Dequantization

```
norm = (q / 15.5) - 1.0       # q=0→-1, q=15.5→0, q=31→+1
g[i] = norm * scale
```

**Maximum quantisation error**: ±1/31 ≈ ±3.2% of scale per element.

## API

```rust
/// Compress: f32[] → (compressed u8[], number_of_groups)
pub fn compress(grads: &[f32]) -> (Vec<u8>, usize);

/// Decompress: (compressed u8[], n_groups, original_len) → f32[]
pub fn decompress(data: &[u8], n_groups: usize, original_len: usize) -> Vec<f32>;
```

## Integration with AdamW

During training, the gradient compression is applied between `loss.backward()` and `optim.step()`:

```
Standard:        gradient → AdamW         → weight update
KuantGrad:       gradient → compress → decompress → AdamW → weight update
```

The `TinyBitNet` in `test_kuantgrad.rs` uses finite-difference gradients on a 3×16 ternary network to demonstrate the loss/accuracy impact.

## Pending / Future

- [ ] Direct integration into Burn's `GradientsParams` pipeline (modify gradient tensors in-place before `optim.step()`)
- [ ] Per-layer adaptive group size (auto-tune GROUP based on gradient variance)
- [ ] Mixed-precision state: store momentum/variance in KuantGrad format
- [ ] GPU kernel for compress/decompress (CUDA I2S-style)
