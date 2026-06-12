# mLSTM Equivalence Fixes (Rust vs Python)

This document summarizes the changes made to the Rust implementation (Burn framework) to achieve perfect numerical and gradient equivalence with the Python implementation (PyTorch).

## Success Metrics
- **Forward Pass (VALOR):** Difference < 1e-7 across sequence lengths 1-12.
- **Backward Pass (GRADIENTS):** Difference < 1e-6 across all trainable parameters and inputs.
- **Verified via:** `cargo run --release --bin mlstm_equivalence` and `python tests/mlstm_equivalence.py`.

## Core Fixes

### 1. Gradient Propagation in Causal Convolution (`conv.rs`)
- **State Initialization:** The `empty_state()` function was previously returning a `Tensor::zeros` without the `.require_grad()` flag. In the recurrent `step()`, Burn would drop the gradient flow when hitting the zeroed padding. 
- **Graph Tracking in `step()`:** The manual multiplication of weights was using `.val()`, which in Burn 0.16 can detach the tensor from the Autodiff graph. This was replaced by calling the module's native `forward()` method on the sliding window state.
- **Fixed:** Recurrent gradient now correctly matches the parallel gradient (e.g., from `0.1` back to the expected `0.6` in testing).

### 2. Numerical Stability in `backends.rs`
- **Cumulative Sum (`cumsum_matrix`):** Replaced the previous `tril.matmul(x)` approach with a native iterative reconstruction using `Tensor::cat`. The matrix multiplication method for cumsum was causing vanishing/exploding gradients in long sequences.
- **Stabilization Logic:** Aligned the stabilized denominator calculation with the Python version using `max_pair` against the `exp(-max_log_D)` term, ensuring the stabilizer logic is identical in both backends.

### 3. State and Normalization Alignment
- **`m_state` Initialization:** Changed default `m_state` from `-30.0` to `0.0` in `layer.rs` and `cell.rs` to match the Python implementation's starting point.
- **LayerNorm (`ln.rs`):** Replaced the `GroupNorm` based `MultiHeadLayerNorm` with a custom implementation that manually calculates mean/variance and applies weights without an implicit bias, mirroring the specific configuration used in the Python version.

## How to Test
Execute the equivalence suite in both environments to verify:
```bash
# Rust
cd rust
cargo run --release --bin mlstm_equivalence

# Python
cd ..
$env:PYTHONPATH = "."
python tests/mlstm_equivalence.py
```
