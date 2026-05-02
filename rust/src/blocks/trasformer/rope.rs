// ─── Rotary Position Embeddings (RoPE) ──────────────────────────────────────
//
// Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// (Su et al., 2021)
//
// RoPE encodes position information by rotating pairs of dimensions in Q and K
// using sinusoidal functions. This allows the dot product between Q_i and K_j
// to depend only on relative position (i - j), giving the model translation
// equivariance without explicit relative position biases.
//
// For each pair of dimensions (2k, 2k+1) at position m:
//   θ_k = base^(-2k/d)
//   [x_{2k}  ]     [cos(m·θ_k)  -sin(m·θ_k)] [x_{2k}  ]
//   [x_{2k+1}]  =  [sin(m·θ_k)   cos(m·θ_k)] [x_{2k+1}]
//
// Extended with NTK-aware scaling for context length extension.
//
// Usage:
//   let rope = RoPEConfig::new(head_dim, max_seq_len).init(&device);
//   let (q_rot, k_rot) = rope.forward(q, k, offset);

use burn::prelude::*;
use burn::config::Config;
use burn::module::Module;

// ─── RoPE Config ────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct RoPEConfig {
    /// Dimension of each attention head (must be even)
    pub head_dim: usize,
    /// Maximum sequence length to precompute embeddings for
    pub max_seq_len: usize,
    /// Base frequency for rotary embeddings (default: 10000.0)
    #[config(default = 10000.0)]
    pub base: f64,
    /// Scaling factor for NTK-aware context extension (default: 1.0 = no scaling)
    #[config(default = 1.0)]
    pub scaling_factor: f64,
}

// ─── RoPE Module ────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct RoPE<B: Backend> {
    /// Precomputed cosine table: (max_seq_len, head_dim/2)
    pub cos_cache: Tensor<B, 2>,
    /// Precomputed sine table: (max_seq_len, head_dim/2)
    pub sin_cache: Tensor<B, 2>,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl RoPEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RoPE<B> {
        assert!(self.head_dim % 2 == 0, "RoPE head_dim must be even, got {}", self.head_dim);

        let half_dim = self.head_dim / 2;
        let scaled_base = self.base * self.scaling_factor;

        // θ_k = base^(-2k/d) for k = 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|k| 1.0 / (scaled_base.powf(2.0 * k as f64 / self.head_dim as f64)) as f32)
            .collect();

        let inv_freq_tensor = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        // positions: [0, 1, 2, ..., max_seq_len - 1]
        let positions: Vec<f32> = (0..self.max_seq_len).map(|i| i as f32).collect();
        let pos_tensor = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

        // freqs = outer(positions, inv_freq) → (max_seq_len, half_dim)
        let freqs = pos_tensor.unsqueeze_dim::<2>(1) * inv_freq_tensor.unsqueeze_dim::<2>(0);

        let cos_cache = freqs.clone().cos();
        let sin_cache = freqs.sin();

        RoPE {
            cos_cache,
            sin_cache,
            head_dim: self.head_dim,
            max_seq_len: self.max_seq_len,
        }
    }
}

impl<B: Backend> RoPE<B> {
    /// Apply RoPE to query and key tensors.
    ///
    /// Inputs:
    ///   q: (batch, seq_len, num_heads, head_dim)
    ///   k: (batch, seq_len, num_kv_heads, head_dim)
    ///   offset: starting position index (for autoregressive generation)
    ///
    /// Returns:
    ///   (q_rotated, k_rotated) with same shapes
    pub fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_b, seq_len, _nh, _dh] = q.dims();
        assert!(
            offset + seq_len <= self.max_seq_len,
            "RoPE: offset({}) + seq_len({}) exceeds max_seq_len({})",
            offset, seq_len, self.max_seq_len
        );

        // Slice the precomputed tables for this position range
        let half_dim = self.head_dim / 2;
        let cos = self.cos_cache.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .unsqueeze_dim::<3>(0)   // (1, S, 1, half_dim)
            .unsqueeze_dim::<4>(2);
        let sin = self.sin_cache.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);

        let q_rotated = self.apply_rotation(q, cos.clone(), sin.clone());
        let k_rotated = self.apply_rotation(k, cos, sin);

        (q_rotated, k_rotated)
    }

    /// Apply RoPE to a single tensor (useful for Q-only or K-only rotation).
    pub fn apply_to_single(
        &self,
        x: Tensor<B, 4>,
        offset: usize,
    ) -> Tensor<B, 4> {
        let [_b, seq_len, _nh, _dh] = x.dims();
        let half_dim = self.head_dim / 2;

        let cos = self.cos_cache.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);
        let sin = self.sin_cache.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);

        self.apply_rotation(x, cos, sin)
    }

    /// Core rotation: split x into even/odd pairs, rotate by (cos, sin)
    ///
    /// x_rot[..., 2k]   = x[..., 2k] * cos[k] - x[..., 2k+1] * sin[k]
    /// x_rot[..., 2k+1] = x[..., 2k] * sin[k] + x[..., 2k+1] * cos[k]
    fn apply_rotation(
        &self,
        x: Tensor<B, 4>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [b, s, nh, dh] = x.dims();
        let half_dim = dh / 2;

        // Split into first-half and second-half
        let x_first = x.clone().slice([0..b, 0..s, 0..nh, 0..half_dim]);
        let x_second = x.slice([0..b, 0..s, 0..nh, half_dim..dh]);

        // Rotate
        let out_first = x_first.clone() * cos.clone() - x_second.clone() * sin.clone();
        let out_second = x_first * sin + x_second * cos;

        // Concatenate back
        Tensor::cat(vec![out_first, out_second], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rope_shape() {
        let device = Default::default();
        let rope = RoPEConfig::new(64, 512).init::<TestBackend>(&device);

        let q = Tensor::zeros([2, 16, 8, 64], &device);
        let k = Tensor::zeros([2, 16, 2, 64], &device);

        let (q_rot, k_rot) = rope.forward(q, k, 0);
        assert_eq!(q_rot.dims(), [2, 16, 8, 64]);
        assert_eq!(k_rot.dims(), [2, 16, 2, 64]);
    }

    #[test]
    fn test_rope_with_offset() {
        let device = Default::default();
        let rope = RoPEConfig::new(32, 128).init::<TestBackend>(&device);

        let q = Tensor::zeros([1, 1, 4, 32], &device);
        let k = Tensor::zeros([1, 1, 2, 32], &device);

        // Should not panic with valid offset
        let (q_rot, k_rot) = rope.forward(q, k, 64);
        assert_eq!(q_rot.dims(), [1, 1, 4, 32]);
        assert_eq!(k_rot.dims(), [1, 1, 2, 32]);
    }
}
