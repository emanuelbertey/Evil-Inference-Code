// ─── Multi-Head Projection Utilities ────────────────────────────────────────
//
// Provides head splitting/merging and per-head linear projections for
// Grouped Query Attention. Handles the asymmetry between Q heads and KV groups.
//
// Key concept:
//   num_heads = 8 (query heads)
//   num_kv_groups = 2 (key/value groups)
//   heads_per_group = num_heads / num_kv_groups = 4
//   → Each KV group is shared by 4 query heads
//
// Projection dimensions:
//   Q projection: d_model → num_heads * head_dim
//   K projection: d_model → num_kv_groups * head_dim
//   V projection: d_model → num_kv_groups * head_dim

use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;
use burn::nn::{Linear, LinearConfig};

// ─── Head Config ────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct HeadConfig {
    /// Model embedding dimension
    pub d_model: usize,
    /// Number of query attention heads
    pub num_heads: usize,
    /// Number of key/value groups (for GQA). Must divide num_heads evenly.
    /// - num_kv_groups == num_heads → standard Multi-Head Attention
    /// - num_kv_groups == 1 → Multi-Query Attention
    /// - 1 < num_kv_groups < num_heads → Grouped Query Attention
    pub num_kv_groups: usize,
    /// Optional head dimension override (default: d_model / num_heads)
    pub head_dim: Option<usize>,
    /// Whether projections use bias
    #[config(default = false)]
    pub bias: bool,
}

impl HeadConfig {
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.d_model / self.num_heads)
    }

    pub fn heads_per_group(&self) -> usize {
        assert!(
            self.num_heads % self.num_kv_groups == 0,
            "num_heads({}) must be divisible by num_kv_groups({})",
            self.num_heads, self.num_kv_groups
        );
        self.num_heads / self.num_kv_groups
    }
}

// ─── QKV Projection ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct QKVProjection<B: Backend> {
    /// Projects input to Q space: d_model → num_heads * head_dim
    pub q_proj: Linear<B>,
    /// Projects input to K space: d_model → num_kv_groups * head_dim
    pub k_proj: Linear<B>,
    /// Projects input to V space: d_model → num_kv_groups * head_dim
    pub v_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
}

impl HeadConfig {
    pub fn init_qkv<B: Backend>(&self, device: &B::Device) -> QKVProjection<B> {
        let hd = self.effective_head_dim();

        let q_proj = LinearConfig::new(self.d_model, self.num_heads * hd)
            .with_bias(self.bias)
            .init(device);
        let k_proj = LinearConfig::new(self.d_model, self.num_kv_groups * hd)
            .with_bias(self.bias)
            .init(device);
        let v_proj = LinearConfig::new(self.d_model, self.num_kv_groups * hd)
            .with_bias(self.bias)
            .init(device);

        QKVProjection {
            q_proj,
            k_proj,
            v_proj,
            num_heads: self.num_heads,
            num_kv_groups: self.num_kv_groups,
            head_dim: hd,
        }
    }
}

impl<B: Backend> QKVProjection<B> {
    /// Project input to Q, K, V and split into heads.
    ///
    /// Input: (batch, seq_len, d_model)
    /// Returns:
    ///   q: (batch, seq_len, num_heads, head_dim)
    ///   k: (batch, seq_len, num_kv_groups, head_dim)
    ///   v: (batch, seq_len, num_kv_groups, head_dim)
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();

        let q = self.q_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);

        (q, k, v)
    }
}

// ─── Output Projection ─────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct OutputProjection<B: Backend> {
    /// Projects concatenated heads back: num_heads * head_dim → d_model
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl HeadConfig {
    pub fn init_output<B: Backend>(&self, device: &B::Device) -> OutputProjection<B> {
        let hd = self.effective_head_dim();
        let o_proj = LinearConfig::new(self.num_heads * hd, self.d_model)
            .with_bias(self.bias)
            .init(device);

        OutputProjection {
            o_proj,
            num_heads: self.num_heads,
            head_dim: hd,
        }
    }
}

impl<B: Backend> OutputProjection<B> {
    /// Merge heads and project output.
    ///
    /// Input: (batch, seq_len, num_heads, head_dim)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        let x_merged = x.reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(x_merged)
    }
}

// ─── KV Repeat for GQA ─────────────────────────────────────────────────────

/// Repeat KV heads to match the number of query heads.
///
/// When using GQA with fewer KV groups than Q heads, this function
/// broadcasts each KV group to serve multiple Q heads.
///
/// Input: (batch, seq_len, num_kv_groups, head_dim)
/// Output: (batch, seq_len, num_heads, head_dim)
///
/// Example: num_heads=8, num_kv_groups=2 → each KV group is repeated 4 times
pub fn repeat_kv<B: Backend>(
    x: Tensor<B, 4>,
    num_heads: usize,
    num_kv_groups: usize,
) -> Tensor<B, 4> {
    if num_kv_groups == num_heads {
        return x; // No repetition needed (standard MHA)
    }

    let repeats = num_heads / num_kv_groups;
    let [batch, seq_len, _nkv, head_dim] = x.dims();

    // (B, S, nkv, hd) → (B, S, nkv, 1, hd) → (B, S, nkv, repeats, hd) → (B, S, num_heads, hd)
    let x = x.unsqueeze_dim::<5>(3);
    let x = x.repeat_dim(3, repeats);
    x.reshape([batch, seq_len, num_heads, head_dim])
}
