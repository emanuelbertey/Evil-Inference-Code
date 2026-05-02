// ─── Grouped Query Attention (GQA) ──────────────────────────────────────────
//
// Implements the full spectrum of multi-head attention variants:
//
//   ┌───────────────────────────────────────────────────────────────────────┐
//   │ Configuration          │ num_heads │ num_kv_groups │ Description     │
//   ├───────────────────────────────────────────────────────────────────────┤
//   │ Multi-Head (MHA)       │     8     │       8       │ Standard        │
//   │ Multi-Query (MQA)      │     8     │       1       │ Fastest         │
//   │ Grouped Query (GQA)    │     8     │       2       │ Best trade-off  │
//   └───────────────────────────────────────────────────────────────────────┘
//
// Pipeline:
//   x → QKV_proj → RoPE(Q, K) → repeat_kv(K, V) → Scaled Dot-Product
//     → softmax → dropout → V_attn → O_proj → output
//
// Features:
//   - Causal masking (autoregressive)
//   - Optional attention logit soft-capping
//   - Dropout on attention weights
//   - KV cache support for autoregressive generation

use burn::prelude::*;
use burn::config::Config;
use burn::module::Module;
use burn::nn::Dropout;

use super::rope::{RoPE, RoPEConfig};
use super::heads::{QKVProjection, OutputProjection, HeadConfig, repeat_kv};

// ─── Attention Config ───────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct AttentionConfig {
    /// Model embedding dimension
    pub d_model: usize,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of key/value groups (GQA). Must divide num_heads.
    #[config(default = 0)]
    pub num_kv_groups: usize,
    /// Optional override for head dimension (default: d_model / num_heads)
    pub head_dim: Option<usize>,
    /// Maximum sequence length for RoPE
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// RoPE base frequency
    #[config(default = 10000.0)]
    pub rope_base: f64,
    /// RoPE scaling factor for context extension
    #[config(default = 1.0)]
    pub rope_scaling: f64,
    /// Whether to use causal (autoregressive) masking
    #[config(default = true)]
    pub causal: bool,
    /// Attention dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Optional soft-capping for attention logits (e.g. 50.0 in Gemma2)
    pub attn_logit_cap: Option<f64>,
    /// Whether projections use bias
    #[config(default = false)]
    pub bias: bool,
}

impl AttentionConfig {
    fn effective_kv_groups(&self) -> usize {
        if self.num_kv_groups == 0 {
            self.num_heads // Default to MHA
        } else {
            self.num_kv_groups
        }
    }

    fn effective_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.d_model / self.num_heads)
    }
}

// ─── Attention Module ───────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    /// QKV projection (handles asymmetric Q vs KV dimensions)
    pub qkv: QKVProjection<B>,
    /// Output projection: concat(heads) → d_model
    pub o_proj: OutputProjection<B>,
    /// Rotary position embeddings
    pub rope: RoPE<B>,
    /// Attention dropout
    pub dropout: Dropout,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV groups
    pub num_kv_groups: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to apply causal mask
    pub causal: bool,
    /// Optional soft-cap for logits
    pub attn_logit_cap: Option<f64>,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let kv_groups = self.effective_kv_groups();
        let hd = self.effective_head_dim();

        let head_config = HeadConfig {
            d_model: self.d_model,
            num_heads: self.num_heads,
            num_kv_groups: kv_groups,
            head_dim: Some(hd),
            bias: self.bias,
        };

        let qkv = head_config.init_qkv(device);
        let o_proj = head_config.init_output(device);

        let rope = RoPEConfig {
            head_dim: hd,
            max_seq_len: self.max_seq_len,
            base: self.rope_base,
            scaling_factor: self.rope_scaling,
        }.init(device);

        let dropout = burn::nn::DropoutConfig::new(self.dropout).init();

        Attention {
            qkv,
            o_proj,
            rope,
            dropout,
            num_heads: self.num_heads,
            num_kv_groups: kv_groups,
            head_dim: hd,
            causal: self.causal,
            attn_logit_cap: self.attn_logit_cap,
        }
    }
}

impl<B: Backend> Attention<B> {
    /// Full attention forward pass.
    ///
    /// Input:  x of shape (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    ///
    /// `offset`: position offset for RoPE (0 during training, increments during generation)
    pub fn forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_batch, seq_len, _d] = x.dims();

        // 1. Project to Q, K, V with per-head shapes
        let (q, k, v) = self.qkv.forward(x);
        // q: (B, S, num_heads, head_dim)
        // k: (B, S, num_kv_groups, head_dim)
        // v: (B, S, num_kv_groups, head_dim)

        // 2. Apply RoPE to Q and K
        let (q, k) = self.rope.forward(q, k, offset);

        // 3. Repeat KV groups to match num_heads (GQA broadcast)
        let k = repeat_kv(k, self.num_heads, self.num_kv_groups);
        let v = repeat_kv(v, self.num_heads, self.num_kv_groups);
        // Now k, v: (B, S, num_heads, head_dim)

        // 4. Transpose for attention: (B, num_heads, S, head_dim)
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // 5. Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        // scores: (B, num_heads, S, S)

        // 6. Optional logit soft-capping (Gemma2 style)
        if let Some(cap) = self.attn_logit_cap {
            scores = scores.div_scalar(cap).tanh().mul_scalar(cap);
        }

        // 7. Causal mask
        if self.causal && seq_len > 1 {
            scores = self.apply_causal_mask(scores, seq_len);
        }

        // 8. Softmax + Dropout
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // 9. Weighted sum of values
        let attn_output = attn_weights.matmul(v);
        // attn_output: (B, num_heads, S, head_dim)

        // 10. Transpose back and project output
        let attn_output = attn_output.swap_dims(1, 2);
        // attn_output: (B, S, num_heads, head_dim)

        self.o_proj.forward(attn_output)
    }

    /// Apply lower-triangular causal mask to attention scores.
    ///
    /// Sets future positions to -infinity so softmax assigns them zero weight.
    fn apply_causal_mask(&self, scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let device = scores.device();

        // Build upper-triangular mask (1 = masked position)
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        let mask = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(mask_data, [seq_len, seq_len]),
                &device,
            )
            .unsqueeze_dim::<3>(0)   // (1, S, S)
            .unsqueeze_dim::<4>(0);  // (1, 1, S, S)

        // Replace masked positions with large negative value
        let neg_inf = mask.clone() * (-1e9);
        let keep = (mask * (-1.0)) + 1.0; // Invert: 0→1, 1→0

        scores * keep + neg_inf
    }
}
