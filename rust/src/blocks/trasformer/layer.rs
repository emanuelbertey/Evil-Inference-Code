// ─── Transformer Layer & Stack ──────────────────────────────────────────────
//
// Full Transformer decoder layer with Pre-Norm residual connections:
//
//   x → RMSNorm → Attention(GQA + RoPE) → +residual
//     → RMSNorm → FeedForward(SwiGLU)    → +residual → output
//
// The TransformerStack chains N identical layers with a final RMSNorm.
//
// Configuration supports all major architecture styles:
//   - GPT-style: causal=true, use_swiglu=false
//   - LLaMA-style: causal=true, use_swiglu=true, GQA
//   - Gemma-style: causal=true, use_swiglu=true, GQA, logit_cap
//   - Mistral-style: causal=true, use_swiglu=true, GQA, sliding window (TODO)

use burn::prelude::*;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Dropout;

use super::attention::{Attention, AttentionConfig};
use super::feedforward::{FeedForwardBlock, FeedForwardConfig};

// ─── RMSNorm (local copy for the transformer module) ────────────────────────

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    /// Forward for 3D: (B, S, D) → (B, S, D)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = x.clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp_min(self.eps as f32);
        let normed = x / rms;
        let w = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        normed * w
    }
}

// ─── Transformer Layer Config ───────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TransformerLayerConfig {
    /// Model embedding dimension
    pub d_model: usize,
    /// Number of query attention heads
    pub num_heads: usize,
    /// Number of key/value groups (0 = same as num_heads, i.e. standard MHA)
    #[config(default = 0)]
    pub num_kv_groups: usize,
    /// Optional head dimension (default: d_model / num_heads)
    pub head_dim: Option<usize>,
    /// FFN expansion factor
    #[config(default = 4.0)]
    pub ffn_expansion: f64,
    /// Use SwiGLU in FFN
    #[config(default = true)]
    pub use_swiglu: bool,
    /// Maximum sequence length for RoPE
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// RoPE base frequency
    #[config(default = 10000.0)]
    pub rope_base: f64,
    /// RoPE scaling factor
    #[config(default = 1.0)]
    pub rope_scaling: f64,
    /// Whether to use causal masking
    #[config(default = true)]
    pub causal: bool,
    /// Attention dropout rate
    #[config(default = 0.0)]
    pub attn_dropout: f64,
    /// FFN dropout rate
    #[config(default = 0.0)]
    pub ffn_dropout: f64,
    /// Residual dropout rate
    #[config(default = 0.0)]
    pub residual_dropout: f64,
    /// Optional attention logit soft-cap
    pub attn_logit_cap: Option<f64>,
    /// Whether linear projections use bias
    #[config(default = false)]
    pub bias: bool,
    /// RMSNorm epsilon
    #[config(default = 1e-5)]
    pub norm_eps: f64,
    /// Round FFN intermediate dim to this multiple
    #[config(default = 64)]
    pub ffn_round_to: usize,
}

// ─── Transformer Layer ──────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    /// Pre-attention normalization
    pub attn_norm: RMSNorm<B>,
    /// Grouped Query Attention with RoPE
    pub attention: Attention<B>,
    /// Pre-FFN normalization
    pub ffn_norm: RMSNorm<B>,
    /// Feed-forward network (Standard or SwiGLU)
    pub ffn: FeedForwardBlock<B>,
    /// Residual dropout
    pub residual_dropout: Dropout,
}

impl TransformerLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerLayer<B> {
        let attn_config = AttentionConfig {
            d_model: self.d_model,
            num_heads: self.num_heads,
            num_kv_groups: self.num_kv_groups,
            head_dim: self.head_dim,
            max_seq_len: self.max_seq_len,
            rope_base: self.rope_base,
            rope_scaling: self.rope_scaling,
            causal: self.causal,
            dropout: self.attn_dropout,
            attn_logit_cap: self.attn_logit_cap,
            bias: self.bias,
        };

        let ffn_config = FeedForwardConfig {
            d_model: self.d_model,
            expansion_factor: self.ffn_expansion,
            use_swiglu: self.use_swiglu,
            dropout: self.ffn_dropout,
            bias: self.bias,
            round_to_multiple: self.ffn_round_to,
        };

        TransformerLayer {
            attn_norm: RMSNorm::new(self.d_model, self.norm_eps, device),
            attention: attn_config.init(device),
            ffn_norm: RMSNorm::new(self.d_model, self.norm_eps, device),
            ffn: ffn_config.init(device),
            residual_dropout: burn::nn::DropoutConfig::new(self.residual_dropout).init(),
        }
    }
}

impl<B: Backend> TransformerLayer<B> {
    /// Forward pass with Pre-Norm residual connections.
    ///
    /// Input:  (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        // 1. Pre-Norm → Attention → Residual
        let residual = x.clone();
        let h = self.attn_norm.forward(x);
        let h = self.attention.forward(h, offset);
        let h = self.residual_dropout.forward(h);
        let x = residual + h;

        // 2. Pre-Norm → FFN → Residual
        let residual = x.clone();
        let h = self.ffn_norm.forward(x);
        let h = self.ffn.forward(h);
        let h = self.residual_dropout.forward(h);
        residual + h
    }
}

// ─── Transformer Stack Config ───────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Per-layer configuration
    pub layer: TransformerLayerConfig,
}

// ─── Transformer Stack ──────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    /// Stack of identical transformer layers
    pub layers: Vec<TransformerLayer<B>>,
    /// Final normalization before output head
    pub final_norm: RMSNorm<B>,
    pub num_layers: usize,
    pub d_model: usize,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let layers: Vec<TransformerLayer<B>> = (0..self.num_layers)
            .map(|_| self.layer.init(device))
            .collect();

        Transformer {
            layers,
            final_norm: RMSNorm::new(self.layer.d_model, self.layer.norm_eps, device),
            num_layers: self.num_layers,
            d_model: self.layer.d_model,
        }
    }
}

impl<B: Backend> Transformer<B> {
    /// Forward through all layers + final norm.
    ///
    /// Input:  (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&self, mut x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, offset);
        }
        self.final_norm.forward(x)
    }
}
