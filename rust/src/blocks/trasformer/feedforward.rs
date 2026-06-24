// ─── FeedForward Networks ───────────────────────────────────────────────────
//
// Provides two variants of feed-forward networks for Transformer blocks:
//
//   1. Standard FFN:      x → Linear_up → GELU → Linear_down
//   2. SwiGLU FFN (GLU):  x → Linear_gate_up → split → SiLU(gate) * up → Linear_down
//
// SwiGLU (used in LLaMA, Gemma, Mistral) typically outperforms standard FFN
// at the cost of an extra projection. The gated variant uses:
//   hidden_dim = round_to_multiple(d_model * expansion * 2/3, 64)
//
// Both variants support dropout and configurable expansion factor.

use burn::prelude::*;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Dropout, SwiGlu, SwiGluConfig};

// ─── FFN Config ─────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    /// Model embedding dimension
    pub d_model: usize,
    /// Expansion factor (default: 4.0 for standard, ~2.67 effective for SwiGLU)
    #[config(default = 4.0)]
    pub expansion_factor: f64,
    /// Use SwiGLU gated activation (recommended for modern architectures)
    #[config(default = true)]
    pub use_swiglu: bool,
    /// Dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether to use bias in projections
    #[config(default = false)]
    pub bias: bool,
    /// Round intermediate dimension to this multiple (for hardware efficiency)
    #[config(default = 64)]
    pub round_to_multiple: usize,
}

impl FeedForwardConfig {
    /// Compute the intermediate (hidden) dimension.
    ///
    /// For SwiGLU: uses 2/3 factor since the gate takes half the projection.
    /// Always rounds up to the configured multiple for hardware efficiency.
    pub fn intermediate_dim(&self) -> usize {
        let raw = if self.use_swiglu {
            // SwiGLU convention: 2/3 * expansion * d_model (since gate consumes half)
            (self.expansion_factor * self.d_model as f64 * 2.0 / 3.0) as usize
        } else {
            (self.expansion_factor * self.d_model as f64) as usize
        };

        // Round up to nearest multiple
        let m = self.round_to_multiple;
        ((raw + m - 1) / m) * m
    }
}

// ─── Standard FFN ───────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
    pub dropout: Dropout,
}

// ─── SwiGLU FFN ─────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct SwiGLUFeedForward<B: Backend> {
    /// burn::nn::SwiGlu: gate + outer (no down projection)
    pub swiglu: SwiGlu<B>,
    /// Down projection: intermediate_dim → d_model
    pub down_proj: Linear<B>,
    pub dropout: Dropout,
    pub intermediate_dim: usize,
}

// ─── Unified FFN enum ───────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub enum FeedForwardBlock<B: Backend> {
    Standard(FeedForward<B>),
    SwiGLU(SwiGLUFeedForward<B>),
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForwardBlock<B> {
        let inter_dim = self.intermediate_dim();
        let dropout = burn::nn::DropoutConfig::new(self.dropout).init();

        if self.use_swiglu {
            let swiglu = SwiGluConfig::new(self.d_model, inter_dim)
                .with_bias(self.bias)
                .init(device);
            let down_proj = LinearConfig::new(inter_dim, self.d_model)
                .with_bias(self.bias)
                .init(device);

            FeedForwardBlock::SwiGLU(SwiGLUFeedForward {
                swiglu,
                down_proj,
                dropout,
                intermediate_dim: inter_dim,
            })
        } else {
            let up_proj = LinearConfig::new(self.d_model, inter_dim)
                .with_bias(self.bias)
                .init(device);
            let down_proj = LinearConfig::new(inter_dim, self.d_model)
                .with_bias(self.bias)
                .init(device);

            FeedForwardBlock::Standard(FeedForward {
                up_proj,
                down_proj,
                dropout,
            })
        }
    }
}

impl<B: Backend> FeedForwardBlock<B> {
    /// Forward pass: (batch, seq_len, d_model) → (batch, seq_len, d_model)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            FeedForwardBlock::Standard(ffn) => ffn.forward(x),
            FeedForwardBlock::SwiGLU(ffn) => ffn.forward(x),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    /// Standard FFN: x → up → GELU → dropout → down
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.up_proj.forward(x);
        let h = burn::tensor::activation::gelu(h);
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }
}

impl<B: Backend> SwiGLUFeedForward<B> {
    /// SwiGLU FFN: x → SwiGlu(gate + outer) → dropout → down
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.swiglu.forward(x);
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }
}
