// ─── BitLinear b1.58 — Ternary {-1, 0, +1} Layer ───────────────────────────
//
// Based on "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
// (BitNet b1.58 — Microsoft Research, 2024)
//
// Key techniques implemented:
//   1. Weight Quantization (AbsMean): W_q = clamp(round(W / mean(|W|)), -1, 1)
//   2. Activation Quantization (AbsMax): X_q = clamp(round(X * Q_b / γ), -Q_b, Q_b)
//      where γ = max(|X|) and Q_b = 2^(bits-1) - 1 (= 127 for 8-bit)
//   3. Straight-Through Estimator (STE): gradients pass through quantization unchanged
//   4. RMSNorm before quantization (Sub-LN style) for training stability
//   5. Full-precision shadow weights maintained for optimizer updates
//
// Architecture (forward pass):
//   x → RMSNorm → ActivationQuant(8-bit) → MatMul(WeightQuant(W)) → rescale → output
//
// During training:
//   - Forward uses quantized weights/activations
//   - Backward uses STE: gradients flow to full-precision shadow weights
//   - Optimizer updates the full-precision weights
//
// During inference:
//   - Weights can be stored as 2-bit (ternary) for massive compression
//   - MatMul becomes additions/subtractions only (no multiplications needed)

use burn::prelude::*;
use burn::module::{Module, Param};
use burn::config::Config;

// ─── RMSNorm (Sub-LN) ──────────────────────────────────────────────────────
// Simplified normalization: RMSNorm(x) = x / sqrt(mean(x²) + ε)
// Used before activation quantization for training stability.
// Unlike LayerNorm, no centering (mean subtraction) — faster and works well
// with quantized networks.

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

    /// Forward for 2D input: (B, D) → (B, D)
    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)
        let rms = x.clone()
            .powf_scalar(2.0)
            .mean_dim(1)
            .sqrt()
            .clamp_min(self.eps as f32);
        let normed = x / rms;
        normed * self.weight.val().unsqueeze::<2>()
    }

    /// Forward for 3D input: (B, S, D) → (B, S, D)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = x.clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp_min(self.eps as f32);
        let normed = x / rms;
        normed * self.weight.val().unsqueeze::<2>().unsqueeze::<3>()
    }
}

// ─── Quantization Functions ─────────────────────────────────────────────────

/// Ternary weight quantization using AbsMean scaling + STE.
///
/// Forward:
///   scale = mean(|W|)
///   W_q = clamp(round(W / scale), -1, 1)     → values in {-1, 0, +1}
///   output = W_q * scale                       → rescaled for correct magnitude
///
/// Backward (STE):
///   ∂L/∂W = ∂L/∂W_q  (gradient passes through as if quantize = identity)
///
/// Implementation trick: `w_quant = w + (quantize(w) - w).detach()`
///   This ensures forward uses quantized values but backward flows to `w`.
fn quantize_weights_ternary<B: Backend>(w: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
    // AbsMean scale factor: scale = mean(|W|) + eps
    let abs_w = w.clone().abs();
    let scale = abs_w.mean(); // scalar → Tensor<B, 1> after reshape

    // Quantize: round(W / scale), clamped to [-1, 1]
    let scale_val = scale.clone().reshape([1, 1]);
    let w_scaled = w.clone() / (scale_val.clone() + 1e-8);
    let w_rounded = w_scaled.clone().round();
    let w_clamped = w_rounded.clamp(-1.0, 1.0);

    // STE trick: forward uses quantized, backward flows to full-precision w
    // w_ste = w + (w_quantized - w).detach()
    // In Burn, .detach() removes from autodiff graph
    let diff = w_clamped - w_scaled.clone();
    let w_quantized_ste = w_scaled + diff.detach();

    // Rescale back: W_dequant = W_q * scale
    let w_dequant = w_quantized_ste * scale_val;

    let scale_1d = scale.reshape([1]);
    (w_dequant, scale_1d)
}

/// 8-bit activation quantization using AbsMax scaling + STE.
///
/// Forward:
///   γ = max(|X|)
///   Q_b = 127  (for 8-bit signed integer range)
///   X_q = clamp(round(X * Q_b / γ), -Q_b, Q_b)
///   output = X_q * (γ / Q_b)   → rescaled to original magnitude
///
/// This enables efficient integer arithmetic during inference.
fn quantize_activations_8bit<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let q_b: f32 = 127.0; // 2^(8-1) - 1

    // γ = max(|x|) per-tensor, clamped to avoid division by zero
    let gamma = x.clone().abs().max().clamp_min(1e-8);

    // Scale to [-127, 127] range
    let x_scaled = x.clone() * (q_b / gamma.clone().into_scalar().elem::<f32>());
    let x_rounded = x_scaled.clone().round();
    let x_clamped = x_rounded.clamp(-q_b, q_b);

    // STE: x + (quantized - x).detach()
    let diff = x_clamped - x_scaled.clone();
    let x_quant_ste = x_scaled + diff.detach();

    // Dequantize: scale back
    let rescale = gamma.into_scalar().elem::<f32>() / q_b;
    x_quant_ste * rescale
}


// ─── BitLinear Configuration ────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct BitLinearConfig {
    /// Input feature dimension
    pub in_features: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Whether to include a bias term (standard practice: false for BitLinear)
    #[config(default = false)]
    pub bias: bool,
    /// Number of bits for activation quantization (default: 8)
    #[config(default = 8)]
    pub activation_bits: usize,
    /// Epsilon for RMSNorm
    #[config(default = 1e-5)]
    pub rms_norm_eps: f64,
}

// ─── BitLinear Layer ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    /// Full-precision shadow weights — the optimizer updates these.
    /// Forward uses quantized versions via STE.
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias (typically not used in BitLinear)
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// RMSNorm applied to activations before quantization (Sub-LN)
    pub rms_norm: RMSNorm<B>,
    /// Number of bits for activation quantization
    pub activation_bits: usize,
    /// Input dimension (for reference)
    pub in_features: usize,
    /// Output dimension (for reference)
    pub out_features: usize,
}

impl BitLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinear<B> {
        // Initialize with Kaiming uniform (fan-in) for better initial scale
        let k = 1.0 / (self.in_features as f64).sqrt();
        let weight = Tensor::random(
            [self.out_features, self.in_features],
            burn::tensor::Distribution::Uniform(-k, k),
            device,
        );

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.out_features], device)))
        } else {
            None
        };

        let rms_norm = RMSNorm::new(self.in_features, self.rms_norm_eps, device);

        BitLinear {
            weight: Param::from_tensor(weight),
            bias,
            rms_norm,
            activation_bits: self.activation_bits,
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}

impl<B: Backend> BitLinear<B> {
    /// Forward pass: (B, S, D_in) → (B, S, D_out)
    ///
    /// Pipeline:
    ///   1. RMSNorm on input (Sub-LN for stable quantization)
    ///   2. Quantize activations to 8-bit integers (STE)
    ///   3. Quantize weights to ternary {-1, 0, +1} (STE)
    ///   4. Matrix multiply: output = X_q @ W_q^T
    ///   5. Add bias if present
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _d_in] = x.dims();

        // 1. Sub-LN: RMSNorm before quantization
        let x_norm = self.rms_norm.forward(x);

        // 2. Quantize activations (8-bit with STE)
        let x_quant = quantize_activations_8bit(x_norm);

        // 3. Quantize weights (ternary with STE)
        let (w_quant, _scale) = quantize_weights_ternary(self.weight.val());

        // 4. MatMul: x_quant @ w_quant^T
        // Reshape for batch matmul: (B*S, D_in) @ (D_out, D_in)^T = (B*S, D_out)
        let x_flat = x_quant.reshape([batch * seq, self.in_features]);
        let output = x_flat.matmul(w_quant.transpose());
        let mut output = output.reshape([batch, seq, self.out_features]);

        // 5. Add bias
        if let Some(b) = &self.bias {
            output = output + b.val().unsqueeze::<2>().unsqueeze::<3>();
        }

        output
    }

    /// Forward pass for 2D input: (B, D_in) → (B, D_out)
    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // 1. Sub-LN: RMSNorm
        let x_norm = self.rms_norm.forward_2d(x);

        // 2. Quantize activations
        let x_quant = quantize_activations_8bit(x_norm);

        // 3. Quantize weights
        let (w_quant, _scale) = quantize_weights_ternary(self.weight.val());

        // 4. MatMul
        let mut output = x_quant.matmul(w_quant.transpose());

        // 5. Bias
        if let Some(b) = &self.bias {
            output = output + b.val().unsqueeze::<2>();
        }

        output
    }

    /// Get the current ternary weight values (for inspection/inference export).
    /// Returns the quantized weight matrix and the AbsMean scale factor.
    pub fn get_ternary_weights(&self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let w = self.weight.val();
        let abs_mean = w.clone().abs().mean().reshape([1]);
        let w_scaled = w / (abs_mean.clone().reshape([1, 1]) + 1e-8);
        let w_ternary = w_scaled.round().clamp(-1.0, 1.0);
        (w_ternary, abs_mean)
    }

    /// Count the distribution of {-1, 0, +1} in the current ternary weights.
    /// Returns (count_neg1, count_zero, count_pos1, total)
    pub fn weight_distribution(&self, device: &B::Device) -> (usize, usize, usize, usize) {
        let (w_ternary, _) = self.get_ternary_weights(device);
        let data = w_ternary.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let total = values.len();
        let mut neg = 0usize;
        let mut zero = 0usize;
        let mut pos = 0usize;
        for &v in values.iter() {
            if v < -0.5 {
                neg += 1;
            } else if v > 0.5 {
                pos += 1;
            } else {
                zero += 1;
            }
        }
        (neg, zero, pos, total)
    }
}

// ─── BitLinear FeedForward (Gated) ─────────────────────────────────────────
// Drop-in replacement for GatedFeedForward using BitLinear layers.
// Architecture: x → BitLinear_up(2*D') → split(gate, value) → GELU(gate)*value → BitLinear_down → y

#[derive(Config, Debug)]
pub struct BitLinearFeedForwardConfig {
    pub embedding_dim: usize,
    #[config(default = 1.3)]
    pub proj_factor: f64,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl BitLinearFeedForwardConfig {
    pub fn proj_up_dim(&self) -> usize {
        let raw = self.proj_factor * self.embedding_dim as f64;
        let multiple = 64usize;
        let mult = (raw / multiple as f64).ceil() as usize;
        mult * multiple
    }
}

#[derive(Module, Debug)]
pub struct BitLinearFeedForward<B: Backend> {
    pub proj_up: BitLinear<B>,
    pub proj_down: BitLinear<B>,
    pub dropout: burn::nn::Dropout,
    pub proj_up_dim: usize,
}

impl BitLinearFeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinearFeedForward<B> {
        let proj_up_dim = self.proj_up_dim();

        let proj_up = BitLinearConfig {
            in_features: self.embedding_dim,
            out_features: 2 * proj_up_dim,
            bias: self.bias,
            activation_bits: 8,
            rms_norm_eps: 1e-5,
        }.init(device);

        let proj_down = BitLinearConfig {
            in_features: proj_up_dim,
            out_features: self.embedding_dim,
            bias: self.bias,
            activation_bits: 8,
            rms_norm_eps: 1e-5,
        }.init(device);

        let dropout = burn::nn::DropoutConfig::new(self.dropout).init();

        BitLinearFeedForward {
            proj_up,
            proj_down,
            dropout,
            proj_up_dim,
        }
    }
}

impl<B: Backend> BitLinearFeedForward<B> {
    /// Forward: (B, S, D) → (B, S, D)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let up = self.proj_up.forward(x);
        let chunks = up.chunk(2, 2);
        let gate_preact = chunks[0].clone();
        let up_proj = chunks[1].clone();
        let gated = burn::tensor::activation::gelu(gate_preact) * up_proj;
        self.dropout.forward(self.proj_down.forward(gated))
    }
}
