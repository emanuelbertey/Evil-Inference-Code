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
use burn::tensor::TensorData;
use super::kernel::{I2SKernel, I2STile16Kernel, KernelKind};

// ─── Pure Raw Inference State ───────────────────────────────────────────────
/// State completely detached from Burn Tensors for maximum CPU inference speed
#[derive(Clone)]
pub struct BitLinearInferenceState {
    pub packed_w: Vec<u32>,
    pub scales: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: Option<Vec<f32>>,
    pub kernel: KernelKind,
}

impl BitLinearInferenceState {
    pub fn from_packed(packed_w: Vec<u32>, scales: Vec<f32>, in_features: usize, out_features: usize, kernel: KernelKind) -> Self {
        BitLinearInferenceState { packed_w, scales, in_features, out_features, bias: None, kernel }
    }

    pub fn forward_raw(&self, x_quant_data: &[f32], batch: usize) -> Vec<f32> {
        let mut out = match self.kernel {
            KernelKind::I2S => I2SKernel::forward_raw(
                x_quant_data,
                batch,
                &self.packed_w,
                &self.scales,
                self.out_features,
                self.in_features,
            ),
            KernelKind::Tile16 => I2STile16Kernel::forward_raw(
                x_quant_data,
                batch,
                &self.packed_w,
                &self.scales,
                self.out_features,
                self.in_features,
            ),
        };
        
        if let Some(b) = &self.bias {
            for batch_idx in 0..batch {
                let offset = batch_idx * self.out_features;
                for i in 0..self.out_features {
                    out[offset + i] += b[i];
                }
            }
        }
        
        out
    }

    pub fn forward_raw_i8(&self, x_i8: &[i8], batch: usize) -> Vec<f32> {
        let mut out = match self.kernel {
            KernelKind::I2S => I2SKernel::forward_raw_i8(
                x_i8, batch, &self.packed_w, &self.scales,
                self.out_features, self.in_features,
            ),
            KernelKind::Tile16 => I2STile16Kernel::forward_raw_i8(
                x_i8, batch, &self.packed_w, &self.scales,
                self.out_features, self.in_features,
            ),
        };

        if let Some(b) = &self.bias {
            for batch_idx in 0..batch {
                let offset = batch_idx * self.out_features;
                for i in 0..self.out_features {
                    out[offset + i] += b[i];
                }
            }
        }

        out
    }
}

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

const GROUP_SIZE: usize = 128;

/// Ternary weight quantization using Per-Group AbsMean scaling + STE.
///
/// Algorithm (BitNet b1.58):
///   1. Flatten weights, pad to multiple of GROUP_SIZE if needed
///   2. Reshape into (n_groups, GROUP_SIZE)
///   3. Per-group scale = mean(|w|), clamped to avoid div-by-zero
///   4. Normalize: w_scaled = w / scale per group
///   5. Round + clip to {-1, 0, +1}
///   6. STE: w_ste = w + (w_dequant - w).detach()
///
/// Returns (w_dequant, scales) where scales has shape [n_groups].
fn quantize_weights_ternary<B: Backend>(w: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let orig_shape = w.dims();
    let rows = orig_shape[0];
    let cols = orig_shape[1];
    let numel = rows * cols;

    // Flatten and pad if needed
    let (w_flat, pad_len) = if numel % GROUP_SIZE != 0 {
        let pad_len = GROUP_SIZE - (numel % GROUP_SIZE);
        let w_flat = w.clone().reshape([numel]);
        let zeros = Tensor::zeros([pad_len], &w.device());
        let w_padded = Tensor::cat(vec![w_flat, zeros], 0);
        (w_padded, pad_len)
    } else {
        (w.clone().reshape([numel]), 0)
    };

    let n_groups = w_flat.dims()[0] / GROUP_SIZE;
    let w_grouped = w_flat.reshape([n_groups, GROUP_SIZE]);

    // Per-group scale = mean(|w|), clamped to avoid div-by-zero
    let scales = w_grouped
        .clone()
        .abs()
        .mean_dim(1)
        .squeeze_dim::<1>(1)
        .clamp_min(1e-8); // [n_groups]

    // Normalize, round, clip to ternary
    let scales_expanded = scales.clone().reshape([n_groups, 1]); // [n_groups, 1]
    let w_scaled = w_grouped.clone() / scales_expanded.clone();
    let w_ternary = w_scaled.clone().round().clamp(-1.0, 1.0);

    // STE: forward uses quantized, backward flows to full-precision w
    let w_dequant_grouped = w_ternary * scales_expanded.clone();
    let diff = w_dequant_grouped - w_grouped.clone();
    let w_ste = w_grouped + diff.detach();
    let w_dequant = w_ste; // scales already applied once in w_dequant_grouped

    // Remove padding and reshape
    let w_dequant = if pad_len > 0 {
        w_dequant
            .reshape([n_groups * GROUP_SIZE])
            .narrow(0, 0, numel)
            .reshape(orig_shape)
    } else {
        w_dequant.reshape(orig_shape)
    };

    (w_dequant, scales)
}

/// 8-bit activation quantization using Per-Token AbsMax scaling + STE.
///
/// Per-token: one absmax scale per token (last dimension), not per-tensor.
/// This preserves dynamic range for each token independently.
///
/// For input (B, S, D): γ = max(|X|, dim=D) per token → shape (B, S, 1)
///   X_q = clamp(round(X * Q_b / γ), -Q_b, Q_b)
///   output = X_q * (γ / Q_b)
fn quantize_activations_8bit<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let q_b: f32 = 127.0;

    // Per-token absmax: max over last dim (d_model) → shape (B, S, 1)
    let gamma = x.clone().abs().max_dim(2).clamp_min(1e-8).unsqueeze::<3>();

    // Scale to [-127, 127] range per token
    let x_scaled = x.clone() * (q_b.clone() as f32 / gamma.clone());
    let x_rounded = x_scaled.clone().round();
    let x_clamped = x_rounded.clamp(-q_b, q_b);

    // STE: x + (quantized - x).detach()
    let diff = x_clamped - x_scaled.clone();
    let x_quant_ste = x_scaled + diff.detach();

    // Dequantize: scale back per token
    x_quant_ste * (gamma / q_b)
}

/// Quantize activations to i8 without dequantizing. For inference only.
/// Returns (i8 data, gamma per token for dequant of output).
fn quantize_to_i8(x_data: &[f32], d_model: usize, seq_len: usize) -> (Vec<i8>, Vec<f32>) {
    let mut x_i8 = vec![0i8; x_data.len()];
    let mut gammas = vec![0.0f32; seq_len];
    let q_b: f32 = 127.0;

    for t in 0..seq_len {
        let offset = t * d_model;
        let mut absmax = 0.0f32;
        for j in 0..d_model {
            let v = unsafe { *x_data.get_unchecked(offset + j) };
            if v > absmax { absmax = v; }
            else if -v > absmax { absmax = -v; }
        }
        if absmax < 1e-8 { absmax = 1e-8; }
        gammas[t] = absmax;
        let scale = q_b / absmax;
        for j in 0..d_model {
            let v = unsafe { *x_data.get_unchecked(offset + j) };
            let quantized = (v * scale).round().clamp(-127.0, 127.0) as i8;
            unsafe { *x_i8.get_unchecked_mut(offset + j) = quantized; }
        }
    }

    (x_i8, gammas)
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
    pub weight: Option<Param<Tensor<B, 2>>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub rms_norm: RMSNorm<B>,
    pub activation_bits: usize,
    pub in_features: usize,
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
            weight: Some(Param::from_tensor(weight)),
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
        let (w_quant, _scale) = quantize_weights_ternary(self.weight.as_ref().unwrap().val());

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
        let [batch, _d_in] = x.dims();

        // 1. Sub-LN: RMSNorm
        let x_norm = self.rms_norm.forward_2d(x);

        // 2. Quantize activations (reshape to 3D for per-token quant, then back)
        let x_3d = x_norm.reshape([batch, 1, self.in_features]);
        let x_quant_3d = quantize_activations_8bit(x_3d);
        let x_quant = x_quant_3d.reshape([batch, self.in_features]);

        // 3. Quantize weights
        let (w_quant, _scale) = quantize_weights_ternary(self.weight.as_ref().unwrap().val());

        // 4. MatMul
        let mut output = x_quant.matmul(w_quant.transpose());

        // 5. Bias
        if let Some(b) = &self.bias {
            output = output + b.val().unsqueeze::<2>();
        }

        output
    }

    /// Get the current ternary weight values (for inspection/inference export).
    /// Returns the quantized weight matrix and per-group AbsMean scales.
    /// scales has shape [n_groups] where n_groups = ceil(rows*cols / GROUP_SIZE).
    pub fn get_ternary_weights(&self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let w = self.weight.as_ref().unwrap().val();
        let dims = w.dims();
        let numel = dims[0] * dims[1];

        // Flatten and pad
        let (w_flat, pad_len) = if numel % GROUP_SIZE != 0 {
            let pad_len = GROUP_SIZE - (numel % GROUP_SIZE);
            let w_flat = w.reshape([numel]);
            let zeros = Tensor::zeros([pad_len], device);
            (Tensor::cat(vec![w_flat, zeros], 0), pad_len)
        } else {
            (w.reshape([numel]), 0)
        };

        let n_groups = w_flat.dims()[0] / GROUP_SIZE;
        let w_grouped = w_flat.reshape([n_groups, GROUP_SIZE]);

        // Per-group scale = mean(|w|)
        let scales = w_grouped.clone().abs().mean_dim(1).squeeze::<1>().clamp_min(1e-8);

        // Quantize per group
        let scales_expanded = scales.clone().reshape([n_groups, 1]);
        let w_scaled = w_grouped / scales_expanded;
        let w_ternary = w_scaled.round().clamp(-1.0, 1.0);

        // Remove padding
        let w_ternary = if pad_len > 0 {
            w_ternary
                .reshape([n_groups * GROUP_SIZE])
                .narrow(0, 0, numel)
                .reshape(dims)
        } else {
            w_ternary.reshape(dims)
        };

        (w_ternary, scales)
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

    pub fn release_weights(&mut self, _device: &B::Device) {
        self.weight = None;
        self.bias = None;
    }

    /// Export to a pure raw inference struct, completely detached from Burn
    pub fn export_inference_layer(&self, device: &B::Device, kernel: KernelKind) -> BitLinearInferenceState {
        let (w_ternary, scales_tensor) = self.get_ternary_weights(device);
        let scales = scales_tensor.into_data().as_slice::<f32>().unwrap().to_vec();
        let w_data = w_ternary.into_data();
        let w_slice = w_data.as_slice::<f32>().unwrap();
        
        let packed_w = I2SKernel::pack_weights(w_slice);
        
        let bias = self.bias.as_ref().map(|b| {
            b.val().into_data().as_slice::<f32>().unwrap().to_vec()
        });

        BitLinearInferenceState {
            packed_w,
            scales,
            in_features: self.in_features,
            out_features: self.out_features,
            bias,
            kernel,
        }
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, state: &BitLinearInferenceState) -> Tensor<B, 3> {
        let [batch, seq, _d_in] = x.dims();
        let device = x.device();
        let n = batch * seq;

        let x_norm = self.rms_norm.forward(x);
        let x_flat = x_norm.reshape([n, self.in_features]);
        let x_data = x_flat.into_data();
        let x_slice = x_data.as_slice::<f32>().unwrap();

        let (x_i8, gammas) = quantize_to_i8(x_slice, self.in_features, n);
        let out_data = state.forward_raw_i8(&x_i8, n);

        let q_b: f32 = 127.0;
        let mut result = out_data;
        for t in 0..n {
            let factor = gammas[t] / q_b;
            let base = t * state.out_features;
            for o in 0..state.out_features {
                result[base + o] *= factor;
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(result, [n, self.out_features]), &device)
            .reshape([batch, seq, self.out_features])
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
