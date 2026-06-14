use burn::prelude::*;
use burn::module::{Module, Param};
use burn::config::Config;
use burn::tensor::activation::{softmax, silu};

// ─── BitLinear Mode ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BitLinearMode {
    /// Training with STE: forward uses ternary, backward updates full-precision shadow weights.
    Training,
    /// Inference using only {-1, 0, +1} weights.
    Ternary,
    /// Inference using full-precision weights (like standard Linear).
    Full16,
}

// ─── RMSNorm (Sub-LN) ────────────────────────────────────────────────────────

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

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = x.clone()
            .powf_scalar(2.0)
            .mean_dim(D - 1)
            .sqrt()
            .clamp_min(self.eps as f32);
        
        let normed = x / rms;
        
        // Use reshape to match rank D for broadcasting
        let mut shape = [1; D];
        shape[D - 1] = self.weight.val().dims()[0];
        let w = self.weight.val().reshape(shape);
        
        normed * w
    }
}

// ─── Quantization Functions ──────────────────────────────────────────────────

const GROUP_SIZE: usize = 128;

/// Per-group absmean ternary quantization with STE.
/// Each group of GROUP_SIZE weights gets its own scale.
fn quantize_weights_ternary<B: Backend>(w: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let orig_shape = w.dims();
    let numel = orig_shape[0] * orig_shape[1];

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

    // Per-group scale = mean(|w|)
    let scales = w_grouped.clone().abs().mean_dim(1).squeeze::<1>().clamp_min(1e-8);

    // Normalize per group, round, clip to ternary
    let scales_expanded = scales.clone().reshape([n_groups, 1]);
    let w_scaled = w_grouped.clone() / scales_expanded.clone();
    let w_ternary = w_scaled.clone().round().clamp(-1.0, 1.0);

    // STE
    let diff = w_ternary - w_scaled.clone();
    let w_ste = w_scaled + diff.detach();
    let w_dequant = w_ste * scales_expanded;

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

fn quantize_activations_8bit<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let q_b: f32 = 127.0;
    let gamma = x.clone().abs().max().clamp_min(1e-8);
    let gamma_val = gamma.into_scalar().elem::<f32>();

    let x_scaled = x.clone() * (q_b / gamma_val);
    let x_rounded = x_scaled.clone().round();
    let x_clamped = x_rounded.clamp(-q_b, q_b);

    let diff = x_clamped - x_scaled.clone();
    let x_quant_ste = x_scaled + diff.detach();

    let rescale = gamma_val / q_b;
    x_quant_ste * rescale
}

// ─── BitLinear Layer ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub rms_norm: RMSNorm<B>,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Config, Debug)]
pub struct BitLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = 1e-5)]
    pub rms_norm_eps: f64,
}

impl BitLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinear<B> {
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
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}

impl<B: Backend> BitLinear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, mode: BitLinearMode) -> Tensor<B, D> {
        let dims = x.dims();
        let batch_product: usize = dims[..D-1].iter().product();
        let d_in = dims[D-1];

        match mode {
            BitLinearMode::Full16 => {
                let x_flat = x.reshape([batch_product, d_in]);
                let out_flat = x_flat.matmul(self.weight.val().transpose());
                
                let mut out_shape = [0; D];
                out_shape[..D-1].copy_from_slice(&dims[..D-1]);
                out_shape[D-1] = self.out_features;
                
                let mut out = out_flat.reshape(out_shape);
                
                if let Some(bias) = &self.bias {
                    let mut b_shape = [1; D];
                    b_shape[D - 1] = self.out_features;
                    let b = bias.val().reshape(b_shape);
                    out = out + b;
                }
                out
            }
            BitLinearMode::Training | BitLinearMode::Ternary => {
                let x_norm = self.rms_norm.forward(x);
                let x_quant = quantize_activations_8bit(x_norm);
                let (w_quant, _scale) = quantize_weights_ternary(self.weight.val());

                let x_flat = x_quant.reshape([batch_product, d_in]);
                let out_flat = x_flat.matmul(w_quant.transpose());

                let mut out_shape = [0; D];
                out_shape[..D-1].copy_from_slice(&dims[..D-1]);
                out_shape[D-1] = self.out_features;
                
                let mut out = out_flat.reshape(out_shape);

                if let Some(bias) = &self.bias {
                    let mut b_shape = [1; D];
                    b_shape[D - 1] = self.out_features;
                    let b = bias.val().reshape(b_shape);
                    out = out + b;
                }
                out
            }
        }
    }
}

// ─── BitTransformer Components ───────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitAttention<B: Backend> {
    pub q_proj: BitLinear<B>,
    pub k_proj: BitLinear<B>,
    pub v_proj: BitLinear<B>,
    pub o_proj: BitLinear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> BitAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mode: BitLinearMode) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        
        let q = self.q_proj.forward(x.clone(), mode).reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone(), mode).reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let v = self.v_proj.forward(x, mode).reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        
        // Simple causal mask
        let scores = self.apply_mask(scores);
        
        let attn = softmax(scores, 3);
        let out = attn.matmul(v);
        
        let out = out.swap_dims(1, 2).reshape([batch, seq, self.num_heads * self.head_dim]);
        self.o_proj.forward(out, mode)
    }

    fn apply_mask(&self, scores: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, seq, _] = scores.dims();
        let device = scores.device();
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = 1.0;
            }
        }
        let mask = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(mask_data, [seq, seq]), &device)
            .unsqueeze::<4>();
        
        scores.mask_fill(mask.equal_elem(1.0), -1e9)
    }
}

#[derive(Module, Debug)]
pub struct BitFFN<B: Backend> {
    pub up: BitLinear<B>,
    pub gate: BitLinear<B>,
    pub down: BitLinear<B>,
}

impl<B: Backend> BitFFN<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mode: BitLinearMode) -> Tensor<B, 3> {
        let up = self.up.forward(x.clone(), mode);
        let gate = self.gate.forward(x, mode);
        let h = silu(gate) * up;
        self.down.forward(h, mode)
    }
}

#[derive(Module, Debug)]
pub struct BitTransformerLayer<B: Backend> {
    pub attention: BitAttention<B>,
    pub ffn: BitFFN<B>,
    pub norm1: RMSNorm<B>,
    pub norm2: RMSNorm<B>,
}

impl<B: Backend> BitTransformerLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mode: BitLinearMode) -> Tensor<B, 3> {
        let h = x.clone() + self.attention.forward(self.norm1.forward(x), mode);
        h.clone() + self.ffn.forward(self.norm2.forward(h), mode)
    }
}

#[derive(Module, Debug)]
pub struct BitTransformer<B: Backend> {
    pub layers: Vec<BitTransformerLayer<B>>,
    pub norm_final: RMSNorm<B>,
}

impl<B: Backend> BitTransformer<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>, mode: BitLinearMode) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, mode);
        }
        self.norm_final.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct BitTransformerLM<B: Backend> {
    pub embedding: burn::nn::Embedding<B>,
    pub transformer: BitTransformer<B>,
    pub head: BitLinear<B>,
    pub mode: BitLinearMode,
}

impl<B: Backend> BitTransformerLM<B> {
    pub fn forward(&self, x: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(x);
        let x = self.transformer.forward(x, self.mode);
        self.head.forward(x, self.mode)
    }

    pub fn print_info(&self) {
        println!("Model Mode: {:?}", self.mode);
        let (neg, zero, pos, total) = self.head.weight_distribution();
        println!("Head Layer Distribution: -1: {} | 0: {} | +1: {} | Total: {}", neg, zero, pos, total);
        let pct = (neg + pos) as f32 / total as f32 * 100.0;
        println!("Head Sparsity (non-zero): {:.2}%", pct);
    }
}

impl<B: Backend> BitLinear<B> {
    pub fn weight_distribution(&self) -> (usize, usize, usize, usize) {
        let w = self.weight.val();
        let dims = w.dims();
        let numel = dims[0] * dims[1];
        let (w_flat, pad_len) = if numel % GROUP_SIZE != 0 {
            let pad_len = GROUP_SIZE - (numel % GROUP_SIZE);
            let w_flat = w.clone().reshape([numel]);
            let zeros = Tensor::zeros([pad_len], &w.device());
            (Tensor::cat(vec![w_flat, zeros], 0), pad_len)
        } else {
            (w.reshape([numel]), 0)
        };
        let n_groups = w_flat.dims()[0] / GROUP_SIZE;
        let w_grouped = w_flat.reshape([n_groups, GROUP_SIZE]);
        let scales = w_grouped.clone().abs().mean_dim(1).squeeze::<1>().clamp_min(1e-8);
        let scales_expanded = scales.reshape([n_groups, 1]);
        let w_scaled = w_grouped / scales_expanded;
        let w_ternary = w_scaled.round().clamp(-1.0, 1.0);
        
        let data = w_ternary.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let total = if pad_len > 0 { numel } else { values.len() };
        let mut neg = 0;
        let mut zero = 0;
        let mut pos = 0;
        for &v in values {
            if v < -0.5 { neg += 1; }
            else if v > 0.5 { pos += 1; }
            else { zero += 1; }
        }
        (neg, zero, pos, total)
    }
}
