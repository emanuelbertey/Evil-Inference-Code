// ─── Stress Test: Transformer FP32 vs BitLinear Training ──────────────
// 4, 5, y 16 capas con gradiente tracking para detectar vanishing/exploding.

use std::error::Error;
use std::time::Instant;

use burn::prelude::*;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::backend::AutodiffBackend;
use burn_flex::Flex;
use burn_autodiff::Autodiff;

use xlstm::blocks::trasformer::layer::{TransformerConfig, TransformerLayerConfig, Transformer};
use xlstm::blocks::trasformer::feedforward::FeedForwardBlock;
use xlstm::blocks::trasformer_bit::model::{
    TransformerBitLinearLM, BitLinearTransformerStack, BitLinearTransformerLayer,
    BitLinearQKVProjection, BitLinearOutputProjection, BitLinearSwiGLUFeedForward, BitLinearRMSNorm,
};
use xlstm::blocks::bitlinear::layer::BitLinearConfig;

use rand::Rng;

type MyBackend = Autodiff<Flex<f32>>;

const D_MODEL: usize = 128;
const NUM_HEADS: usize = 4;
const NUM_KV_GROUPS: usize = 2;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS;
const FFN_FACTOR: f64 = 4.0;
const SEQ_LEN: usize = 64;
const BATCH_SIZE: usize = 4;
const VOCAB_SIZE: usize = 2048;
const TRAIN_STEPS: usize = 50;
const LOG_INTERVAL: usize = 10;
const LR: f64 = 3e-4;
const MAX_SEQ_LEN: usize = 2048;

fn ffn_intermediate_dim() -> usize {
    ((FFN_FACTOR * D_MODEL as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64
}

// ─── Normal Transformer LM (FP32) ─────────────────────────────────────

#[derive(Module, Debug)]
pub struct NormalTransformerLM<B: Backend> {
    embedding: burn::nn::Embedding<B>,
    transformer: Transformer<B>,
    head: burn::nn::Linear<B>,
}

impl<B: Backend> NormalTransformerLM<B> {
    fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer.forward(x, 0);
        self.head.forward(x)
    }
}

fn build_normal_lm(device: &<MyBackend as burn::tensor::backend::BackendTypes>::Device, num_layers: usize) -> NormalTransformerLM<MyBackend> {
    let transformer: Transformer<MyBackend> = TransformerConfig {
        num_layers,
        layer: TransformerLayerConfig {
            d_model: D_MODEL,
            num_heads: NUM_HEADS,
            num_kv_groups: NUM_KV_GROUPS,
            head_dim: Some(HEAD_DIM),
            ffn_expansion: FFN_FACTOR,
            use_swiglu: true,
            max_seq_len: MAX_SEQ_LEN,
            rope_base: 10000.0,
            rope_scaling: 1.0,
            causal: true,
            attn_dropout: 0.0,
            ffn_dropout: 0.0,
            residual_dropout: 0.0,
            attn_logit_cap: None,
            bias: false,
            norm_eps: 1e-5,
            ffn_round_to: 64,
        },
    }.init(device);

    NormalTransformerLM {
        embedding: EmbeddingConfig::new(VOCAB_SIZE, D_MODEL).init(device),
        transformer,
        head: LinearConfig::new(D_MODEL, VOCAB_SIZE).with_bias(false).init(device),
    }
}

// ─── BitLinear Transformer LM ─────────────────────────────────────────

fn build_bit_lm(device: &<MyBackend as burn::tensor::backend::BackendTypes>::Device, num_layers: usize) -> TransformerBitLinearLM<MyBackend> {
    let ffn_dim = ffn_intermediate_dim();

    let layers: Vec<BitLinearTransformerLayer<MyBackend>> = (0..num_layers).map(|_| {
        BitLinearTransformerLayer {
            attn_norm: BitLinearRMSNorm::new(D_MODEL, 1e-5, device),
            qkv: BitLinearQKVProjection {
                q_proj: BitLinearConfig { in_features: D_MODEL, out_features: NUM_HEADS * HEAD_DIM, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                k_proj: BitLinearConfig { in_features: D_MODEL, out_features: NUM_KV_GROUPS * HEAD_DIM, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                v_proj: BitLinearConfig { in_features: D_MODEL, out_features: NUM_KV_GROUPS * HEAD_DIM, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                num_heads: NUM_HEADS,
                num_kv_groups: NUM_KV_GROUPS,
                head_dim: HEAD_DIM,
            },
            o_proj: BitLinearOutputProjection {
                o_proj: BitLinearConfig { in_features: NUM_HEADS * HEAD_DIM, out_features: D_MODEL, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                num_heads: NUM_HEADS,
                head_dim: HEAD_DIM,
            },
            ffn_norm: BitLinearRMSNorm::new(D_MODEL, 1e-5, device),
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: BitLinearConfig { in_features: D_MODEL, out_features: 2 * ffn_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                down_proj: BitLinearConfig { in_features: ffn_dim, out_features: D_MODEL, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                dropout: burn::nn::DropoutConfig::new(0.0).init(),
                intermediate_dim: ffn_dim,
            },
            residual_dropout: burn::nn::DropoutConfig::new(0.0).init(),
        }
    }).collect();

    TransformerBitLinearLM {
        embedding: EmbeddingConfig::new(VOCAB_SIZE, D_MODEL).init(device),
        transformer: BitLinearTransformerStack {
            final_norm: BitLinearRMSNorm::new(D_MODEL, 1e-5, device),
            num_layers,
            d_model: D_MODEL,
            layers,
        },
        head: BitLinearConfig { in_features: D_MODEL, out_features: VOCAB_SIZE, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: false }.init(device),
        vocab_size: VOCAB_SIZE,
        d_model: D_MODEL,
        num_layers,
    }
}

// ─── Gradient Helpers ─────────────────────────────────────────────────

/// Norma L2 de un gradiente. `w.grad()` requiere AutodiffBackend.
fn grad_norm<B: AutodiffBackend, const D: usize>(w: &burn::module::Param<Tensor<B, D>>, grads: &B::Gradients) -> Option<f32> {
    w.grad(grads).map(|g| g.powf_scalar(2.0).sum().sqrt().into_scalar().elem())
}

fn log_layer_grad_norms(layer_idx: usize, g_q: Option<f32>, g_o: Option<f32>, g_gate: Option<f32>, g_down: Option<f32>, g_attn_norm: Option<f32>, g_ffn_norm: Option<f32>) {
    let q_str = g_q.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    let o_str = g_o.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    let gate_str = g_gate.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    let down_str = g_down.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    let an_str = g_attn_norm.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    let fn_str = g_ffn_norm.map_or("N/A".to_string(), |v| format!("{:.2e}", v));
    println!("             Layer {:>2}: q_proj={:>10} o_proj={:>10} gate={:>10} down={:>10}  attn_norm={:>10} ffn_norm={:>10}",
             layer_idx, q_str, o_str, gate_str, down_str, an_str, fn_str);
}

// ─── Normal Transformer Gradient Logging ─────────────────────────────

fn log_grad_stats_normal(model: &NormalTransformerLM<MyBackend>, grads: &<MyBackend as AutodiffBackend>::Gradients, step: usize, num_layers: usize) {
    let emb_norm = grad_norm(&model.embedding.weight, grads);
    let head_norm = grad_norm(&model.head.weight, grads);
    let final_norm_norm = grad_norm(&model.transformer.final_norm.weight, grads);

    println!("       Gradients at step {:>2}: emb={:.2e}  head={:.2e}  final_norm={:.2e}",
             step, emb_norm.unwrap_or(0.0), head_norm.unwrap_or(0.0), final_norm_norm.unwrap_or(0.0));

    let layers_to_show: Vec<usize> = if num_layers <= 4 {
        (0..num_layers).collect()
    } else {
        vec![0, num_layers / 2, num_layers - 1]
    };

    for &li in &layers_to_show {
        let layer = &model.transformer.layers[li];
        let g_q = grad_norm(&layer.attention.qkv.q_proj.weight, grads);
        let g_o = grad_norm(&layer.attention.o_proj.o_proj.weight, grads);
        let g_attn_norm = grad_norm(&layer.attn_norm.weight, grads);
        let g_ffn_norm = grad_norm(&layer.ffn_norm.weight, grads);

        let (g_gate, g_down) = match &layer.ffn {
            FeedForwardBlock::SwiGLU(ffn) => (
                grad_norm(&ffn.gate_up_proj.weight, grads),
                grad_norm(&ffn.down_proj.weight, grads),
            ),
            FeedForwardBlock::Standard(ffn) => (
                grad_norm(&ffn.up_proj.weight, grads),
                grad_norm(&ffn.down_proj.weight, grads),
            ),
        };
        log_layer_grad_norms(li, g_q, g_o, g_gate, g_down, g_attn_norm, g_ffn_norm);
    }
}

// ─── BitLinear Transformer Gradient Logging ──────────────────────────

fn bit_grad_norm(w: &Option<burn::module::Param<Tensor<MyBackend, 2>>>, grads: &<MyBackend as AutodiffBackend>::Gradients) -> Option<f32> {
    w.as_ref().and_then(|p| p.grad(grads).map(|g| g.powf_scalar(2.0).sum().sqrt().into_scalar().elem()))
}

fn log_grad_stats_bit(model: &TransformerBitLinearLM<MyBackend>, grads: &<MyBackend as AutodiffBackend>::Gradients, step: usize, num_layers: usize) {
    let emb_norm = grad_norm(&model.embedding.weight, grads);
    let head_norm = bit_grad_norm(&model.head.weight, grads);
    let final_norm_norm = grad_norm(&model.transformer.final_norm.weight, grads);

    println!("       Gradients at step {:>2}: emb={:.2e}  head={:.2e}  final_norm={:.2e}",
             step, emb_norm.unwrap_or(0.0), head_norm.unwrap_or(0.0), final_norm_norm.unwrap_or(0.0));

    let layers_to_show: Vec<usize> = if num_layers <= 4 {
        (0..num_layers).collect()
    } else {
        vec![0, num_layers / 2, num_layers - 1]
    };

    for &li in &layers_to_show {
        let layer = &model.transformer.layers[li];
        let g_q = bit_grad_norm(&layer.qkv.q_proj.weight, grads);
        let g_o = bit_grad_norm(&layer.o_proj.o_proj.weight, grads);
        let g_gate = bit_grad_norm(&layer.ffn.gate_up_proj.weight, grads);
        let g_down = bit_grad_norm(&layer.ffn.down_proj.weight, grads);
        let g_attn_norm = grad_norm(&layer.attn_norm.weight, grads);
        let g_ffn_norm = grad_norm(&layer.ffn_norm.weight, grads);
        log_layer_grad_norms(li, g_q, g_o, g_gate, g_down, g_attn_norm, g_ffn_norm);
    }
}

// ─── Training Functions ──────────────────────────────────────────────

fn train_normal(
    input: Tensor<MyBackend, 2, Int>,
    targets: Tensor<MyBackend, 2, Int>,
    num_layers: usize,
    label: &str,
) -> (Vec<f64>, f64) {
    let device = input.device();
    let model = build_normal_lm(&device, num_layers);
    let mut model = model.clone();
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut losses = Vec::with_capacity(TRAIN_STEPS);

    let start = Instant::now();
    for step in 0..TRAIN_STEPS {
        let logits = model.forward(input.clone());
        let flat_logits = logits.reshape([BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        let flat_y = targets.clone().reshape([BATCH_SIZE * SEQ_LEN]);
        let loss = loss_fn.forward(flat_logits, flat_y);
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        losses.push(loss_val);
        let grads = loss.backward();

        if (step + 1) % LOG_INTERVAL == 0 || step == 0 {
            log_grad_stats_normal(&model, &grads, step + 1, num_layers);
            println!("       Normal FP32  step {:3}/{:3}  loss {:.6}",
                     step + 1, TRAIN_STEPS, loss_val);
        }

        let grads_p = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads_p);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tok_s = (TRAIN_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("     {} Normal FP32  total {:7.3}s  {:9.1} tok/s  loss {:.4} -> {:.4}",
             label, elapsed, tok_s, losses[0], losses[TRAIN_STEPS - 1]);
    (losses, tok_s)
}

fn train_bit(
    input: Tensor<MyBackend, 2, Int>,
    targets: Tensor<MyBackend, 2, Int>,
    num_layers: usize,
    label: &str,
) -> (Vec<f64>, f64) {
    let device = input.device();
    let model = build_bit_lm(&device, num_layers);
    let mut model = model.clone();
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut losses = Vec::with_capacity(TRAIN_STEPS);

    let start = Instant::now();
    for step in 0..TRAIN_STEPS {
        let logits = model.forward(input.clone());
        let flat_logits = logits.reshape([BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        let flat_y = targets.clone().reshape([BATCH_SIZE * SEQ_LEN]);
        let loss = loss_fn.forward(flat_logits, flat_y);
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        losses.push(loss_val);
        let grads = loss.backward();

        if (step + 1) % LOG_INTERVAL == 0 || step == 0 {
            log_grad_stats_bit(&model, &grads, step + 1, num_layers);
            println!("       BitLinear    step {:3}/{:3}  loss {:.6}",
                     step + 1, TRAIN_STEPS, loss_val);
        }

        let grads_p = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads_p);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tok_s = (TRAIN_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("     {} BitLinear    total {:7.3}s  {:9.1} tok/s  loss {:.4} -> {:.4}",
             label, elapsed, tok_s, losses[0], losses[TRAIN_STEPS - 1]);
    (losses, tok_s)
}

// ─── Single Test Runner ──────────────────────────────────────────────

fn run_stress_test(input: Tensor<MyBackend, 2, Int>, targets: Tensor<MyBackend, 2, Int>, num_layers: usize, label: &str) {
    println!();
    let title = format!("  {} ({} layers)", label, num_layers);
    println!("  ---- {title} ---------------------------");

    // Normal Transformer
    println!("\n     ---- Normal Transformer FP32 Training ({} steps) --------", TRAIN_STEPS);
    let (n_loss, n_tok_s) = train_normal(input.clone(), targets.clone(), num_layers, label);

    // BitLinear Transformer
    println!("\n     ---- BitLinear (1.58-bit) Training ({} steps) -----------", TRAIN_STEPS);
    let (b_loss, b_tok_s) = train_bit(input.clone(), targets.clone(), num_layers, label);

    // Summary
    let n_final = n_loss[TRAIN_STEPS - 1];
    let b_final = b_loss[TRAIN_STEPS - 1];
    println!("\n     ---- Results for {} ------------", label);
    println!("       Normal FP32:   {:.4} -> {:.4}   {:9.1} tok/s", n_loss[0], n_final, n_tok_s);
    println!("       BitLinear:     {:.4} -> {:.4}   {:9.1} tok/s", b_loss[0], b_final, b_tok_s);

    if b_final > n_final * 1.5 {
        let gap = (b_final - n_final) / n_final * 100.0;
        println!("       !! BitLinear {:.1}% worse than FP32", gap);
    } else if (b_final - n_final).abs() < 0.1 {
        println!("       OK BitLinear achieves near-FP32 loss!");
    } else {
        println!("       ~ BitLinear gap moderate");
    }
}

// ─── Main ────────────────────────────────────────────────────────────

pub fn stress_transformer_vs_bit_main() -> Result<(), Box<dyn Error>> {
    println!();
    println!("  ============================================================");
    println!("  STRESS TEST: Transformer FP32 vs BitLinear (128 dim, AdamW + grad clip 1.0)");
    println!("  ============================================================");
    println!("  d_model={}  heads={}  kv_groups={}  head_dim={}", D_MODEL, NUM_HEADS, NUM_KV_GROUPS, HEAD_DIM);
    println!("  seq_len={}  batch={}  vocab={}  ffn_dim={}", SEQ_LEN, BATCH_SIZE, VOCAB_SIZE, ffn_intermediate_dim());
    println!("  steps={}  log every {}  Adam LR={}", TRAIN_STEPS, LOG_INTERVAL, LR);
    println!();

    let device = Default::default();

    let mut rng = rand::rng();
    let input_ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN)
        .map(|_| (rng.random::<f32>() * VOCAB_SIZE as f32) as i64).collect();
    let target_ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN)
        .map(|_| (rng.random::<f32>() * VOCAB_SIZE as f32) as i64).collect();
    let input = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::new(input_ids, [BATCH_SIZE, SEQ_LEN]), &device);
    let targets = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::new(target_ids, [BATCH_SIZE, SEQ_LEN]), &device);

    println!("  Building models...");

    run_stress_test(input.clone(), targets.clone(), 4, "Test A (4 layers)");
    run_stress_test(input.clone(), targets.clone(), 5, "Test B (5 layers)");
    run_stress_test(input.clone(), targets.clone(), 16, "Test C (16 layers)");

    println!();
    Ok(())
}

fn main() {
    if let Err(e) = stress_transformer_vs_bit_main() {
        eprintln!("Error: {}", e);
    }
}
