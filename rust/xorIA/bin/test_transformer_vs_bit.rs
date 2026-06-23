// ─── Transformer FP32 vs BitLinear (1.58-bit) ─────────────────────────────
// Compara velocidad de inferencia y entrenamiento + loss entre:
//   1. Normal Transformer (FP32, SwiGLU, GQA, RoPE)
//   2. BitLinear Transformer (ternary {-1,0,+1}, 8-bit activaciones, STE)
//
// Mismas dimensiones: d_model=128, 3 layers, 4 heads, 2 KV groups, vocab=2048
//
// Uso: cargo run --release --bin test_transformer_vs_bit

use std::error::Error;
use std::time::Instant;

use burn::prelude::*;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn_flex::Flex;
use burn_autodiff::Autodiff;

use xlstm::blocks::trasformer::layer::{TransformerConfig, TransformerLayerConfig, Transformer};
use xlstm::blocks::trasformer_bit::model::{
    TransformerBitLinearLM, BitLinearTransformerStack, BitLinearTransformerLayer,
    BitLinearQKVProjection, BitLinearOutputProjection, BitLinearSwiGLUFeedForward, BitLinearRMSNorm,
};
use xlstm::blocks::bitlinear::layer::BitLinearConfig;

use rand::Rng;

type MyBackend = Autodiff<Flex<f32>>;

// ─── Config ───────────────────────────────────────────────────────────────
const D_MODEL: usize = 128;
const NUM_LAYERS: usize = 3;
const NUM_HEADS: usize = 4;
const NUM_KV_GROUPS: usize = 2;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS;
const FFN_FACTOR: f64 = 4.0;
const SEQ_LEN: usize = 64;
const BATCH_SIZE: usize = 4;
const VOCAB_SIZE: usize = 2048;
const WARMUP_STEPS: usize = 5;
const INFERENCE_STEPS: usize = 50;
const TRAIN_STEPS: usize = 30;
const LR: f64 = 3e-4;
const MAX_SEQ_LEN: usize = 2048;

fn ffn_intermediate_dim() -> usize {
    ((FFN_FACTOR * D_MODEL as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64
}

// ─── Normal Transformer LM (FP32) ─────────────────────────────────────────
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

fn build_normal_lm(device: &<MyBackend as burn::tensor::backend::BackendTypes>::Device) -> NormalTransformerLM<MyBackend> {
    let ffn_dim = ffn_intermediate_dim();
    let transformer: Transformer<MyBackend> = TransformerConfig {
        num_layers: NUM_LAYERS,
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

// ─── BitLinear Transformer LM ─────────────────────────────────────────────
fn build_bit_lm(device: &<MyBackend as burn::tensor::backend::BackendTypes>::Device) -> TransformerBitLinearLM<MyBackend> {
    let ffn_dim = ffn_intermediate_dim();

    let layers: Vec<BitLinearTransformerLayer<MyBackend>> = (0..NUM_LAYERS).map(|_| {
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
            num_layers: NUM_LAYERS,
            d_model: D_MODEL,
            layers,
        },
        head: BitLinearConfig { in_features: D_MODEL, out_features: VOCAB_SIZE, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
        vocab_size: VOCAB_SIZE,
        d_model: D_MODEL,
        num_layers: NUM_LAYERS,
    }
}

// ─── Parameter counter ────────────────────────────────────────────────────
fn estimate_params_normal() -> usize {
    let ffn_dim = ffn_intermediate_dim();
    let embed = VOCAB_SIZE * D_MODEL;                // embedding
    let head = D_MODEL * VOCAB_SIZE;                 // lm head
    let per_layer =
        D_MODEL * (NUM_HEADS + 2 * NUM_KV_GROUPS) * HEAD_DIM  // QKV proj
        + (NUM_HEADS * HEAD_DIM) * D_MODEL                    // O proj
        + D_MODEL * 2 * ffn_dim                               // gate_up
        + ffn_dim * D_MODEL;                                   // down
    embed + head + per_layer * NUM_LAYERS
}

fn estimate_params_bit() -> usize {
    estimate_params_normal() // same shadow weights, scales negligible
}

// ─── Benchmarks ───────────────────────────────────────────────────────────
fn benchmark_inference_normal(
    model: &NormalTransformerLM<MyBackend>,
    input: Tensor<MyBackend, 2, Int>,
) -> (f64, f64) {
    for _ in 0..WARMUP_STEPS {
        let _ = model.forward(input.clone());
    }
    let start = Instant::now();
    for _ in 0..INFERENCE_STEPS {
        let _ = model.forward(input.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = (elapsed / INFERENCE_STEPS as f64) * 1000.0;
    let tok_s = (INFERENCE_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("  Normal FP32       {:9.3} ms/forward  {:10.1} tok/s", avg_ms, tok_s);
    (avg_ms, tok_s)
}

fn benchmark_inference_bit(
    model: &TransformerBitLinearLM<MyBackend>,
    input: Tensor<MyBackend, 2, Int>,
) -> (f64, f64) {
    for _ in 0..WARMUP_STEPS {
        let _ = model.forward(input.clone());
    }
    let start = Instant::now();
    for _ in 0..INFERENCE_STEPS {
        let _ = model.forward(input.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = (elapsed / INFERENCE_STEPS as f64) * 1000.0;
    let tok_s = (INFERENCE_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("  BitLinear (1.58b) {:9.3} ms/forward  {:10.1} tok/s", avg_ms, tok_s);
    (avg_ms, tok_s)
}

fn benchmark_training_normal(
    model: &NormalTransformerLM<MyBackend>,
    input: Tensor<MyBackend, 2, Int>,
    targets: Tensor<MyBackend, 2, Int>,
) -> (Vec<f64>, f64) {
    let device = input.device();
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
        let grads_p = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads_p);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tok_s = (TRAIN_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("  Normal FP32       {:8.3}s total  {:10.1} tok/s  loss: {:.4} \u{2192} {:.4}",
             elapsed, tok_s, losses[0], losses[TRAIN_STEPS - 1]);
    (losses, tok_s)
}

fn benchmark_training_bit(
    model: &TransformerBitLinearLM<MyBackend>,
    input: Tensor<MyBackend, 2, Int>,
    targets: Tensor<MyBackend, 2, Int>,
) -> (Vec<f64>, f64) {
    let device = input.device();
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
        let grads_p = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads_p);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tok_s = (TRAIN_STEPS as f64 * BATCH_SIZE as f64 * SEQ_LEN as f64) / elapsed;
    println!("  BitLinear (1.58b) {:8.3}s total  {:10.1} tok/s  loss: {:.4} → {:.4}",
             elapsed, tok_s, losses[0], losses[TRAIN_STEPS - 1]);
    (losses, tok_s)
}

// ─── Main ─────────────────────────────────────────────────────────────────
pub fn test_transformer_vs_bit_main() -> Result<(), Box<dyn Error>> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Transformer FP32 vs BitLinear (1.58-bit ternario)          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  d_model={}  layers={}  heads={}  kv_groups={}  head_dim={}",
             D_MODEL, NUM_LAYERS, NUM_HEADS, NUM_KV_GROUPS, HEAD_DIM);
    println!("  seq_len={}  batch={}  vocab={}  ffn_dim={}",
             SEQ_LEN, BATCH_SIZE, VOCAB_SIZE, ffn_intermediate_dim());
    println!();
    println!("  Params Normal: ~{:.1}M  BitLinear: ~{:.1}M",
             estimate_params_normal() as f64 / 1e6,
             estimate_params_bit() as f64 / 1e6);
    println!();

    let device = Default::default();

    // generate synthetic token data
    let mut rng = rand::rng();
    let input_ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN)
        .map(|_| (rng.random::<f32>() * VOCAB_SIZE as f32) as i64).collect();
    let target_ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN)
        .map(|_| (rng.random::<f32>() * VOCAB_SIZE as f32) as i64).collect();
    let input = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::new(input_ids, [BATCH_SIZE, SEQ_LEN]), &device);
    let targets = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::new(target_ids, [BATCH_SIZE, SEQ_LEN]), &device);

    // ── Build models ──────────────────────────────────────────────────────
    println!("Building Normal Transformer (FP32)...");
    let normal_lm = build_normal_lm(&device);
    println!("Building BitLinear Transformer (1.58-bit)...");
    let bit_lm = build_bit_lm(&device);
    println!();

    // ── Inference ─────────────────────────────────────────────────────────
    println!("─── Inference ({INFERENCE_STEPS} forwards, {WARMUP_STEPS} warmup) ─────────────────");
    println!("  Model              ms/forward       tok/s");
    println!("  ─────────────────────────────────────────────");
    let (ninf_ms, ninf_tok) = benchmark_inference_normal(&normal_lm, input.clone());
    let (b_inf_ms, b_inf_tok) = benchmark_inference_bit(&bit_lm, input.clone());
    println!("  ─────────────────────────────────────────────");
    println!("  BitLinear vs Normal: {:.2}x speedup\n", b_inf_ms / ninf_ms);

    // ── Training ──────────────────────────────────────────────────────────
    println!("─── Training ({TRAIN_STEPS} steps, Adam LR={LR}) ──────────────────────────");
    println!("  Model              total time      tok/s        loss progression");
    println!("  ──────────────────────────────────────────────────────────────");
    let (n_loss, n_train_tok) = benchmark_training_normal(&normal_lm, input.clone(), targets.clone());
    let (b_loss, b_train_tok) = benchmark_training_bit(&bit_lm, input.clone(), targets.clone());
    println!("  ──────────────────────────────────────────────────────────────");
    println!("  BitLinear vs Normal: {:.2}x speedup (training)\n", b_train_tok / n_train_tok);

    // ── Summary ───────────────────────────────────────────────────────────
    let n_final = n_loss[TRAIN_STEPS - 1];
    let b_final = b_loss[TRAIN_STEPS - 1];
    let n_last5: f64 = n_loss[TRAIN_STEPS - 5..].iter().sum::<f64>() / 5.0;
    let b_last5: f64 = b_loss[TRAIN_STEPS - 5..].iter().sum::<f64>() / 5.0;

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                         FINAL COMPARISON                          ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║  Metric                    │  Normal FP32    │  BitLinear 1.58b    ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║  Inference (ms/forward)    │  {:14.3} │  {:18.3} ║", ninf_ms, b_inf_ms);
    println!("║  Inference (tok/s)         │  {:14.1} │  {:18.1} ║", ninf_tok, b_inf_tok);
    println!("║  Training (tok/s)          │  {:14.1} │  {:18.1} ║", n_train_tok, b_train_tok);
    println!("║  Initial loss              │  {:14.4} │  {:18.4} ║", n_loss[0], b_loss[0]);
    println!("║  Final loss                │  {:14.4} │  {:18.4} ║", n_final, b_final);
    println!("║  Avg loss (last 5 steps)   │  {:14.4} │  {:18.4} ║", n_last5, b_last5);
    println!("║  Loss gap (Bit - Normal)   │  {:>14} │  {:+.18} ║", "", b_final - n_final);
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    if b_final > n_final * 1.5 {
        println!("  ⚠ NOTE: BitLinear loss is significantly higher. This is expected");
        println!("     for ternary quantization — trades precision for speed/memory.");
        println!("     The gap should shrink with proper training (label smoothing,");
        println!("     learning rate tuning, longer training).");
    } else if (b_final - n_final).abs() < 0.1 {
        println!("  ✓ BitLinear achieves near-FP32 loss! Excellent quantization.");
    } else {
        println!("  ~ BitLinear loss gap is moderate.");
    }

    println!();
    Ok(())
}

fn main() {
    if let Err(e) = test_transformer_vs_bit_main() {
        eprintln!("Error: {}", e);
    }
}
