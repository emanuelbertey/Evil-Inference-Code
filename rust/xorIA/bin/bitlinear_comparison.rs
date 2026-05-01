// ─── BitLinear vs Normal Linear: Training Comparison ────────────────────────
//
// This test trains a simple MLP on a regression task using:
//   1. Normal nn::Linear layers
//   2. BitLinear ternary {-1, 0, +1} layers
//
// Both models have the same architecture and are trained on the same data.
// We compare convergence speed, final loss, and weight distribution.

use burn::prelude::*;
use burn::tensor::{Tensor, Distribution};
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use burn::module::{Module, Param};
use burn::config::Config;
use burn::nn;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;

use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearConfig};

type MyBackend = Autodiff<Flex<f32>>;

// ─── Normal MLP (using standard nn::Linear) ────────────────────────────────

#[derive(Module, Debug)]
struct NormalMLP<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
}

#[derive(Config, Debug)]
struct NormalMLPConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl NormalMLPConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> NormalMLP<B> {
        NormalMLP {
            fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim)
                .with_bias(true)
                .init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(true)
                .init(device),
            fc3: nn::LinearConfig::new(self.hidden_dim, self.output_dim)
                .with_bias(true)
                .init(device),
        }
    }
}

impl<B: Backend> NormalMLP<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = burn::tensor::activation::gelu(self.fc1.forward(x));
        let h = burn::tensor::activation::gelu(self.fc2.forward(h));
        self.fc3.forward(h)
    }
}

// ─── BitLinear MLP (using BitLinear ternary layers) ─────────────────────────

#[derive(Module, Debug)]
struct BitMLP<B: Backend> {
    fc1: BitLinear<B>,
    fc2: BitLinear<B>,
    fc3: BitLinear<B>,
}

#[derive(Config, Debug)]
struct BitMLPConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl BitMLPConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> BitMLP<B> {
        BitMLP {
            fc1: BitLinearConfig {
                in_features: self.input_dim,
                out_features: self.hidden_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(device),
            fc2: BitLinearConfig {
                in_features: self.hidden_dim,
                out_features: self.hidden_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(device),
            fc3: BitLinearConfig {
                in_features: self.hidden_dim,
                out_features: self.output_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(device),
        }
    }
}

impl<B: Backend> BitMLP<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = burn::tensor::activation::gelu(self.fc1.forward(x));
        let h = burn::tensor::activation::gelu(self.fc2.forward(h));
        self.fc3.forward(h)
    }
}

// ─── MSE Loss ───────────────────────────────────────────────────────────────

fn mse_loss<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.powf_scalar(2.0).mean().reshape([1])
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   BitLinear b1.58 vs Normal Linear — Training Comparison       ║");
    println!("║   Ternary Weights {{-1, 0, +1}} with STE + AbsMean Quantization ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let device = Default::default();

    let batch_size = 4;
    let seq_len = 8;
    let input_dim = 32;
    let hidden_dim = 64;
    let output_dim = 16;
    let lr = 3e-4;
    let steps = 200;

    // ─── Create a fixed regression target ───────────────────────────────
    // We generate random input and a "teacher" function output to learn
    let x_train = Tensor::<MyBackend, 3>::random(
        [batch_size, seq_len, input_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Target: a simple nonlinear function
    let target_weight = Tensor::<MyBackend, 2>::random(
        [input_dim, output_dim],
        Distribution::Normal(0.0, 0.5),
        &device,
    );
    let target_raw = x_train.clone()
        .reshape([batch_size * seq_len, input_dim])
        .matmul(target_weight)
        .reshape([batch_size, seq_len, output_dim]);
    let y_target = burn::tensor::activation::silu(target_raw);

    // ─── Model 1: Normal Linear MLP ─────────────────────────────────────
    println!("\n━━━ Phase 1: Training Normal Linear MLP ━━━");
    let normal_config = NormalMLPConfig {
        input_dim,
        hidden_dim,
        output_dim,
    };
    let mut normal_model = normal_config.init::<MyBackend>(&device);
    let mut normal_optim = AdamConfig::new().init();

    let mut normal_losses = Vec::new();
    for step in 1..=steps {
        let pred = normal_model.forward(x_train.clone());
        let loss = mse_loss(pred, y_target.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();

        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &normal_model);
        normal_model = normal_optim.step(lr, normal_model, grads_params);

        if step == 1 || step % 20 == 0 {
            println!("  Step {:4}: Loss = {:.8}", step, loss_val);
        }
        normal_losses.push(loss_val);
    }

    // ─── Model 2: BitLinear MLP ─────────────────────────────────────────
    println!("\n━━━ Phase 2: Training BitLinear (Ternary) MLP ━━━");
    let bit_config = BitMLPConfig {
        input_dim,
        hidden_dim,
        output_dim,
    };
    let mut bit_model = bit_config.init::<MyBackend>(&device);
    let mut bit_optim = AdamConfig::new().init();

    let mut bit_losses = Vec::new();
    for step in 1..=steps {
        let pred = bit_model.forward(x_train.clone());
        let loss = mse_loss(pred, y_target.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();

        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &bit_model);
        bit_model = bit_optim.step(lr, bit_model, grads_params);

        if step == 1 || step % 20 == 0 {
            println!("  Step {:4}: Loss = {:.8}", step, loss_val);
        }
        bit_losses.push(loss_val);
    }

    // ─── Results Comparison ─────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     RESULTS COMPARISON                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let normal_first = normal_losses.first().unwrap();
    let normal_last = normal_losses.last().unwrap();
    let bit_first = bit_losses.first().unwrap();
    let bit_last = bit_losses.last().unwrap();

    println!("║ Normal Linear MLP:                                             ║");
    println!("║   Initial Loss: {:.8}                                    ║", normal_first);
    println!("║   Final Loss:   {:.8}                                    ║", normal_last);
    println!("║   Reduction:    {:.2}x                                         ║", normal_first / normal_last);
    println!("║                                                                ║");
    println!("║ BitLinear (Ternary) MLP:                                       ║");
    println!("║   Initial Loss: {:.8}                                    ║", bit_first);
    println!("║   Final Loss:   {:.8}                                    ║", bit_last);
    println!("║   Reduction:    {:.2}x                                         ║", bit_first / bit_last);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // ─── Weight Distribution Analysis ───────────────────────────────────
    println!("\n━━━ Phase 3: BitLinear Weight Distribution Analysis ━━━");
    let (neg, zero, pos, total) = bit_model.fc1.weight_distribution(&device);
    println!("  Layer fc1 ({} x {}):", hidden_dim, input_dim);
    println!("    -1: {:5} ({:.1}%)", neg, 100.0 * neg as f64 / total as f64);
    println!("     0: {:5} ({:.1}%)", zero, 100.0 * zero as f64 / total as f64);
    println!("    +1: {:5} ({:.1}%)", pos, 100.0 * pos as f64 / total as f64);

    let (neg, zero, pos, total) = bit_model.fc2.weight_distribution(&device);
    println!("  Layer fc2 ({} x {}):", hidden_dim, hidden_dim);
    println!("    -1: {:5} ({:.1}%)", neg, 100.0 * neg as f64 / total as f64);
    println!("     0: {:5} ({:.1}%)", zero, 100.0 * zero as f64 / total as f64);
    println!("    +1: {:5} ({:.1}%)", pos, 100.0 * pos as f64 / total as f64);

    let (neg, zero, pos, total) = bit_model.fc3.weight_distribution(&device);
    println!("  Layer fc3 ({} x {}):", output_dim, hidden_dim);
    println!("    -1: {:5} ({:.1}%)", neg, 100.0 * neg as f64 / total as f64);
    println!("     0: {:5} ({:.1}%)", zero, 100.0 * zero as f64 / total as f64);
    println!("    +1: {:5} ({:.1}%)", pos, 100.0 * pos as f64 / total as f64);

    // ─── Memory Savings Estimate ────────────────────────────────────────
    println!("\n━━━ Memory Savings Estimate ━━━");
    let total_params_normal = input_dim * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim;
    let total_params_bit = input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * output_dim;
    let fp32_bytes = total_params_normal * 4;
    let ternary_bytes = (total_params_bit * 2 + 7) / 8; // 2 bits per weight
    let rms_norm_bytes = (input_dim + hidden_dim + hidden_dim) * 4; // RMSNorm weights in fp32
    let total_bit_bytes = ternary_bytes + rms_norm_bytes;

    println!("  Normal Linear (FP32):  {} bytes ({:.1} KB)", fp32_bytes, fp32_bytes as f64 / 1024.0);
    println!("  BitLinear (Ternary):   {} bytes ({:.1} KB)", total_bit_bytes, total_bit_bytes as f64 / 1024.0);
    println!("  Compression ratio:     {:.1}x", fp32_bytes as f64 / total_bit_bytes as f64);

    // ─── Gradient Flow Check ────────────────────────────────────────────
    println!("\n━━━ Phase 4: Gradient Flow Verification (STE) ━━━");
    let x_test = Tensor::<MyBackend, 3>::random(
        [1, 4, input_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let pred = bit_model.forward(x_test);
    let loss = pred.powf_scalar(2.0).mean();
    let grads = loss.backward();

    // Check that gradients flow through all BitLinear layers
    if let Some(grad) = bit_model.fc1.weight.grad(&grads) {
        let norm: f32 = grad.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
        println!("  fc1.weight gradient norm: {:.10}", norm);
        if norm > 0.0 {
            println!("  ✅ STE gradient flows correctly through fc1");
        } else {
            println!("  ❌ STE gradient is zero — quantization blocks gradients!");
        }
    }

    if let Some(grad) = bit_model.fc2.weight.grad(&grads) {
        let norm: f32 = grad.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
        println!("  fc2.weight gradient norm: {:.10}", norm);
        if norm > 0.0 {
            println!("  ✅ STE gradient flows correctly through fc2");
        } else {
            println!("  ❌ STE gradient is zero — quantization blocks gradients!");
        }
    }

    if let Some(grad) = bit_model.fc3.weight.grad(&grads) {
        let norm: f32 = grad.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
        println!("  fc3.weight gradient norm: {:.10}", norm);
        if norm > 0.0 {
            println!("  ✅ STE gradient flows correctly through fc3");
        } else {
            println!("  ❌ STE gradient is zero — quantization blocks gradients!");
        }
    }

    println!("\n═══ TEST COMPLETE ═══");
}
