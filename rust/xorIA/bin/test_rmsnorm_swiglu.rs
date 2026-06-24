use burn::prelude::*;
use burn::tensor::{Tensor, Distribution};
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use burn::module::{Module, Param};
use burn::nn;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;

type MyBackend = Autodiff<Flex<f32>>;

const NUM_LAYERS: usize = 4;
const D_MODEL: usize = 64;
const HIDDEN: usize = 128;
const BATCH: usize = 4;
const SEQ: usize = 16;
const STEPS: usize = 200;
const LR: f64 = 3e-4;

// ─── Our custom RMSNorm ─────────────────────────────────────────────

#[derive(Module, Debug)]
struct OurRMSNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> OurRMSNorm<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps: 1e-5,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let denom = (x.clone().powf_scalar(2.0).mean_dim(2) + self.eps as f32).sqrt();
        let normed = x / denom;
        let w = self.weight.val().unsqueeze::<2>().unsqueeze::<3>();
        normed * w
    }
}

// ─── Our custom SwiGLU ──────────────────────────────────────────────

#[derive(Module, Debug)]
struct OurSwiGLU<B: Backend> {
    gate: nn::Linear<B>,
    up: nn::Linear<B>,
    down: nn::Linear<B>,
}

impl<B: Backend> OurSwiGLU<B> {
    fn new(d_model: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            gate: nn::LinearConfig::new(d_model, hidden).with_bias(false).init(device),
            up: nn::LinearConfig::new(d_model, hidden).with_bias(false).init(device),
            down: nn::LinearConfig::new(hidden, d_model).with_bias(false).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = burn::tensor::activation::silu(self.gate.forward(x.clone()));
        let u = self.up.forward(x);
        self.down.forward(g * u)
    }
}

// ─── Our Block: RMSNorm → SwiGLU + Residual ────────────────────────

#[derive(Module, Debug)]
struct OurBlock<B: Backend> {
    norm: OurRMSNorm<B>,
    ffn: OurSwiGLU<B>,
}

impl<B: Backend> OurBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.norm.forward(x.clone());
        let h = self.ffn.forward(h);
        x + h
    }
}

#[derive(Module, Debug)]
struct OurStack<B: Backend> {
    embed: nn::Linear<B>,
    layers: Vec<OurBlock<B>>,
    final_norm: OurRMSNorm<B>,
    head: nn::Linear<B>,
}

impl<B: Backend> OurStack<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.embed.forward(x);
        for layer in &self.layers {
            h = layer.forward(h);
        }
        let h = self.final_norm.forward(h);
        self.head.forward(h)
    }
}

// ─── Burn Block: RmsNorm → SwiGlu + Residual ───────────────────────

#[derive(Module, Debug)]
struct BurnBlock<B: Backend> {
    norm: nn::RmsNorm<B>,
    swiglu: nn::SwiGlu<B>,
    down: nn::Linear<B>,
}

impl<B: Backend> BurnBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.norm.forward(x.clone());
        let h = self.swiglu.forward(h);
        let h = self.down.forward(h);
        x + h
    }
}

#[derive(Module, Debug)]
struct BurnStack<B: Backend> {
    embed: nn::Linear<B>,
    layers: Vec<BurnBlock<B>>,
    final_norm: nn::RmsNorm<B>,
    head: nn::Linear<B>,
}

impl<B: Backend> BurnStack<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.embed.forward(x);
        for layer in &self.layers {
            h = layer.forward(h);
        }
        let h = self.final_norm.forward(h);
        self.head.forward(h)
    }
}

// ─── MSE Loss ───────────────────────────────────────────────────────

fn mse_loss<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.powf_scalar(2.0).mean().reshape([1])
}

// ─── Main ───────────────────────────────────────────────────────────

pub fn test_rmsnorm_swiglu_main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  RMSNorm + SwiGLU: Custom vs burn::nn  ({}-layer stack)          ║", NUM_LAYERS);
    println!("║  d_model={}, hidden={}, batch={}, seq={}, steps={}               ║", D_MODEL, HIDDEN, BATCH, SEQ, STEPS);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let device = Default::default();

    // Fixed regression target
    let x_train = Tensor::<MyBackend, 3>::random(
        [BATCH, SEQ, D_MODEL], Distribution::Normal(0.0, 1.0), &device,
    );
    let target_weight = Tensor::<MyBackend, 2>::random(
        [D_MODEL, D_MODEL], Distribution::Normal(0.0, 0.5), &device,
    );
    let target_raw = x_train.clone()
        .reshape([BATCH * SEQ, D_MODEL])
        .matmul(target_weight)
        .reshape([BATCH, SEQ, D_MODEL]);
    let y_target = burn::tensor::activation::silu(target_raw);

    // ─── Train Our Model ──────────────────────────────────────────────
    println!("\n━━━ Phase 1: Custom RMSNorm + SwiGLU ({} layers) ━━━", NUM_LAYERS);
    let mut our_model = OurStack {
        embed: nn::LinearConfig::new(D_MODEL, D_MODEL).with_bias(false).init(&device),
        layers: (0..NUM_LAYERS).map(|_| OurBlock {
            norm: OurRMSNorm::new(D_MODEL, &device),
            ffn: OurSwiGLU::new(D_MODEL, HIDDEN, &device),
        }).collect(),
        final_norm: OurRMSNorm::new(D_MODEL, &device),
        head: nn::LinearConfig::new(D_MODEL, D_MODEL).with_bias(false).init(&device),
    };
    let mut our_optim = AdamConfig::new().init();

    let mut our_losses = Vec::new();
    for step in 1..=STEPS {
        let pred = our_model.forward(x_train.clone());
        let loss = mse_loss(pred, y_target.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &our_model);
        our_model = our_optim.step(LR, our_model, grads_params);
        if step == 1 || step % 40 == 0 || step == STEPS {
            println!("  Step {:4}: Loss = {:.8}", step, loss_val);
        }
        our_losses.push(loss_val);
    }

    // ─── Train Burn Model ─────────────────────────────────────────────
    println!("\n━━━ Phase 2: burn::nn RmsNorm + SwiGlu ({} layers) ━━━", NUM_LAYERS);
    let mut burn_model = BurnStack {
        embed: nn::LinearConfig::new(D_MODEL, D_MODEL).with_bias(false).init(&device),
        layers: (0..NUM_LAYERS).map(|_| BurnBlock {
            norm: nn::RmsNormConfig::new(D_MODEL).with_epsilon(1e-5).init(&device),
            swiglu: nn::SwiGluConfig::new(D_MODEL, HIDDEN).init(&device),
            down: nn::LinearConfig::new(HIDDEN, D_MODEL).with_bias(false).init(&device),
        }).collect(),
        final_norm: nn::RmsNormConfig::new(D_MODEL).with_epsilon(1e-5).init(&device),
        head: nn::LinearConfig::new(D_MODEL, D_MODEL).with_bias(false).init(&device),
    };
    let mut burn_optim = AdamConfig::new().init();

    let mut burn_losses = Vec::new();
    for step in 1..=STEPS {
        let pred = burn_model.forward(x_train.clone());
        let loss = mse_loss(pred, y_target.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &burn_model);
        burn_model = burn_optim.step(LR, burn_model, grads_params);
        if step == 1 || step % 40 == 0 || step == STEPS {
            println!("  Step {:4}: Loss = {:.8}", step, loss_val);
        }
        burn_losses.push(loss_val);
    }

    // ─── Results ──────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     RESULTS ({} layers)                         ║", NUM_LAYERS);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    let o_first = our_losses.first().unwrap();
    let o_last = our_losses.last().unwrap();
    let b_first = burn_losses.first().unwrap();
    let b_last = burn_losses.last().unwrap();
    println!("║ Custom RMSNorm + SwiGLU:                                       ║");
    println!("║   Initial Loss: {:.8}                                    ║", o_first);
    println!("║   Final Loss:   {:.8}                                    ║", o_last);
    println!("║   Reduction:    {:.2}x                                         ║", o_first / o_last);
    println!("║                                                                ║");
    println!("║ burn::nn RmsNorm + SwiGlu:                                     ║");
    println!("║   Initial Loss: {:.8}                                    ║", b_first);
    println!("║   Final Loss:   {:.8}                                    ║", b_last);
    println!("║   Reduction:    {:.2}x                                         ║", b_first / b_last);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n═══ TEST COMPLETE ═══");
}
