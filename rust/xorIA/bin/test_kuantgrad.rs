// ─── KuantGrad: BitNet 3 capas — Normal vs KuantGrad vs BitLinear real ──
//
// Red ternaria (STE) 3 capas, d_model=16.
// Compara:
//   1. TinyBitNet + AdamW manual (baseline)
//   2. TinyBitNet + AdamW manual con gradientes KuantGrad
//   3. BitLinear real (codebase) + AdamW nativo de Burn
//
// Usage: cargo run --release --bin test_kuantgrad

use std::error::Error;

use burn::prelude::*;
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use burn::module::Module;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearConfig};
use xlstm::blocks::kuantgrad::adamw::{AdamWConfig, AdamWState};
use xlstm::blocks::kuantgrad::compress::{compress, decompress};

type MyBackend = Autodiff<Flex<f32>>;

// ─── Tiny BitNet (ternary weights, STE) ────────────────────────────
#[derive(Clone)]
struct TinyBitNet {
    w1: Vec<f32>, b1: Vec<f32>,
    w2: Vec<f32>, b2: Vec<f32>,
    w3: Vec<f32>, b3: Vec<f32>,
}

impl TinyBitNet {
    fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let s1 = (1.0_f32 / 16.0).sqrt();
        let s2 = (1.0_f32 / 16.0).sqrt();
        let s3 = (1.0_f32 / 2.0).sqrt();
        Self {
            w1: (0..256).map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s1).collect(),
            b1: vec![0.0; 16],
            w2: (0..256).map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s2).collect(),
            b2: vec![0.0; 16],
            w3: (0..32).map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s3).collect(),
            b3: vec![0.0; 2],
        }
    }

    fn quantize_ternary(w: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let group = 128;
        let n = w.len();
        let n_groups = (n + group - 1) / group;
        let mut wq = vec![0.0; n];
        let mut scales = Vec::with_capacity(n_groups);
        for g in 0..n_groups {
            let start = g * group;
            let end = (start + group).min(n);
            let slice = &w[start..end];
            let scale = slice.iter().map(|v| v.abs()).sum::<f32>() / (end - start) as f32;
            let scale = scale.max(1e-8);
            scales.push(scale);
            for i in start..end {
                let q = (w[i] / scale).round().clamp(-1.0, 1.0) as i8;
                wq[i] = (q as f32) * scale;
            }
        }
        (wq, scales)
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (w1q, _) = Self::quantize_ternary(&self.w1);
        let (w2q, _) = Self::quantize_ternary(&self.w2);
        let (w3q, _) = Self::quantize_ternary(&self.w3);

        let mut h1 = vec![0.0; 16];
        for j in 0..16 {
            let mut s = self.b1[j];
            for i in 0..16 { s += x[i] * w1q[j * 16 + i]; }
            h1[j] = s.tanh();
        }
        let mut h2 = vec![0.0; 16];
        for j in 0..16 {
            let mut s = self.b2[j];
            for i in 0..16 { s += h1[i] * w2q[j * 16 + i]; }
            h2[j] = s.tanh();
        }
        let mut out = vec![0.0; 2];
        for j in 0..2 {
            let mut s = self.b3[j];
            for i in 0..16 { s += h2[i] * w3q[j * 16 + i]; }
            out[j] = s;
        }
        let max = out.iter().cloned().fold(-1e10f32, f32::max);
        let exps: Vec<f32> = out.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    fn accuracy(&self, data: &[(Vec<f32>, usize)]) -> f32 {
        let correct: usize = data.iter().filter(|(x, label)| {
            let probs = self.forward(x);
            probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *label
        }).count();
        correct as f32 / data.len() as f32
    }

    fn cross_entropy(&self, data: &[(Vec<f32>, usize)]) -> f32 {
        let mut loss = 0.0;
        for (x, label) in data {
            let probs = self.forward(x);
            loss -= (probs[*label] + 1e-8).ln();
        }
        loss / data.len() as f32
    }

    fn grad_loss_wrt(&self, param_name: &str, param_idx: usize, eps: f32, data: &[(Vec<f32>, usize)]) -> f32 {
        let mut hi = self.clone();
        let mut lo = self.clone();
        let (src_hi, src_lo): (&mut Vec<f32>, &mut Vec<f32>) = match param_name {
            "w1" => (&mut hi.w1, &mut lo.w1),
            "b1" => (&mut hi.b1, &mut lo.b1),
            "w2" => (&mut hi.w2, &mut lo.w2),
            "b2" => (&mut hi.b2, &mut lo.b2),
            "w3" => (&mut hi.w3, &mut lo.w3),
            "b3" => (&mut hi.b3, &mut lo.b3),
            _ => panic!("bad param"),
        };
        let orig = src_hi[param_idx];
        src_hi[param_idx] = orig + eps;
        src_lo[param_idx] = orig - eps;
        (hi.cross_entropy(data) - lo.cross_entropy(data)) / (2.0 * eps)
    }
}

// ─── AdamW del módulo kuantgrad ───────────────────────────────────
// Se usa AdamWConfig + AdamWState de xlstm::blocks::kuantgrad::adamw

// ─── BitNet real con BitLinear del codebase ───────────────────────
#[derive(Module, Debug)]
struct BitNet3Layer<B: Backend> {
    ln1: BitLinear<B>,
    ln2: BitLinear<B>,
    ln3: BitLinear<B>,
}

impl<B: Backend> BitNet3Layer<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            ln1: BitLinearConfig { in_features: 16, out_features: 16, bias: true, activation_bits: 8, rms_norm_eps: 1e-5 }.init(device),
            ln2: BitLinearConfig { in_features: 16, out_features: 16, bias: true, activation_bits: 8, rms_norm_eps: 1e-5 }.init(device),
            ln3: BitLinearConfig { in_features: 16, out_features: 2, bias: true, activation_bits: 8, rms_norm_eps: 1e-5 }.init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.ln1.forward_2d(x).tanh();
        let x = self.ln2.forward_2d(x).tanh();
        self.ln3.forward_2d(x)
    }
}

// ─── Generar datos sintéticos ─────────────────────────────────────
fn gen_data(n: usize) -> Vec<(Vec<f32>, usize)> {
    use rand::Rng;
    let mut rng = rand::rng();
    let means = [[1.0; 16], [-1.0; 16]];
    (0..n).map(|_| {
        let label = if rng.random::<f32>() > 0.5 { 0 } else { 1 };
        let x: Vec<f32> = means[label].iter().map(|m| m + rng.random::<f32>() * 2.0 - 1.0).collect();
        (x, label)
    }).collect()
}

fn param_names_and_lens(m: &TinyBitNet) -> Vec<(&'static str, usize)> {
    vec![
        ("w1", m.w1.len()), ("b1", m.b1.len()),
        ("w2", m.w2.len()), ("b2", m.b2.len()),
        ("w3", m.w3.len()), ("b3", m.b3.len()),
    ]
}

fn param_mut(m: &mut TinyBitNet, pi: usize) -> &mut Vec<f32> {
    match pi {
        0 => &mut m.w1, 1 => &mut m.b1,
        2 => &mut m.w2, 3 => &mut m.b2,
        4 => &mut m.w3, 5 => &mut m.b3,
        _ => panic!("bad pi"),
    }
}

fn data_to_tensors(data: &[(Vec<f32>, usize)])
    -> (Tensor<MyBackend, 2>, Tensor<MyBackend, 1, Int>)
{
    let device = Default::default();
    let n = data.len();
    let mut xs_flat = Vec::with_capacity(n * 16);
    let mut ys = Vec::with_capacity(n);
    for (x, y) in data {
        xs_flat.extend_from_slice(x);
        ys.push(*y as i32);
    }
    let xs = Tensor::<MyBackend, 2>::from_data(TensorData::new(xs_flat, [n, 16]), &device);
    let labels = Tensor::<MyBackend, 1, Int>::from_data(TensorData::new(ys, [n]), &device);
    (xs, labels)
}

fn compute_loss(model: &BitNet3Layer<MyBackend>, xs: Tensor<MyBackend, 2>, labels: Tensor<MyBackend, 1, Int>) -> Tensor<MyBackend, 1> {
    let logits = model.forward(xs);
    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits, labels);
    loss
}

fn accuracy_burn(model: &BitNet3Layer<MyBackend>, xs: &Tensor<MyBackend, 2>, ys: &Tensor<MyBackend, 1, Int>) -> f32 {
    let logits = model.forward(xs.clone().detach());
    let preds = logits.argmax(1);
    let preds_s: Vec<i32> = preds.into_data().as_slice().unwrap().to_vec();
    let ys_s: Vec<i32> = ys.clone().into_data().as_slice().unwrap().to_vec();
    let correct: usize = preds_s.iter().zip(ys_s.iter()).filter(|(p, y)| *p == *y).count();
    correct as f32 / ys_s.len() as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    test_kuantgrad_main()
}

pub fn test_kuantgrad_main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   KuantGrad: TinyBitNet vs TinyBitNet+Kuant vs BitNet real     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let data = gen_data(500);
    let train = data[..400].to_vec();
    let test = data[400..].to_vec();

    println!("\n  Red: 16→16→16→2, ternaria (STE), 500 samples");

    let mut model_a = TinyBitNet::new();
    let mut model_b = model_a.clone();

    let eps = 1e-4;
    let adam_cfg = AdamWConfig { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 1e-4 };

    // Inicializar estados AdamW
    let pnl = param_names_and_lens(&model_a);
    let mut states_a: Vec<AdamWState> = pnl.iter().map(|(_, n)| AdamWState::new(*n)).collect();
    let mut states_b: Vec<AdamWState> = pnl.iter().map(|(_, n)| AdamWState::new(*n)).collect();

    // ─── Modelo C: BitNet real (BitLinear × 3) con Burn AdamW ──────
    let device = Default::default();
    let mut model_c = BitNet3Layer::<MyBackend>::new(&device);
    let mut optim_c = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(adam_cfg.wd)))
        .init::<MyBackend, BitNet3Layer<MyBackend>>();
    let lr_c = adam_cfg.lr as f64;

    // Pre-convertir datos a tensores Burn una sola vez
    let (train_xs, train_ys) = data_to_tensors(&train);
    let (test_xs, test_ys) = data_to_tensors(&test);

    println!("\n  ── Entrenando (10 epochs) ──\n");

    for epoch in 0..30 {
        // ── Modelo A: TinyBitNet + AdamW normal ──
        let mut grads_a: Vec<Vec<f32>> = Vec::new();
        for (_pi, (pname, plen)) in pnl.iter().enumerate() {
            let mut g = vec![0.0; *plen];
            for i in 0..*plen {
                g[i] = model_a.grad_loss_wrt(pname, i, eps, &train);
            }
            grads_a.push(g);
        }

        // ── Modelo B: TinyBitNet + KuantGrad ──
        let mut grads_b: Vec<Vec<f32>> = Vec::new();
        {
            let model_work = model_b.clone();
            for (_pi, (pname, plen)) in pnl.iter().enumerate() {
                let mut g = vec![0.0; *plen];
                for i in 0..*plen {
                    g[i] = model_work.grad_loss_wrt(pname, i, eps, &train);
                }
                let (compressed, ng) = compress(&g);
                grads_b.push(decompress(&compressed, ng, g.len()));

                if epoch == 0 {
                    let ratio = g.len() as f64 * 4.0 / compressed.len() as f64;
                    println!("    {} → {} bytes (ratio {:.2}×)", pname, compressed.len(), ratio);
                }
            }
        }

        // Aplicar AdamW a modelos A y B
        for pi in 0..pnl.len() {
            states_a[pi].step(param_mut(&mut model_a, pi), &grads_a[pi], &adam_cfg);
            states_b[pi].step(param_mut(&mut model_b, pi), &grads_b[pi], &adam_cfg);
        }

        // ── Modelo C: BitNet real + Burn AdamW ──
        let loss_c = compute_loss(&model_c, train_xs.clone(), train_ys.clone());
        let grads_c = loss_c.backward();
        let grads_p_c = GradientsParams::from_grads(grads_c, &model_c);
        model_c = optim_c.step(lr_c, model_c, grads_p_c);

        if epoch % 2 == 0 || epoch == 9 {
            let loss_a = model_a.cross_entropy(&train);
            let loss_b = model_b.cross_entropy(&train);
            let loss_c_val = compute_loss(&model_c, train_xs.clone(), train_ys.clone()).into_scalar();
            let acc_a = model_a.accuracy(&test);
            let acc_b = model_b.accuracy(&test);
            let acc_c = accuracy_burn(&model_c, &test_xs, &test_ys);
            println!("  ep {:2}: Normal loss={:.4} acc={:.2}% | Kuant loss={:.4} acc={:.2}% | BitNet loss={:.4} acc={:.2}%",
                     epoch + 1, loss_a, acc_a * 100.0, loss_b, acc_b * 100.0, loss_c_val, acc_c * 100.0);
        }
    }

    let acc_a = model_a.accuracy(&test);
    let acc_b = model_b.accuracy(&test);
    let acc_c = accuracy_burn(&model_c, &test_xs, &test_ys);
    let loss_a = model_a.cross_entropy(&test);
    let loss_b = model_b.cross_entropy(&test);
    let loss_c_val = compute_loss(&model_c, test_xs, test_ys).into_scalar();

    println!("\n  ── Final (test set) ──");
    println!("  TinyBitNet Normal:    loss={:.4}  acc={:.2}%", loss_a, acc_a * 100.0);
    println!("  TinyBitNet KuantGrad: loss={:.4}  acc={:.2}%", loss_b, acc_b * 100.0);
    println!("  BitNet real (Burn):   loss={:.4}  acc={:.2}%", loss_c_val, acc_c * 100.0);

    let diff_ab = (acc_a - acc_b).abs() * 100.0;
    let diff_ac = (acc_a - acc_c).abs() * 100.0;
    println!("\n  Diferencia Normal vs KuantGrad: {:.2} puntos porcentuales", diff_ab);
    println!("  Diferencia Normal vs BitNet real: {:.2} puntos porcentuales", diff_ac);

    if diff_ab < 5.0 { println!("  ✓ KuantGrad mantiene rendimiento similar"); }
    else { println!("  ⚠ Diferencia significativa KuantGrad"); }

    Ok(())
}
