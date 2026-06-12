use burn::tensor::{Tensor, Distribution, Int, activation, backend::Backend};
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig};

type TestBackend = burn_ndarray::NdArray<f32>;
type AdBackend  = Autodiff<TestBackend>;

// ────────── MinGRU INTERNAL ──────────

fn log_g<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()) + 0.5).log();
    let neg = activation::softplus(x.neg(), 1.0).neg(); 
    neg.mask_where(mask, pos)
}

/// g(x) en espacio lineal (paper Eq.): x>=0 → x+0.5, x<0 → sigmoid(x)
fn g<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = activation::relu(x.clone()) + 0.5;
    let neg = activation::sigmoid(x);
    neg.mask_where(mask, pos)
}

fn log_cumsum_exp<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // CENTRADO ÓPTIMO EN F32 (Sin bucles / Sin Apensors):
    // El rango dinámico de exp(f32) es [-87.3, 88.7].
    // Centramos toda la secuencia dividiendo el rango a la mitad para evitar t=0 -> log(0) -> NaN
    let max = x.clone().detach().max_dim(1);
    let min = x.clone().detach().neg().max_dim(1).neg(); // Equivale a min_dim(1)
    let m = (max + min) / 2.0;
    
    (x - m.clone()).exp().cumsum(1).log() + m
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, _s, h] = log_values.dims();
    let device = log_values.device();
    
    let a_star = Tensor::cat(vec![
        Tensor::zeros([b, 1, h], &device),
        log_coeffs.cumsum(1)
    ], 1);
    
    let x = log_values - a_star.clone();
    let log_h0_plus_b_star = log_cumsum_exp(x);
    
    let log_h = a_star + log_h0_plus_b_star;
    let dims = log_h.dims();
    
    log_h.exp().slice([0..b, 1..dims[1], 0..h])
}

#[derive(Module, Debug)]
pub struct MinGru<B: Backend> {
    pub linear_z: Linear<B>,
    pub linear_h: Linear<B>,
    pub proj: Linear<B>,
}

impl<B: Backend> MinGru<B> {
    pub fn init(dim: usize, expansion: usize, device: &B::Device) -> Self {
        let hidden = dim * expansion;
        Self {
            linear_z: LinearConfig::new(dim, hidden).with_bias(false).init(device),
            linear_h: LinearConfig::new(dim, hidden).with_bias(false).init(device),
            proj:     LinearConfig::new(hidden, dim).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, h_0: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let update_gate = self.linear_z.forward(x.clone());
        let hidden_state = self.linear_h.forward(x.clone());
        
        let k = activation::softplus(update_gate, 1.0).neg();
        let log_z = activation::softplus(k.clone().neg(), 1.0).neg();
        let log_coeffs = activation::softplus(k, 1.0).neg();
        
        let log_h_0 = log_g(h_0);
        let log_tilde_h = log_g(hidden_state);
        
        let log_values = Tensor::cat(vec![log_h_0, log_z + log_tilde_h], 1);
        let output = parallel_scan_log(log_coeffs, log_values);
        
        let [b, s, h] = output.dims();
        let latest_hidden = output.clone().slice([0..b, s-1..s, 0..h]);
        
        (self.proj.forward(output), latest_hidden)
    }

    /// Modo secuencial: procesa 1 token con estado recurrente.
    /// IMPORTANTE: h_prev debe ser g(h_0) en el primer paso (no h_0 crudo),
    /// porque el forward paralelo aplica log_g(h_0) internamente.
    /// Para pasos subsiguientes, h_prev es el h_t retornado por el paso anterior.
    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let update_gate = self.linear_z.forward(x_t.clone());
        let hidden_state = self.linear_h.forward(x_t);
        
        let k = activation::softplus(update_gate, 1.0).neg();
        let log_z = activation::softplus(k.clone().neg(), 1.0).neg();
        let log_coeffs = activation::softplus(k, 1.0).neg();
        
        let log_tilde_h = log_g(hidden_state);
        
        // Recurrencia single-step equivalente al parallel_scan_log:
        // h_t = exp(log_coeffs) * h_prev + exp(log_z + log_tilde_h)
        let h_t = (log_coeffs.exp() * h_prev) + (log_z + log_tilde_h).exp();
        
        (self.proj.forward(h_t.clone()), h_t)
    }
}

#[derive(Module, Debug)]
struct CopyTaskModel<B: Backend> {
    embed: Embedding<B>,
    mingru: MinGru<B>,
    to_logits: Linear<B>,
}

// ────────── TESTS ──────────

fn test_gradients() {
    let device = Default::default();
    let seq_len = 250;
    let b = 1;
    let d = 16;
    let expansion = 2;

    let model = MinGru::<AdBackend>::init(d, expansion, &device);
    let x = Tensor::<AdBackend, 3>::random([b, seq_len, d], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let h0 = Tensor::<AdBackend, 3>::zeros([b, 1, d * expansion], &device);

    let (out, _) = model.forward(x.clone(), h0);
    let grads = out.sum().backward();
    let grad = x.grad(&grads).expect("No grad");

    println!("--- TEST 1: Gradient Flow (S={seq_len}) ---");
    let mut checkpoints: Vec<usize> = (0..seq_len).step_by(50).collect();
    checkpoints.push(seq_len - 1);
    checkpoints.dedup();

    for t in checkpoints {
        let g_t = grad.clone().slice([0..b, t..t+1, 0..d]).abs().mean().into_scalar();
        println!("  t={t:3}  |grad|={g_t:.10}");
    }
}

fn test_sequential_equivalence() {
    let device = Default::default();
    let b = 2;
    let d = 16;
    let expansion = 2;
    let hidden = d * expansion;
    let seq_len = 8;

    let model = MinGru::<TestBackend>::init(d, expansion, &device);
    let x = Tensor::<TestBackend, 3>::random([b, seq_len, d], Distribution::Normal(0.0, 1.0), &device);
    let h0 = Tensor::<TestBackend, 3>::zeros([b, 1, hidden], &device);

    // ── Parallel forward ──
    let (out_parallel, _) = model.forward(x.clone(), h0.clone());

    // ── Sequential forward (step by step) ──
    // CLAVE: h_0 debe pasar por g() para igualar lo que el parallel hace con log_g(h_0)
    let mut h_prev = g(h0);
    let mut seq_outputs = Vec::new();
    
    for t in 0..seq_len {
        let x_t = x.clone().slice([0..b, t..t+1, 0..d]);
        let (out_t, h_next) = model.sequential_mode(x_t, h_prev);
        seq_outputs.push(out_t);
        h_prev = h_next;
    }
    let out_sequential = Tensor::cat(seq_outputs, 1);

    println!("\n--- TEST 3: Parallel vs Sequential Equivalence (S={seq_len}) ---");
    let diff = (out_parallel.clone() - out_sequential.clone()).abs();
    let max_diff = diff.clone().max().into_scalar();
    let mean_diff = diff.mean().into_scalar();
    
    println!("  Max  |parallel - sequential| = {max_diff:.10}");
    println!("  Mean |parallel - sequential| = {mean_diff:.10}");

    // Verificar con distintas seq_len
    for test_len in [1, 4, 16, 64] {
        let x2 = Tensor::<TestBackend, 3>::random([1, test_len, d], Distribution::Normal(0.0, 1.0), &device);
        let h0_2 = Tensor::<TestBackend, 3>::zeros([1, 1, hidden], &device);
        
        let (out_p, _) = model.forward(x2.clone(), h0_2.clone());
        
        let mut h = g(h0_2);
        let mut outs = Vec::new();
        for t in 0..test_len {
            let xt = x2.clone().slice([0..1, t..t+1, 0..d]);
            let (ot, hn) = model.sequential_mode(xt, h);
            outs.push(ot);
            h = hn;
        }
        let out_s = Tensor::cat(outs, 1);
        let md = (out_p - out_s).abs().max().into_scalar();
        let status = if md < 1e-4 { "OK" } else { "FAIL" };
        println!("  S={test_len:3}  max_diff={md:.10}  [{status}]");
    }

    if max_diff < 1e-4 {
        println!("SUCCESS: Parallel ≈ Sequential (max_diff < 1e-4)");
    } else {
        println!("FAILURE: Diferencia significativa ({max_diff:.6})");
    }
}

fn test_copy_task() {
    let device = Default::default();
    let vocab_size = 10;
    let d = 16;
    let expansion = 2;
    let seq_len = 32;
    let batch_size = 4;

    let mut pattern = Vec::new();
    for _ in 0..200 { pattern.extend(0..10i64); }
    let data = Tensor::<AdBackend, 1, Int>::from_data(pattern.as_slice(), &device);

    let config_embed = EmbeddingConfig::new(vocab_size, d);
    let config_mingru = MinGru::<AdBackend>::init(d, expansion, &device);
    let config_logits = LinearConfig::new(d, vocab_size).with_bias(false);
    
    let mut model = CopyTaskModel::<AdBackend> {
        embed: config_embed.init(&device),
        mingru: config_mingru,
        to_logits: config_logits.init(&device),
    };

    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    println!("\n--- TEST 2: LM Copy Task (S={seq_len}, B={batch_size}) ---");
    
    let mut losses = Vec::new();
    let total_len = pattern.len();
    let num_batches = (total_len - 1) / (batch_size * seq_len);
    let trimmed_len = num_batches * batch_size * seq_len;
    let row_len = trimmed_len / batch_size;

    for step in 1..=200 {
        let mut step_loss = 0.0;
        let mut count = 0;

        for chunk_offset in (0..(row_len - seq_len)).step_by(seq_len) {
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();
            
            for b_idx in 0..batch_size {
                let start = b_idx * row_len + chunk_offset;
                x_batch.push(data.clone().slice([start..start + seq_len]));
                y_batch.push(data.clone().slice([start + 1..start + seq_len + 1]));
            }
            
            let x_tensor = Tensor::cat(x_batch, 0).reshape([batch_size, seq_len]);
            let y_tensor = Tensor::cat(y_batch, 0).reshape([batch_size * seq_len]);

            let x_emb = model.embed.forward(x_tensor);
            let h0 = Tensor::zeros([batch_size, 1, d * expansion], &device);
            let (out, _) = model.mingru.forward(x_emb, h0);
            let logits = model.to_logits.forward(out.reshape([batch_size * seq_len, d]));
            
            let loss = loss_fn.forward(logits, y_tensor);
            step_loss += loss.clone().into_scalar();
            count += 1;

            let grads = loss.backward();
            let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(2e-3, model, grads_p);
        }

        let avg = step_loss / count as f32;
        losses.push(avg);
        if step == 1 || step % 50 == 0 {
            println!("  Step {step:3}  loss={avg:.4}");
        }
    }

    if losses[losses.len()-1] < losses[0] * 0.5 {
        println!("SUCCESS: Copy Task converge!");
    } else {
        println!("FAILURE: Copy Task no converge ({:.4} -> {:.4})", losses[0], losses[losses.len()-1]);
    }
}

fn main() {
    println!("============================================");
    println!("  TEST SUITE: MinGRU (Inline Implementation)");
    println!("============================================");
    
    test_gradients();
    test_copy_task();
    test_sequential_equivalence();
    
    println!("\n============================================");
    println!("  ALL TESTS COMPLETE");
    println!("============================================");
}
