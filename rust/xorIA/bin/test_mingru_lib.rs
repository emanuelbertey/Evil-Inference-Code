/// Test de MinGRU usando la implementación de la librería (xlstm::MinGru)
/// Equivalente a test_mingru.rs pero usando MinGruConfig + MinGruState del módulo real.
use burn::tensor::{Tensor, Distribution, Int, backend::Backend};
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig};

use xlstm::{MinGru, MinGruConfig, MinGruState};

type TestBackend = burn_ndarray::NdArray<f32>;
type AdBackend  = Autodiff<TestBackend>;

// ────────── MODELO COPY TASK ──────────

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
    let hidden = d * expansion;

    let config = MinGruConfig { input_features: d, expansion_factor: expansion };
    let model = config.init::<AdBackend>(&device);
    
    let x = Tensor::<AdBackend, 3>::random([b, seq_len, d], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let h0 = Tensor::<AdBackend, 3>::zeros([b, 1, hidden], &device);
    let state = Some(vec![MinGruState::new(h0)]);

    let (out, _) = model.forward(x.clone(), state);
    let grads = out.sum().backward();
    let grad = x.grad(&grads).expect("No grad");

    println!("--- TEST 1: Gradient Flow (S={seq_len}) [LIBRARY MinGru] ---");
    let mut checkpoints: Vec<usize> = (0..seq_len).step_by(50).collect();
    checkpoints.push(seq_len - 1);
    checkpoints.dedup();

    let mut all_valid = true;
    for t in &checkpoints {
        let g_t = grad.clone().slice([0..b, *t..*t+1, 0..d]).abs().mean().into_scalar();
        println!("  t={t:3}  |grad|={g_t:.10}");
        if g_t.is_nan() || g_t == 0.0 {
            all_valid = false;
        }
    }
    
    if all_valid {
        println!("SUCCESS: Todos los gradientes son válidos (no NaN, no zero)");
    } else {
        println!("FAILURE: Hay gradientes NaN o zero");
    }
}

fn test_sequential_equivalence() {
    let device = Default::default();
    let b = 2;
    let d = 16;
    let expansion = 2;
    let hidden = d * expansion;
    let seq_len = 8;

    let config = MinGruConfig { input_features: d, expansion_factor: expansion };
    let model = config.init::<TestBackend>(&device);
    
    let x = Tensor::<TestBackend, 3>::random([b, seq_len, d], Distribution::Normal(0.0, 1.0), &device);
    let h0 = Tensor::<TestBackend, 3>::zeros([b, 1, hidden], &device);

    // Parallel forward
    let (out_parallel, _) = model.forward(x.clone(), Some(vec![MinGruState::new(h0.clone())]));

    // Sequential forward (step by step)
    let mut h_prev = h0;
    let mut seq_outputs = Vec::new();
    for t in 0..seq_len {
        let x_t = x.clone().slice([0..b, t..t+1, 0..d]);
        let (out_t, h_next) = model.sequential_mode(x_t, h_prev);
        seq_outputs.push(out_t);
        h_prev = h_next;
    }
    let out_sequential = Tensor::cat(seq_outputs, 1);

    println!("\n--- TEST 2: Parallel vs Sequential Equivalence (S={seq_len}) ---");
    let diff = (out_parallel.clone() - out_sequential.clone()).abs();
    let max_diff = diff.clone().max().into_scalar();
    let mean_diff = diff.mean().into_scalar();
    
    println!("  Max  |parallel - sequential| = {max_diff:.10}");
    println!("  Mean |parallel - sequential| = {mean_diff:.10}");

    if max_diff < 1e-4 {
        println!("SUCCESS: Parallel ≈ Sequential (max_diff < 1e-4)");
    } else {
        println!("WARNING: Diferencia significativa ({max_diff:.6})");
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

    let config = MinGruConfig { input_features: d, expansion_factor: expansion };
    let mingru = config.init::<AdBackend>(&device);
    let config_logits = LinearConfig::new(d, vocab_size).with_bias(false);
    
    let mut model = CopyTaskModel::<AdBackend> {
        embed: EmbeddingConfig::new(vocab_size, d).init(&device),
        mingru,
        to_logits: config_logits.init(&device),
    };

    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    println!("\n--- TEST 3: LM Copy Task (S={seq_len}, B={batch_size}) [LIBRARY MinGru] ---");
    
    let mut losses = Vec::new();
    let total_len = pattern.len();
    let num_batches = (total_len - 1) / (batch_size * seq_len);
    let _trimmed_len = num_batches * batch_size * seq_len;
    let row_len = (num_batches * batch_size * seq_len) / batch_size;

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
            let (out, _) = model.mingru.forward(x_emb, Some(vec![MinGruState::new(h0)]));
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

fn test_h0_persistence() {
    let device = Default::default();
    let b = 1;
    let d = 16;
    let expansion = 2;
    let hidden = d * expansion;
    let seq_len = 16;

    let config = MinGruConfig { input_features: d, expansion_factor: expansion };
    let model = config.init::<TestBackend>(&device);
    
    let x = Tensor::<TestBackend, 3>::random([b, seq_len, d], Distribution::Normal(0.0, 1.0), &device);
    
    // Forward con h0 = zeros
    let h0_zeros = Tensor::<TestBackend, 3>::zeros([b, 1, hidden], &device);
    let (out_zeros, states_zeros) = model.forward(x.clone(), Some(vec![MinGruState::new(h0_zeros)]));
    
    // Forward con h0 = estado anterior (simula persistencia)
    let h0_prev = states_zeros[0].hidden.clone();
    let (out_with_state, _) = model.forward(x.clone(), Some(vec![MinGruState::new(h0_prev)]));

    println!("\n--- TEST 4: h_0 Persistence Effect ---");
    let diff = (out_zeros - out_with_state).abs().max().into_scalar();
    println!("  Max |out(h0=0) - out(h0=prev)| = {diff:.10}");
    
    if diff > 1e-6 {
        println!("SUCCESS: h_0 persistente cambia la salida (diff={diff:.6}) — el estado se propaga");
    } else {
        println!("WARNING: h_0 no tiene efecto — posible bug");
    }
}

fn main() {
    println!("============================================");
    println!("  TEST SUITE: MinGRU Library Implementation");
    println!("============================================\n");
    
    test_gradients();
    test_sequential_equivalence();
    test_copy_task();
    test_h0_persistence();
    
    println!("\n============================================");
    println!("  ALL TESTS COMPLETE");
    println!("============================================");
}
