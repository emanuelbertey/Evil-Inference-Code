use xlstm::{MinGru, MinGruConfig};
use burn::tensor::{Tensor, Distribution, backend::Backend};
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, EmbeddingConfig};

type TestBackend = burn_ndarray::NdArray<f64>;
type AdBackend  = Autodiff<TestBackend>;

// ─── TEST 1: Gradient Flow S=250, report every 50 ────────────────────────────
fn test_gradients() {
    let device = Default::default();
    let seq_len    = 250usize;
    let batch_size = 1usize;
    let input_dim  = 16usize;
    let expansion  = 2usize;

    let config: MinGruConfig = MinGruConfig::new(input_dim).with_expansion_factor(expansion);
    let model: MinGru<AdBackend> = config.init(&device);

    let x = Tensor::<AdBackend, 3>::random(
        [batch_size, seq_len, input_dim],
        Distribution::Normal(0.0, 0.1),
        &device,
    ).require_grad();


    let (out, _) = model.forward(x.clone(), None);
    // loss sobre TODOS los timesteps (usamos media para estabilidad en S=250)
    let loss = out.mean();
    let grads = loss.backward();
    let x_grad = x.grad(&grads).expect("grad debe existir");

    println!("--- TEST 1: Gradient Flow (S={seq_len}, loss=sum_all) ---");
    let mut checkpoints: Vec<usize> = (0..seq_len).step_by(50).collect();
    checkpoints.push(seq_len - 1);
    checkpoints.dedup();

    for t in &checkpoints {
        let g = x_grad.clone()
            .slice([0..batch_size, *t..*t+1, 0..input_dim])
            .abs().mean().into_scalar();
        println!("  t={t:3}  |grad|={g:.10}");
    }

    let g_first = x_grad.clone().slice([0..batch_size, 0..1, 0..input_dim]).abs().mean().into_scalar();
    let g_last  = x_grad.clone().slice([0..batch_size, seq_len-1..seq_len, 0..input_dim]).abs().mean().into_scalar();
    let ratio   = g_first / (g_last + 1e-30);
    
    println!("  ratio t=0/t={}: {:.4}", seq_len - 1, ratio);

    if g_first > 1e-10 {
        println!("SUCCESS: Gradiente llega al inicio!\n");
    } else {
        println!("FAILURE: Desvanecimiento detectado.\n");
    }
}

// ─── TEST 2: Copy Task — tira FIJA, varios epochs ─────────────────────────────
#[derive(Module, Debug)]
struct CopyModel<B: Backend> {
    embed:  burn::nn::Embedding<B>,
    mingru: MinGru<B>,
    head:   Linear<B>,
}

fn test_copy_task() {
    let device     = Default::default();
    let vocab_size = 10usize;
    let embed_dim  = 16usize;
    let expansion  = 2usize;
    let seq_len    = 32usize;
    let batch_size = 4usize;

    // Tira fija: [0,1,2,...,9,0,1,2,...] * 50 = 500 tokens
    let pattern: Vec<i64> = (0..500i64).map(|i| i % vocab_size as i64).collect();

    // get_batches: igual que el Jupyter
    let total   = pattern.len();
    let n_batch = (total - 1) / (batch_size * seq_len);
    // Para poder tomar y = data[i+1 : i+seq+1], necesitamos n_batch * batch * seq + 1 tokens
    let trimmed = &pattern[..n_batch * batch_size * seq_len + 1];

    let row_len = n_batch * seq_len;
    let mut batches_x: Vec<Vec<i64>> = Vec::new();
    let mut batches_y: Vec<Vec<i64>> = Vec::new();

    for chunk_start in (0..row_len).step_by(seq_len) {
        if chunk_start + seq_len > row_len { break; }
        
        let mut x_batch: Vec<i64> = Vec::new();
        let mut y_batch: Vec<i64> = Vec::new();
        for b in 0..batch_size {
            let offset = b * row_len + chunk_start;
            x_batch.extend_from_slice(&trimmed[offset..offset + seq_len]);
            y_batch.extend_from_slice(&trimmed[offset + 1..offset + seq_len + 1]);
        }
        batches_x.push(x_batch);
        batches_y.push(y_batch);
    }

    let model = CopyModel::<AdBackend> {
        embed:  EmbeddingConfig::new(vocab_size, embed_dim).init(&device),
        mingru: MinGruConfig::new(embed_dim).with_expansion_factor(expansion).init(&device),
        head:   LinearConfig::new(embed_dim, vocab_size).with_bias(false).init(&device),
    };

    let mut optim = AdamConfig::new().init();
    let loss_fn   = burn::nn::loss::CrossEntropyLossConfig::new().init(&device);
    let mut model = model;
    let mut first_loss = 0f64;
    let mut last_loss  = 0f64;

    println!("--- TEST 2: Copy Task (tira fija, steps=200, B={batch_size}, S={seq_len}, vocab={vocab_size}) ---");

    for step in 1..=200usize {
        let mut running = 0f64;
        let mut n = 0usize;

        for (x_ids_raw, y_ids_raw) in batches_x.iter().zip(batches_y.iter()) {
            let x_ids = Tensor::<AdBackend, 2, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(x_ids_raw.clone(), [batch_size, seq_len]), &device,
            );
            let y_ids = Tensor::<AdBackend, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(y_ids_raw.clone(), [batch_size * seq_len]), &device,
            );

            let embedded = model.embed.forward(x_ids);
            let (out, _) = model.mingru.forward(embedded, None);
            let logits   = model.head.forward(
                out.reshape([batch_size * seq_len, embed_dim])
            );

            let loss = loss_fn.forward(logits, y_ids);
            running += loss.clone().into_data().as_slice::<f64>().unwrap()[0];
            n += 1;

            let grads      = loss.backward();
            if step % 50 == 0 && n == 1 {
                if let Some(g) = model.head.weight.grad(&grads) {
                    let g_val = g.abs().mean().into_scalar();
                    println!("    [GRADIENTE] Step {step} | Head Weight Grad Mean: {g_val:.10}");
                }
            }

            let grads_p    = burn::optim::GradientsParams::from_grads(grads, &model);
            model          = optim.step(2e-3, model, grads_p);
        }

        let avg = running / n as f64;
        if step == 1 { first_loss = avg; }
        last_loss = avg;
        
        if step % 20 == 0 || step == 1 {
            println!("  Step {step:3}  loss={avg:.4}");
        }
    }

    if last_loss < first_loss * 0.5 {
        println!("SUCCESS: Copy Task converge!\n");
    } else {
        println!("FAILURE: Copy Task no converge ({first_loss:.4} -> {last_loss:.4}).\n");
    }
}

fn test_sequential_mode() {
    let device = Default::default();
    let b = 2;
    let seq_len = 5;
    let input_features = 4;
    let expansion_factor = 2;
    
    let config = MinGruConfig::new(input_features).with_expansion_factor(expansion_factor);
    let model: MinGru<TestBackend> = config.init(&device);
    
    let x = Tensor::<TestBackend, 3>::random(
        [b, seq_len, input_features],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    
    let hidden_size = input_features * expansion_factor;
    let h0 = Tensor::<TestBackend, 3>::zeros([b, 1, hidden_size], &device).add_scalar(0.5);
    
    let (out_par, _) = model.forward(x.clone(), None);
    
    let mut h_seq = h0.clone();
    let mut seq_outs = Vec::new();
    
    for t in 0..seq_len {
        let x_t = x.clone().slice([0..b, t..(t + 1), 0..input_features]);
        let (out_t, h_t) = model.sequential_mode(x_t, h_seq.clone());
        h_seq = h_t;
        seq_outs.push(out_t);
    }
    
    let out_seq = Tensor::cat(seq_outs, 1);
    let diff = (out_par.clone() - out_seq).abs().max().into_scalar();
    println!("--- TEST 3: Sequential vs Parallel Mode Equivalence ---");
    println!("Max diff: {}", diff);
    if diff < 1e-5 {
        println!("SUCCESS: Sequential mode matches parallel mode logic.\n");
    } else {
        println!("FAILURE: Sequential mode differs greatly from parallel ({})\n", diff);
    }
}

fn main() {
    println!("=== minGRU Tests (Rust --release) ===\n");
    test_gradients();
    test_copy_task();
    test_sequential_mode();
}
