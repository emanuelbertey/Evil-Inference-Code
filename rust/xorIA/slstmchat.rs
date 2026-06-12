#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization
Updated to use the new xLSTM Rust port with mLSTM and sLSTM blocks.
*/

use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, Optimizer, GradientsParams},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::Backend, Int},
    nn::loss::CrossEntropyLossConfig,
};
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use burn_ndarray::NdArray;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::{XLstm, XLstmConfig};
use xlstm::xlstm_block_stack::XLSTMBlockStackConfig;
use xlstm::blocks::mlstm::block::MLSTMBlockConfig;
use xlstm::blocks::mlstm::layer::MLSTMLayerConfig;
use xlstm::blocks::slstm::block::SLSTMBlockConfig;
use xlstm::blocks::slstm::layer::SLSTMLayerConfig;
use xlstm::components::feedforward::GatedFeedForwardConfig;

// Use NdArray backend with Autodiff
type MyBackend = Autodiff<NdArray<f32>>;

/// Professional Tokenizer using Hugging Face 'tokenizers'
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| format!("Error building BPE: {}", e))?;
            
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("{}", e))?;
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

/// Create training batch (B, S)
fn create_batch<B: Backend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i;
        for j in 0..seq_length {
            let x_idx = if current_start + j < tokens.len() { tokens[current_start + j] } else { 0 };
            let y_idx = if current_start + j + 1 < tokens.len() { tokens[current_start + j + 1] } else { 0 };
            x_indices.push(x_idx as i64);
            y_indices.push(y_idx as i64);
        }
    }

    let x = Tensor::<B, 2, Int>::from_data(
        TensorData::new(x_indices, [batch_size, seq_length]),
        device,
    );
    let y = Tensor::<B, 2, Int>::from_data(
        TensorData::new(y_indices, [batch_size, seq_length]),
        device,
    );

    (x, y)
}

/// Stochastic sampling with Top-K, Top-P (Nucleus) and temperature
fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, 
    temperature: f32,
    top_k: usize,
    top_p: f32
) -> usize {
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs = Vec::with_capacity(k);
    let mut cumulative_prob = 0.0;
    for (i, p) in probs_vec.into_iter() {
        filtered_probs.push((i, p));
        cumulative_prob += p;
        if filtered_probs.len() >= k || cumulative_prob >= top_p {
            break;
        }
    }

    let indices: Vec<usize> = filtered_probs.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = filtered_probs.iter().map(|(_, p)| *p).collect();

    if temperature <= 1e-6 {
        return indices[0];
    }

    for p in weights.iter_mut() {
        *p = (p.max(1e-10).ln() / temperature).exp();
    }

    let sum: f32 = weights.iter().sum();
    use rand::Rng;
    let mut rng = rand::rng(); 
    let sample: f32 = rng.random::<f32>() * sum; 
    let mut cumulative = 0.0;
    for (i, &p) in weights.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative {
            return indices[i];
        }
    }
    indices[0]
}

/// Stateful text generation
fn generate_text<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
) -> String {
    let seed_ids = tokenizer.encode(seed_text);
    if seed_ids.is_empty() {
        return seed_text.to_string();
    }
    
    let mut current_state = model.empty_state(1, device);
    let mut last_id = 0;

    // Warm up state with seed
    for &id in &seed_ids {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![id as i64], [1]), device);
        let (_logits, next_state) = model.step(input, current_state);
        current_state = next_state;
        
        // Use argmax of last token for next token if we were just warming up
        last_id = id;
    }

    let mut result_ids = Vec::new();

    for _ in 0..length {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![last_id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;

        let next_token = sample_from_logits(logits, 0.4, 40, 0.9);
        result_ids.push(next_token);
        last_id = next_token;
        
        // Early stop on newline or similar tokens if needed
        if let Some(t) = tokenizer.id_to_token(next_token) {
             if t.contains('Ċ') { break; }
        }
    }

    tokenizer.decode(&result_ids)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Text Generation - Migrated to new Rust port");
    println!("================================================\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin msltmchat -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer_SLSTM.json";
    let model_path = "slstm_chat_model"; 

    // Load or create tokenizer
    let target_vocab_size = 1024;
    let tokenizer = if Path::new(tokenizer_path).exists() {
        println!("Cargando tokenizador existente...");
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Entrenando nuevo tokenizador BPE...");
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Tamaño del vocabulario: {}\n", vocab_size);

    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    println!("Tokens totales: {}\n", tokens.len());

    // Model configuration
    let embedding_dim = 256;
    let num_blocks = 1;
    let seq_length = 128;
    let batch_size = 24;
    let num_epochs = 50;

    let device = Default::default();

    let mlstm_cfg = MLSTMBlockConfig {
        mlstm: MLSTMLayerConfig {
            embedding_dim,
            num_heads: 4,
            conv1d_kernel_size: 4,
            qkv_proj_blocksize: 4,
            proj_factor: 2.0,
            bias: true,
            dropout: 0.1,
            context_length: seq_length,
        },
    };

    let slstm_cfg = SLSTMBlockConfig {
        slstm: SLSTMLayerConfig {
            embedding_dim,
            num_heads: 4,
            conv1d_kernel_size: 4,
            dropout: 0.1,
        },
        feedforward: Some(GatedFeedForwardConfig {
            embedding_dim,
            proj_factor: 1.3,
            bias: true,
            dropout: 0.1,
        }),
    };

    let config = XLstmConfig {
        vocab_size,
        add_embedding_dropout: true,
        block_stack: XLSTMBlockStackConfig {
            embedding_dim,
            num_blocks,
            context_length: seq_length,
            add_post_blocks_norm: true,
            bias: true,
            dropout: 0.1,
            mlstm_block: Some(mlstm_cfg),
            slstm_block: Some(slstm_cfg),
         //   slstm_at: vec![0, 1, 2],
            slstm_at: vec![0],
        },
    };

    let model_file = format!("{}.mpk", model_path);
    let mut model: xlstm::XLstm<MyBackend> = if Path::new(&model_file).exists() {
        println!("¡Modelo encontrado! Cargando pesos...");
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.into(), &device)?;
        config.init(&device).load_record(record)
    } else {
        println!("Iniciando entrenamiento desde cero...\n");
        config.init(&device)
    };

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let tokens_per_batch = batch_size * seq_length;
    let num_batches = tokens.len() / tokens_per_batch;

    println!("Entrenando con {} batches por época (Stride optimizado)\n", num_batches);

    let total_start = Instant::now();

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * tokens_per_batch;
            
            // Sequential batching for stability on CPU
            let mut x_indices = Vec::with_capacity(tokens_per_batch);
            let mut y_indices = Vec::with_capacity(tokens_per_batch);
            for i in 0..batch_size {
                let base = start_idx + i * seq_length;
                for j in 0..seq_length {
                    let idx = (base + j) % tokens.len();
                    let next_idx = (base + j + 1) % tokens.len();
                    x_indices.push(tokens[idx] as i64);
                    y_indices.push(tokens[next_idx] as i64);
                }
            }
            let input = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), &device);
            let targets = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), &device);

            let logits = model.forward(input);
            let [b, s, v] = logits.dims();
            
            let logits_flat = logits.reshape([b * s, v]);
            let targets_flat = targets.reshape([b * s]);

            let loss = loss_fn.forward(logits_flat, targets_flat);
            total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            
            let lr = 2e-3; // Stability for sLSTM
            model = optim.step(lr, model, grads);

            if batch_idx % 5 == 0 || batch_idx == num_batches - 1 {
                let elapsed_total = total_start.elapsed().as_secs_f32();
                let batches_done = epoch * num_batches + batch_idx + 1;
                let total_batches = num_epochs * num_batches;
                let tps = (batches_done * tokens_per_batch) as f32 / elapsed_total;
                
                let remaining_batches = total_batches.saturating_sub(batches_done);
                let eta_min = (remaining_batches * tokens_per_batch) as f32 / tps / 60.0;

                print!("\r  Epoch [{}/{}] Batch [{}/{}] Loss: {:.4} | Speed: {:.1} tok/s | Total ETA: {:.1}h", 
                    epoch+1, num_epochs, batch_idx+1, num_batches, total_loss / (batch_idx+1) as f32, tps, eta_min / 60.0);
                io::stdout().flush()?;
            }
        }

        println!("\n  Epoch completada en {:.2}s. Generando ejemplo...", epoch_start.elapsed().as_secs_f32());
        let generated = generate_text(&model.valid(), &tokenizer, "El ", 64, &device);
        println!("  Generado: {}\n", generated);

        let recorder = CompactRecorder::new();
        model.clone().save_file(model_path, &recorder)?;
    }

    println!("\n¡Entrenamiento completado!");

    // Interactive Loop
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO - GENERACIÓN DE TEXTO         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    loop {
        print!("Semilla > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        let generated = generate_text(&model.valid(), &tokenizer, input, 200, &device);
        println!("\n--- TEXTO GENERADO ---\n{}\n----------------------\n", generated);
    }

    Ok(())
}
