#![recursion_limit = "256"]

/*!
xLSTM Pro Trainer - High Performance Data Streaming with WGPU
    
This tool is designed to train xLSTM models on large datasets (directories, 
multiple file formats, and gigabyte-sized files) without exhausting RAM.
Updated to use the new mLSTM architecture with 2 blocks on GPU.
Streaming: processes files in 10 MB fragments, training starts immediately
without waiting for the whole file to load.
*/

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::AutodiffModule,
    module::Module,
    optim::{AdamConfig, Optimizer, GradientsParams},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::Backend, Int},
    nn::loss::CrossEntropyLossConfig,
};
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::error::Error;
use std::fs;
use std::io::{self, Write, Read, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use xlstm::{XLstm, XLstmConfig};
use xlstm::xlstm_block_stack::XLSTMBlockStackConfig;
use xlstm::blocks::mlstm::block::MLSTMBlockConfig;
use xlstm::blocks::mlstm::layer::MLSTMLayerConfig;
use xlstm::blocks::slstm::block::SLSTMBlockConfig;
use xlstm::blocks::slstm::layer::SLSTMLayerConfig;
use xlstm::components::feedforward::GatedFeedForwardConfig;
use std::collections::HashSet;

// backend con autodiff y wgpu
type MyBackend = Autodiff<Wgpu<f32, i32>>;

/// Professional BPE Tokenizer
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
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        // Agregar token especial <|endoftext|>
        let special_token = "<|endoftext|>";
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from(special_token, true)]);

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect::<HashSet<_>>())
            .special_tokens(vec![tokenizers::AddedToken::from(special_token, true)])
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train_pro_gpu.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error in training: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true).map_err(|e| format!("Error saving: {}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("Error loading: {}", e))?;
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
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

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.tokenizer.token_to_id(token).map(|id| id as usize)
    }
}

/// Recursively find files with specific extensions
fn find_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_files(&path, extensions));
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    files.push(path);
                }
            }
        }
    }
    files
}

/// Iterator for large files that yields fragments (buffers) to save RAM
struct FileFragmentIterator {
    reader: BufReader<fs::File>,
    buffer_size: usize,
    finished: bool,
}

impl FileFragmentIterator {
    fn new(path: &Path, buffer_size_mb: usize) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            buffer_size: buffer_size_mb * 1024 * 1024,
            finished: false,
        })
    }
}

impl Iterator for FileFragmentIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }

        let mut buffer = vec![0u8; self.buffer_size];
        let mut total_read = 0;

        while total_read < self.buffer_size {
            match self.reader.read(&mut buffer[total_read..]) {
                Ok(0) => { self.finished = true; break; }
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.finished = true; break; }
            }
        }

        if total_read == 0 { return None; }
        buffer.truncate(total_read);

        // Ensure we don't split a UTF-8 character
        while !buffer.is_empty() && String::from_utf8(buffer.clone()).is_err() {
            buffer.pop();
        }

        if buffer.is_empty() { return None; }
        String::from_utf8(buffer).ok()
    }
}

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

    let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), device);
    let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), device);
    (x, y)
}

fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, 
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    previous_tokens: &[usize],
) -> usize {
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    if repetition_penalty != 1.0 {
        for (id, prob) in probs_vec.iter_mut() {
            if previous_tokens.contains(id) {
                *prob /= repetition_penalty;
            }
        }
    }

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

    if temperature <= 1e-6 { return indices[0]; }

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
        if sample <= cumulative { return indices[i]; }
    }
    indices[0]
}

fn generate_text<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
) -> String {
    let seed_ids = tokenizer.encode(seed_text);
    if seed_ids.is_empty() { return seed_text.to_string(); }

    let mut current_state = model.empty_state(1, device);
    let mut last_id = 0;
    let mut last_logits = None;

    for &id in &seed_ids {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;
        last_id = id;
        last_logits = Some(logits);
    }

    let mut result_ids = Vec::new();
    let mut history = seed_ids.clone();

    let temp = 0.7;
    let top_k = 40;
    let top_p = 0.9;
    let r_penalty = 1.1;

    let mut next_id = if let Some(logits) = last_logits {
        sample_from_logits(logits, temp, top_k, top_p, r_penalty, &history)
    } else {
        last_id
    };

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "<|endoftext|>" { break; }
        }
        result_ids.push(next_id);
        history.push(next_id);
        if history.len() > 128 { history.remove(0); }

        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next_id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;
        next_id = sample_from_logits(logits, temp, top_k, top_p, r_penalty, &history);
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let text = tokenizer.decode(&result_ids);
    println!("  [Generación] Tokens generados: {} en {:.2}s", result_ids.len(), elapsed);
    text
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Pro Trainer (WGPU Version)");
    println!("================================");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin pro_trainer -- <archivo_o_carpeta>");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let tokenizer_path = "tokenizer_pro.json";
    let model_base_path = "xlstm_pro_wgpu";
    let extensions = ["txt", "gd", "gdshader"];

    let all_files = if input_path.is_dir() {
        find_files(input_path, &extensions)
    } else {
        vec![input_path.to_path_buf()]
    };

    if all_files.is_empty() {
        return Err("No se encontraron archivos válidos.".into());
    }
    println!("Archivos encontrados: {}", all_files.len());

    let target_vocab_size = 4096;
    let tokenizer = if Path::new(tokenizer_path).exists() {
        println!("Cargando tokenizador...");
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("🚀 Entrenando nuevo tokenizador Pro...");
        let mut sample_text = String::new();
        for file_path in all_files.iter().take(100) {
            if let Ok(mut file) = fs::File::open(file_path) {
                let mut content = String::new();
                let mut reader = BufReader::new(file).take(512_000);
                if reader.read_to_string(&mut content).is_ok() {
                    sample_text.push_str(&content);
                    sample_text.push_str("<|endoftext|>");
                }
            }
        }
        let t = Tokenizer::from_text(&sample_text, target_vocab_size)?;
        t.save(tokenizer_path)?;
        t
    };

    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 256;
    let num_blocks = 2;
    let seq_length = 256;
    let batch_size = 16;
    let num_epochs = 30;

    let device = WgpuDevice::default();

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
            slstm_at: vec![], // Solo mLSTM
        },
    };

    let model_file = format!("{}.mpk", model_base_path);
    let existe_modelo = Path::new(&model_file).exists();
    
    let mut modo_inferencia = false;
    if existe_modelo {
        loop {
            print!("\n¡Modelo GPU encontrado! ¿Deseas (e)ntrenar o (i)nferir solamente? [e/i]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();
            match choice.as_str() {
                "i" => { modo_inferencia = true; break; }
                "e" => { break; }
                _ => continue,
            }
        }
    }

    let mut model: XLstm<MyBackend> = if existe_modelo {
        println!("Cargando modelo...");
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.into(), &device)?;
        config.init(&device).load_record(record)
    } else {
        println!("Iniciando nuevo modelo...");
        config.init(&device)
    };

    if modo_inferencia {
        println!("\n--- MODO INFERENCIA ---");
        loop {
            print!("\nSemilla > ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            if input == "salir" { break; }
            if input.is_empty() { continue; }

            println!("Generando...");
            let gen = generate_text(&model.valid(), &tokenizer, input, 200, &device);
            println!("\nRESULTADO:\n----------\n{}\n----------", gen);
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let total_start = Instant::now();
    let mut last_save = Instant::now();
    let save_interval = std::time::Duration::from_secs(4 * 60);
    let lr = 8e-4;

    for epoch in 0..num_epochs {
        println!("\nÉpoca {}/{}", epoch + 1, num_epochs);
        let mut total_loss = 0.0;
        let mut batches_count = 0;

        for (f_idx, file_path) in all_files.iter().enumerate() {
            let fname = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            println!("  [{}/{}] ARCHIVO: {}", f_idx + 1, all_files.len(), fname);

            let fragments = FileFragmentIterator::new(file_path, 1)?; 

            for (frag_idx, fragment) in fragments.enumerate() {
                let tokens = tokenizer.encode(&fragment);
                let tokens_per_batch = batch_size * seq_length;
                let num_batches = tokens.len() / tokens_per_batch;
                
                if num_batches == 0 { continue; }

                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * tokens_per_batch;
                    
                 //   println!("    [Batch {}] PRE create_batch", batch_idx + 1);
                    let (input, targets) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_length, &device);

                  //  println!("    [Batch {}] PRE forward", batch_idx + 1);
                    let logits = model.forward(input);
                    let [b, s, v] = logits.dims();
                    
                 //   println!("    [Batch {}] PRE loss", batch_idx + 1);
                    let logits_flat = logits.reshape([b * s, v]);
                    let targets_flat = targets.reshape([b * s]);
                    let loss = loss_fn.forward(logits_flat, targets_flat);

                  //  println!("    [Batch {}] PRE sync (loss into_data)", batch_idx + 1);
                    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                    
                    // --- Detección de NaN/Inf ---
                    let avg_loss = total_loss / batches_count.max(1) as f32;
                    if loss_val.is_nan() || loss_val.is_infinite() || avg_loss.is_nan() || avg_loss.is_infinite() {
                        println!("\n🚨 ERROR: ¡Detectado NaN o Inf en Loss ({:.4}, avg: {:.4})! Deteniendo...", loss_val, avg_loss);
                        std::process::exit(1);
                    }

                    total_loss += loss_val;
                    batches_count += 1;

              //      println!("    [Batch {}] PRE backward", batch_idx + 1);
                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    
                 //   println!("    [Batch {}] PRE step", batch_idx + 1);
                    model = optim.step(lr as f64, model, grads);

                    let elapsed = total_start.elapsed().as_secs_f32();
                    let tps = (batches_count * tokens_per_batch) as f32 / elapsed;
                    let avg_loss = total_loss / batches_count as f32;

                    println!("    [Batch {}] OK | Loss: {:.4} | Avg: {:.4} | Speed: {:.1} tok/s", 
                        batch_idx + 1, loss_val, avg_loss, tps);
                    io::stdout().flush()?;

                    // --- Autoguardado cada 4 min (si todo está bien) ---
                    if last_save.elapsed() >= save_interval {
                        println!("\n💾 [4 min] Guardando modelo preventivamente...");
                        let recorder = CompactRecorder::new();
                        model.clone().save_file(model_base_path, &recorder).ok();
                        last_save = Instant::now();
                    }
                }
                println!();
            }
        }

        println!("\n  Guardando modelo...");
        let recorder = CompactRecorder::new();
        model.clone().save_file(model_base_path, &recorder)?;

        let gen = generate_text(&model.valid(), &tokenizer, "func _ready():", 100, &device);
        println!("  Ejemplo: {}", gen);
    }

    Ok(())
}
