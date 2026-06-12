#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization (GPU version)
Updated to use the new xLSTM Rust port with 1 mLSTM block on WebGPU.
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
use burn_ndarray::NdArray;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::AddedToken;
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

// Use NdArray backend with Autodiff (CPU)

//type MyBackend = Autodiff<NdArray<f64>>;

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

        let special_token = "<|endoftext|>";
        tokenizer.add_special_tokens(&[AddedToken::from(special_token, true)]);

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from(special_token, true)
            ])
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train_gpu.txt";
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

struct FileFragmentIterator {
    reader: io::BufReader<fs::File>,
    buffer_size: usize,
    finished: bool,
}

impl FileFragmentIterator {
    fn new(path: &Path, buffer_size_mb: usize) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        Ok(Self {
            reader: io::BufReader::new(file),
            buffer_size: buffer_size_mb * 2048 * 1024,
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
        use std::io::Read;
        while total_read < self.buffer_size {
            match self.reader.read(&mut buffer[total_read..]) {
                Ok(0) => { self.finished = true; break; }
                Ok(n) => total_read += n,
                _ => { self.finished = true; break; }
            }
        }
        if total_read == 0 { return None; }
        buffer.truncate(total_read);
        
        // Debug: Imprimir 1KB de dato crudo del disco
        let debug_len = total_read.min(1024);
        let raw_preview = String::from_utf8_lossy(&buffer[..debug_len]);
        println!("\n--- [LECTURA CRUDA DISCO - 1KB] ---");
        println!("{}", raw_preview);
        println!("--- [FIN LECTURA CRUDA] ---\n");

        Some(String::from_utf8_lossy(&buffer).into_owned())
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
        let current_start = start_idx + i * seq_length;
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

/// Stochastic sampling with Top-K, Top-P, Temperature and Repetition Penalty
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

    // Aplicar penalización de repetición
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
fn generate_text_typed<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
    temp: f32,
    r_penalty: f32,
) -> (String, usize, f32) {
    let seed_ids = tokenizer.encode(seed_text);
    if seed_ids.is_empty() {
        return (seed_text.to_string(), 0, 0.0);
    }
    
    let start_time = Instant::now();
    let mut current_state = model.empty_state(1, device);
    let mut last_logits = None;

    // Warm up state with seed
    for &id in &seed_ids {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;
        last_logits = Some(logits);
    }

    let mut result_ids = Vec::new();
    let mut history = seed_ids.clone();

    // Parámetros limpios para evitar que se rompan las palabras con vocab=2048
    let top_k = 20;
    let top_p = 1.0;

    // Primer token: samplear desde los logits del último paso
    let mut next_id = if let Some(logits) = last_logits {
        sample_from_logits(logits, temp, top_k, top_p, r_penalty, &history)
    } else {
        0
    };

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "<|endoftext|>" { break; }
        }
        result_ids.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next_id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;
        next_id = sample_from_logits(logits, temp, top_k, top_p, r_penalty, &history);
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let text = tokenizer.decode(&result_ids);
    println!("  [Generación] Tokens generados: {} en {:.2}s", result_ids.len(), elapsed);
    (text, result_ids.len(), elapsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Text Generation - GPU Port (WGPU)");
    println!("========================================\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin msltmchat_gpu -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "min.json";
    let model_base_path = "min"; 
    let model_file = format!("{}.mpk", model_base_path);

    // Load or create tokenizer
    let target_vocab_size = 2048;
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

    let mut embedding_dim = 256;
    let mut num_blocks = 2;
    let mut num_heads = 4;
    let mut lr = 8e-4;
    let mut num_epochs = 30;
    let mut batch_size = 16;
    let mut temperature = 0.8;
    let mut r_penalty = 1.1;
    let seq_length = 256; // Bajado de 256 para estabilidad inicial en GPU

    let device = Default::default();
    println!("Backend CPU (NdArray) inicializado.");
    
    let existe_modelo = Path::new(&model_file).exists();
    
    let mut modo_inferencia = false;
    
    loop {
        println!("\n--- CONFIGURACIÓN ACTUAL ---");
        println!("  (1) Bloques: {}", num_blocks);
        println!("  (2) Heads:   {}", num_heads);
        println!("  (3) LR:      {}", lr);
        println!("  (4) Épocas:  {}", num_epochs);
        println!("  (5) Batch:   {}", batch_size);
        println!("  (6) Temp:    {}", temperature);
        println!("  (7) R-Pen:   {}", r_penalty);
        println!("----------------------------");
        print!("¿Entrenar (e), Inferir (i) o Ajustar parámetros (s)? [e/i/s]: ");
        io::stdout().flush()?;
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim().to_lowercase();
        
        if choice == "i" {
            modo_inferencia = true;
            break;
        } else if choice == "e" {
            break;
        } else if choice == "s" {
            println!("\nAjustar parámetros (Enter para mantener actual):");
            
            print!("Cant. Bloques [{}]: ", num_blocks);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { num_blocks = v; }

            print!("Cant. Heads [{}]: ", num_heads);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { num_heads = v; }

            print!("Learning Rate [{}]: ", lr);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { lr = v; }

            print!("Épocas [{}]: ", num_epochs);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { num_epochs = v; }

            print!("Batch Size [{}]: ", batch_size);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { batch_size = v; }

            print!("Temperatura [{}]: ", temperature);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { temperature = v; }

            print!("Repetition Penalty [{}]: ", r_penalty);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { r_penalty = v; }
        }
    }

    let mlstm_cfg = MLSTMBlockConfig {
        mlstm: MLSTMLayerConfig {
            embedding_dim,
            num_heads,
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
            num_heads,
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
            slstm_at: vec![], // Only mLSTM
        },
    };

    let mut model: xlstm::XLstm<MyBackend> = if existe_modelo {
        println!("Cargando pesos del modelo GPU...");
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.clone().into(), &device)?;
        config.init(&device).load_record(record)
    } else {
        println!("Iniciando nuevo modelo GPU desde cero...\n");
        config.init(&device)
    };

    if !modo_inferencia {
        let mut optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init();

        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let tokens_per_batch = batch_size * seq_length;

        println!("Entrenando en CPU con Streaming de 1MB por fragmento...");

        let total_start = Instant::now();
        let mut last_save = Instant::now();
        let save_interval = std::time::Duration::from_secs(4 * 60);

        for epoch in 0..num_epochs {
            let epoch_start = Instant::now();
            let mut total_loss = 0.0;

            let fragments = FileFragmentIterator::new(Path::new(text_file), 1)?;

            let mut frag_count = 0;
            for fragment in fragments {
                frag_count += 1;
                let tokens = tokenizer.encode(&fragment);
                let num_batches = tokens.len() / tokens_per_batch;
                
                let preview = fragment.chars().take(256).collect::<String>();
                println!("\n  --- Fragmento {}: {} tokens ---", frag_count, tokens.len());
                println!("  Preview: {}...", preview.replace('\n', " "));
                if num_batches == 0 { continue; }

                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * tokens_per_batch;
                    
                    let (input, targets) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_length, &device);

                    let logits = model.forward(input);
                    let [b, s, v] = logits.dims();
                    
                    let logits_flat = logits.reshape([b * s, v]);
                    let targets_flat = targets.reshape([b * s]);

                    let loss = loss_fn.forward(logits_flat, targets_flat);
                    
                    let current_loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                    
                    // --- DETECCIÓN DE NaN / INF ---
                    if current_loss_val.is_nan() || current_loss_val.is_infinite() {
                        println!("\n🚨 ERROR: ¡Detectado NaN o Inf en la Loss ({:.4})! Deteniendo...", current_loss_val);
                        std::process::exit(1);
                    }
                    
                    total_loss += current_loss_val;

                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    
                    model = optim.step(lr as f64, model, grads);

                    // --- AUTOGUARDADO CADA 4 MINUTOS ---
                    if last_save.elapsed() >= save_interval {
                        println!("\n💾 Autoguardado preventivo (cada 4 min)...");
                        let recorder = CompactRecorder::new();
                        model.clone().save_file(model_file.clone(), &recorder).ok();
                        last_save = Instant::now();
                    }

                    if batch_idx % 1 == 0 || batch_idx == num_batches - 1 {
                        let elapsed_total = total_start.elapsed().as_secs_f32();
                        let batches_done = epoch * num_batches + batch_idx + 1;
                        let total_batches_all = num_epochs * num_batches;
                        let tps = (batches_done * tokens_per_batch) as f32 / elapsed_total;
                        
                        let remaining_batches = total_batches_all.saturating_sub(batches_done);
                        let eta_min = (remaining_batches * tokens_per_batch) as f32 / tps / 60.0;

                        let avg_loss = total_loss / (batch_idx + 1) as f32;

                        print!("\r  Epoch [{}/{}] Batch [{}/{}] Loss: {:.4} | Avg: {:.4} | Speed: {:.1} tok/s | ETA: {:.1}h      ", 
                            epoch+1, num_epochs, batch_idx+1, num_batches, current_loss_val, avg_loss, tps, eta_min / 60.0);
                        io::stdout().flush()?;
                    }
                }
            }
            println!(); // Salto de línea al final de la época

            println!("\n  Epoch completada en {:.2}s. Generando ejemplo...", epoch_start.elapsed().as_secs_f32());
            let (generated, _, _) = generate_text_typed(&model.valid(), &tokenizer, "El ", 64, &device, temperature, r_penalty);
            println!("  Generado: {}\n", generated);

            let recorder = CompactRecorder::new();
            model.clone().save_file(model_file.clone(), &recorder)?;
        }
        println!("\n¡Entrenamiento GPU completado!");
    } else {
        println!("Entrando en modo de inferencia pura (GPU).\n");
    }

    // Interactive Loop
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO GPU - GENERACIÓN             ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("Comandos:");
    println!("  - Escribe tu semilla para generar texto.");
    println!("  - 'len <n>': Cambia la cantidad de tokens a generar.");
    println!("  - 'salir' o 'exit' para terminar.\n");

    let mut current_len = 200;

    loop {
        print!("GPU Chat [len: {}] > ", current_len);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.to_lowercase().starts_with("len ") {
            if let Ok(new_len) = input[4..].trim().parse::<usize>() {
                current_len = new_len;
                println!("  -> Longitud cambiada a: {} tokens.\n", current_len);
                continue;
            }
        }

        if input.is_empty() {
            continue;
        }

        let (generated, tokens_count, elapsed) = generate_text_typed(&model.valid(), &tokenizer, input, current_len, &device, temperature, r_penalty);
        let tps = tokens_count as f32 / elapsed;
        
        println!("\n--- TEXTO GENERADO (GPU) ---");
        println!("{}", generated);
        println!("----------------------------");
        println!("Tokens: {} | Tiempo: {:.2}s | Velocidad: {:.2} tok/s\n", tokens_count, elapsed, tps);
    }

    Ok(())
}
