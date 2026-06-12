#![recursion_limit = "256"]

/*!
Text Generation with xLSTMLarge using the new optimized architecture.
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

use xlstm::blocks::xlstm_large::{XLSTMLarge, XLSTMLargeConfig};

// Use NdArray backend with Autodiff (CPU)
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
        let temp_file = "temp_train_large.txt";
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
        Some(String::from_utf8_lossy(&buffer).into_owned())
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
        let current_start = start_idx + i * seq_length;
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

fn generate_text_typed<B: Backend>(
    model: &XLSTMLarge<B>,
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

    // --- PARALLEL PREFILL ---
    // Enviamos todo el seed de golpe en modo paralelo. Es mucho más rápido.
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(seed_ids.iter().map(|&id| id as i64).collect(), [1, seed_ids.len()]), 
        device
    );
    // Al pasar None, el backend activa la ejecución paralela y ahora devuelve el estado final corregido.
    let (logits, next_state) = model.forward(input, None);
    current_state = next_state.expect("El prefill paralelo debe devolver el estado");
    
    // Solo nos interesan los logits del ÚLTIMO token para empezar a samplear
    let [_, s_len, v_dim] = logits.dims();
    last_logits = Some(logits.slice([0..1, (s_len - 1)..s_len]).reshape([1, v_dim]));

    let mut result_ids = Vec::new();
    let mut history = seed_ids.clone();

    let top_k = 20;
    let top_p = 1.0;

    let mut next_id = if let Some(logits) = last_logits {
        // logits is [1, V], sample_from_logits expects [1, V]
        let [_, v] = logits.dims();
        sample_from_logits(logits.clone(), temp, top_k, top_p, r_penalty, &history)
    } else {
        0
    };

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "<|endoftext|>" { 
                println!("=================<|endoftext|>======================\n");
                break; 
            }
        }

        result_ids.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_id as i64], [1, 1]), device);
        let (logits, next_state) = model.forward(input, Some(current_state));
        current_state = next_state.expect("State should be returned in recurrent mode");
        
        let [_, _, v] = logits.dims();
        next_id = sample_from_logits(logits.reshape([1, v]), temp, top_k, top_p, r_penalty, &history);
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let text = tokenizer.decode(&result_ids);
    (text, result_ids.len(), elapsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTMLarge Text Generation - Rust Port");
    println!("========================================\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin large_chat -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let mut tokenizer_path = "llarge_v1_cuda.json".to_string();
    let mut model_file = "llarge_v1_model_cuda.mpk".to_string();

    let mut target_vocab_size = 1024;
    let mut modo_inferencia = false;

    let mut embedding_dim = 256;
    let mut num_blocks = 2;
    let mut num_heads = 4;
    let mut lr = 3e-3;
    let mut num_epochs = 20;
    let mut batch_size = 16;
    let mut temperature = 0.8;
    let mut r_penalty = 1.1;
    let seq_length = 256;
    let mut mode =  "single".to_string();

    let device = Default::default();

    
    loop {
        println!("\n--- CONFIGURACIÓN ACTUAL (xLSTMLarge) ---");
        println!("  (1) Bloques: {}", num_blocks);
        println!("  (2) Heads:   {}", num_heads);
        println!("  (3) LR:      {}", lr);
        println!("  (4) Épocas:  {}", num_epochs);
        println!("  (5) Batch:   {}", batch_size);
        println!("  (6) Temp:    {}", temperature);
        println!("  (7) R-Pen:   {}", r_penalty);
        println!("  (8) Emb dim: {}", embedding_dim);
        println!("  (9) Mode:    {}", mode);
        println!("  (10) Model:  {}", model_file); 
        println!("  (11) token:  {}", target_vocab_size); 
        println!("  (12) pathbpe:{}", tokenizer_path);
        println!("-----------------------------------------");
        print!("¿Entrenar (e), Inferir (i) o Ajustar parámetros (s)? [e/i/s]: ");
        io::stdout().flush()?;
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim().to_lowercase();
        
        if choice == "i" { modo_inferencia = true; break; } 
        else if choice == "e" { break; } 
        else if choice == "s" {
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

            print!("embedding dim [{}]: ", embedding_dim);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if let Ok(v) = input.trim().parse() { embedding_dim = v; }

            print!("Mode [{}]: ", mode);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let v = input.trim();
            if !v.is_empty() { mode = v.to_string(); }

            print!("Model Path [{}]: ", model_file);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let v = input.trim();
            if !v.is_empty() { model_file = v.to_string(); }

            print!("Vocab Size [{}]: ", target_vocab_size);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let v = input.trim();
            if !v.is_empty() { if let Ok(parsed) = v.parse() { target_vocab_size = parsed; } }

            print!("Tokenizer Path [{}]: ", tokenizer_path);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let v = input.trim();
            if !v.is_empty() { tokenizer_path = v.to_string(); }
        }
    }
   // let filbpe = tokenizer_path.to_string();
    let tokenizer = if Path::new(&tokenizer_path.to_string()).exists() { 
        println!("Cargando tokenizador existente...");
        Tokenizer::load(&tokenizer_path.to_string())?
    } else {
        println!("Entrenando nuevo tokenizador BPE...");
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(&tokenizer_path.to_string())?;
        tokenizer
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Tamaño del vocabulario: {}\n", vocab_size);
    let filemod = model_file.to_string().clone();
    let existe_modelo = Path::new(&filemod).exists();
    


    let config = XLSTMLargeConfig {
        embedding_dim,
        num_heads,
        num_blocks,
        vocab_size,
        use_bias: true,
        norm_eps: 1e-6,
        norm_reduction_force_float32: true,
        add_out_norm: true,
        qk_dim_factor: 0.5,
        v_dim_factor: 1.0,
        mlstm_backend: xlstm::blocks::xlstm_large::config::MLSTMBackendConfig::new(),
        ffn_proj_factor: 2.6667,
        ffn_round_up_to_multiple_of: 64,
        gate_soft_cap: Some(15.0),
        output_logit_soft_cap: Some(30.0),
        weight_mode: mode.to_string(),
    };

    let mut model: XLSTMLarge<MyBackend> = if existe_modelo {
        println!("Cargando pesos del modelo...");
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.clone().into(), &device)?;
        XLSTMLarge::init(&config, &device).load_record(record)
    } else {
        println!("Iniciando nuevo modelo desde cero...\n");
        XLSTMLarge::init(&config, &device)
    };

    if !modo_inferencia {
        let mut optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init();

        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let tokens_per_batch = batch_size * seq_length;

        let total_train_start = Instant::now();
        let mut last_save = Instant::now();
        let save_interval = std::time::Duration::from_secs(4 * 60);

        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let fragments = FileFragmentIterator::new(Path::new(text_file), 1)?;
            let mut batch_count = 0;
            let epoch_start = Instant::now();

            for fragment in fragments {
                let tokens = tokenizer.encode(&fragment);
                let num_batches = tokens.len() / tokens_per_batch;
                if num_batches == 0 { continue; }

                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * tokens_per_batch;
                    let (input, targets) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_length, &device);

                    let (logits, _) = model.forward(input, None);
                    let [b, s, v] = logits.dims();
                    let loss = loss_fn.forward(logits.reshape([b * s, v]), targets.reshape([b * s]));
                    
                    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                    if loss_val.is_nan() || loss_val.is_infinite() {
                        println!("🚨 Error: Loss es NaN/Inf");
                        std::process::exit(1);
                    }
                    total_loss += loss_val;
                    batch_count += 1;

                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optim.step(lr as f64, model, grads);

                    // Autoguardado cada 4 minutos
                    if last_save.elapsed() >= save_interval {
                        println!("\n💾 Autoguardado preventivo (cada 4 min)...");
                        let recorder = CompactRecorder::new();
                        model.clone().save_file(model_file.clone(), &recorder).ok();
                        last_save = Instant::now();
                    }

                    if batch_idx % 1 == 0 || batch_idx == num_batches - 1 {
                        let total_elapsed = total_train_start.elapsed().as_secs_f32();
                        // Estimate total batches based on processed so far or first epoch
                        // batches_per_epoch is not known exactly due to streaming, 
                        // but we can estimate: (batch_idx+1) in this fragment, but we want absolute progress.
                        // For a rough ETA, we'll use the total elapsed and current epoch progress.
                        
                        let epoch_progress = (epoch as f32) / (num_epochs as f32);
                        // Within epoch, we don't know total batches but let's assume num_batches*total_fragments
                        // A better way: total_elapsed / amount_processed * amount_remaining
                        
                        let tps = (batch_count * tokens_per_batch) as f32 / epoch_start.elapsed().as_secs_f32().max(0.001);
                        let avg_loss = total_loss / batch_count as f32;

                        // Very rough ETA: (processed_so_far / total_expected)
                        // Let's just use a simple time per epoch estimate.
                        let time_per_epoch = total_elapsed / (epoch as f32 + (batch_idx as f32 / num_batches as f32).min(1.0)).max(0.001);
                        let remaining_epochs = (num_epochs as f32) - (epoch as f32 + (batch_idx as f32 / num_batches as f32).min(1.0));
                        let eta_h = (remaining_epochs * time_per_epoch) / 3600.0;

                        print!("\r  Epoch [{}/{}] Batch [{}/{}] Loss: {:.4} | Avg: {:.4} | Speed: {:.1} tok/s | ETA: {:.1}h   ", 
                            epoch+1, num_epochs, batch_idx+1, num_batches, loss_val, avg_loss, tps, eta_h);
                        io::stdout().flush()?;
                    }
                }
            }
            let epoch_duration = epoch_start.elapsed().as_secs_f32();
            println!("\n  Epoch completada en {:.2}s. Generando ejemplo...", epoch_duration);
            
            let (generated, tokens_count, elapsed_gen) = generate_text_typed(&model.valid(), &tokenizer, "El ", 64, &device, temperature, r_penalty);
            let tps_gen = tokens_count as f32 / elapsed_gen.max(0.001);
            println!("  Generado ({:.1} tok/s): {}\n", tps_gen, generated);

            let recorder = CompactRecorder::new();
            print!("Large model file  [file: {}] > ", model_file.clone()); 
            model.clone().save_file(model_file.to_string().clone(), &recorder)?;
        }
    }

    println!("\n--- MODO INTERACTIVO ---");
    let mut current_len = 200;
    loop {
        print!("Large Chat [len: {}] > ", current_len); 
        io::stdout().flush()?;
        let mut input = String::new(); 
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }
        if input.is_empty() { continue; }

        if input.to_lowercase().starts_with("len ") {
            if let Ok(new_len) = input[4..].trim().parse::<usize>() {
                current_len = new_len;
                println!("  -> Longitud cambiada a: {} tokens.\n", current_len);
                continue;
            }
        }

        let (generated, tokens, elapsed) = generate_text_typed(&model.valid(), &tokenizer, input, current_len, &device, temperature, r_penalty);
        let tps = tokens as f32 / elapsed.max(0.001);
        
        println!("\n--- TEXTO GENERADO (xLSTMLarge) ---");
        println!("{}", generated);
        println!("-----------------------------------");
        println!("Tokens: {} | Tiempo: {:.2}s | Velocidad: {:.2} tok/s\n", tokens, elapsed, tps);
    }

    Ok(())
}
