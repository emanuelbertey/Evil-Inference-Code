// ─── Transformer Chat: BPE-Level Language Model ──────────────────────────────
//
// Uses the custom Transformer module with GQA + RoPE + SwiGLU for
// BPE-level text generation using Hugging Face 'tokenizers'.
// Trains on input.txt and provides an interactive generation mode.
//
// Architecture:
//   Embedding → Transformer(N layers × GQA+RoPE+SwiGLU) → Linear → logits
//
// Features:
//   - KV Cache for fast autoregressive generation
//   - Top-K / Top-P sampling
//   - Repetition Penalty
//   - Flex backend (CPU/Portable acceleration)
//
// Usage:
//   cargo run --bin transformer_chat --release -- xorIA/input.txt

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{Module, AutodiffModule},
    optim::{AdamConfig, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::Backend, TensorData, Int},
    nn::loss::CrossEntropyLossConfig,
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
};
use burn_autodiff::Autodiff;
use burn_flex::Flex;
use std::error::Error;
use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::collections::{HashMap, BTreeSet};
use std::time::Instant;

use tokenizers::AddedToken;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::blocks::trasformer::layer::{
    Transformer, TransformerConfig, TransformerLayerConfig,
};
use xlstm::blocks::trasformer::attention::KVCache;

type MyBackend = Autodiff<Flex<f32>>;

// ─── BPE Tokenizer ──────────────────────────────────────────────────────────

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
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, false)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, false)));

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
        let temp_file = "temp_train_transformer.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento de tokenizer: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("{}", e))?;
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, false)));
        Ok(Self { tokenizer })
    }

    pub fn load_pretrained(path: &str) -> Result<Self, Box<dyn Error>> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("{}", e))?;
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

// ─── File Fragment Iterator (Streaming) ─────────────────────────────────────

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

        while !buffer.is_empty() && String::from_utf8(buffer.clone()).is_err() {
            buffer.pop();
        }

        if buffer.is_empty() { return None; }
        String::from_utf8(buffer).ok()
    }
}

// ─── Language Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerLM<B: Backend> {
    pub embedding: Embedding<B>,
    pub transformer: Transformer<B>,
    pub head: Linear<B>,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
}

impl<B: Backend> TransformerLM<B> {
    /// Standard forward (for training, no cache)
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer.forward(x, 0);
        self.head.forward(x)
    }

    /// Forward with KV cache
    pub fn forward_with_cache(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<Option<KVCache<B>>>,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer.forward_with_cache(x, offset, caches);
        (self.head.forward(x), new_caches)
    }
}

// ─── Batch Creation ─────────────────────────────────────────────────────────

fn create_batch<B: Backend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i * stride;
        for j in 0..seq_length {
            x_indices.push(tokens[current_start + j] as i64);
            y_indices.push(tokens[current_start + j + 1] as i64);
        }
    }

    let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), device);
    let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), device);
    (x, y)
}

// ─── Sampling ───────────────────────────────────────────────────────────────

fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    previous_tokens: &[usize],
) -> usize {
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.into_data()
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

// ─── Text Generation ────────────────────────────────────────────────────────

fn generate_text_cached<B: Backend>(
    model: &TransformerLM<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    mut caches: Vec<Option<KVCache<B>>>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<KVCache<B>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }

    let start_gen = Instant::now();

    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]),
        device,
    );

    let (logits, updated_caches) = model.forward_with_cache(input, current_offset, caches);
    let mut caches = updated_caches;

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim])
        .reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    // Trim rule: if cache length > threshold, remove `remove_count` oldest tokens
    if current_offset >= 255 {
        let remove_count = 160usize; // remove 30 oldest when threshold exceeded
        if let Some(first) = caches.get(0) {
            let mut dims = first.cached_k.dims();
            let mut seq = dims[1];
            if seq > 70 {
                let remove = remove_count.min(seq);
                let keep = seq - remove;
                for c in caches.iter_mut() {
                    *c = c.keep_last(keep);
                }
                current_offset = current_offset.saturating_sub(remove);
                println!("(Cache trimmed: removed {} tokens; kept last {} tokens; new offset: {})", remove, keep, current_offset);
                seq = keep;
            }
        }
    }

    let mut next_id = sample_from_logits(
        last_logits, temperature, top_k, top_p, repetition_penalty, &history,
    );

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "<|endoftext|>" { break; }
        }

        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('▁', " ").replace(' ', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![next_id as i64], [1, 1]),
            device,
        );

        let cache_input: Vec<Option<KVCache<B>>> = caches.into_iter().map(|c| Some(c)).collect();
        let (logits, new_caches) = model.forward_with_cache(input, current_offset, cache_input);
        caches = new_caches;
        current_offset += 1;

        // Trim rule during generation: if cache length > threshold, remove `remove_count` oldest tokens
        if current_offset >= 255 {
            let remove_count = 160usize; // remove 30 oldest when threshold exceeded
            if let Some(first) = caches.get(0) {
                let mut dims = first.cached_k.dims();
                let mut seq = dims[1];
                if seq > 70 {
                    let remove = remove_count.min(seq);
                    let keep = seq - remove;
                    for c in caches.iter_mut() {
                        *c = c.keep_last(keep);
                    }
                    current_offset = current_offset.saturating_sub(remove);
                    println!("(Cache trimmed: removed {} tokens; kept last {} tokens; new offset: {})", remove, keep, current_offset);
                    seq = keep;
            }
        }
    }
        let [_, _, v] = logits.dims();
        let logits_2d = logits.reshape([1, v]);

        next_id = sample_from_logits(
            logits_2d, temperature, top_k, top_p, repetition_penalty, &history,
        );
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

// ─── Main ───────────────────────────────────────────────────────────────────

pub fn transformer_chat() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Chat — GQA + RoPE + SwiGLU                     ║");
    println!("║     BPE-Level Language Model (Hugging Face)                    ║");
    println!("║     + KV Cache + Top-K/P + Repetition Penalty                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 {
        args[1].clone()
    } else {
        "xorIA/input.txt".to_string()
    };

    let model_path = "transformer_chat";
    let model_file = format!("{}.mpk", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let model_exists = Path::new(&model_file).exists();

    let target_vocab_size = 16000;
    let tokenizer = if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer BPE desde {}...", tokenizer_file);
        Tokenizer::load(&tokenizer_file)?
    } else {
        println!("Leyendo primeros 50MB para entrenar tokenizer...");
        let mut frag_iter = FileFragmentIterator::new(Path::new(&text_file), 50)?;
        let text = frag_iter.next().unwrap_or_default();
        println!("Entrenando tokenizer BPE (vocab_size={})...", target_vocab_size);
        let tok = Tokenizer::from_text(&text, target_vocab_size)?;
        tok.save(&tokenizer_file)?;
        tok
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size (BPE): {}", vocab_size);

    let mut temperature = 0.8;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.95;
    let mut repetition_penalty: f32 = 1.1;

    // Parâmetros ajustables (expuestos en el menú 's')
    let mut d_model: usize = 720;
    let mut num_layers: usize = 24;
    let mut num_heads: usize = 8;
    let mut lr: f64 = 3e-4;
    let mut num_epochs: usize = 50;
    let mut batch_size: usize = 16;
    let mut seq_len: usize = 128;
    let mut gradient_accumulation_steps: usize = 1;

    let mut modo_inferencia = false;
    if model_exists {
        loop {
            println!("\n--- CONFIGURACIÓN ACTUAL ---");
            println!("  (1) d_model: {}", d_model);
            println!("  (2) Num layers: {}", num_layers);
            println!("  (3) Heads:   {}", num_heads);
            println!("  (4) LR:      {}", lr);
            println!("  (5) Épocas:  {}", num_epochs);
            println!("  (6) Batch:   {}", batch_size);
            println!("  (7) Temp:    {}", temperature);
            println!("  (8) R-Pen:   {}", repetition_penalty);
            println!("  (9) seq_len: {} | stride: {}", seq_len, seq_len);
            println!("  (10) Grad Accum: {}x (eff batch: {})", gradient_accumulation_steps, batch_size * gradient_accumulation_steps);
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

                print!("d_model [{}]: ", d_model);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { d_model = v; }

                print!("Num layers [{}]: ", num_layers);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { num_layers = v; }

                print!("Heads [{}]: ", num_heads);
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

                print!("Repetition Penalty [{}]: ", repetition_penalty);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { repetition_penalty = v; }
                print!("Seq len [{}]: ", seq_len); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { seq_len = v; } }
                print!("Gradient accumulation steps [{}]: ", gradient_accumulation_steps); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { gradient_accumulation_steps = v; } }
            }
        }
    } else {
        loop {
            println!("\n--- NUEVO MODELO — CONFIGURACIÓN ---");
            println!("  (1) d_model: {}", d_model);
            println!("  (2) Num layers: {}", num_layers);
            println!("  (3) Heads:   {}", num_heads);
            println!("  (4) LR:      {}", lr);
            println!("  (5) Épocas:  {}", num_epochs);
            println!("  (6) Batch:   {}", batch_size);
            println!("  (7) seq_len: {} | stride: {}", seq_len, seq_len);
            println!("  (8) Grad Accum: {}x (eff batch: {})", gradient_accumulation_steps, batch_size * gradient_accumulation_steps);
            println!("------------------------------------");
            print!("¿Entrenar (e) o Ajustar parámetros (s)? [e/s]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();
            if choice == "e" { break; }
            else if choice == "s" {
                println!("\nAjustar parámetros (Enter para mantener actual):");
                print!("d_model [{}]: ", d_model); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { d_model = v; }
                print!("Num layers [{}]: ", num_layers); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_layers = v; }
                print!("Heads [{}]: ", num_heads); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_heads = v; }
                print!("Learning Rate [{}]: ", lr); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { lr = v; }
                print!("Épocas [{}]: ", num_epochs); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_epochs = v; }
                print!("Batch Size [{}]: ", batch_size); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { batch_size = v; }
                print!("Seq len [{}]: ", seq_len); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { seq_len = v; } }
                print!("Gradient accumulation steps [{}]: ", gradient_accumulation_steps); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { gradient_accumulation_steps = v; } }
            }
        }
    }

    let device = Default::default();

    let num_kv_groups = 4; 

    println!("\n── Configuración del Transformer ──");
    println!("  d_model:       {}", d_model);
    println!("  num_layers:    {}", num_layers);
    println!("  num_heads:     {} (query)", num_heads);
    println!("  num_kv_groups: {} (key/value)", num_kv_groups);
    println!("  heads/group:   {}", num_heads / num_kv_groups);
    println!("  head_dim:      {}", d_model / num_heads);
    println!("  FFN:           SwiGLU");
    println!("  Positional:    RoPE");
    println!("  KV Cache:      Enabled\n");

    let transformer_config = TransformerConfig {
        num_layers,
        layer: TransformerLayerConfig {
            d_model,
            num_heads,
            num_kv_groups,
            head_dim: None, 
            ffn_expansion: 4.0,
            use_swiglu: true,
            max_seq_len: 256,
            rope_base: 10000.0,
            rope_scaling: 1.0,
            causal: true,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
            residual_dropout: 0.1,
            attn_logit_cap: None,
            bias: false,
            norm_eps: 1e-5,
            ffn_round_to: 64,
        },
    };

    let mut model: TransformerLM<MyBackend> = TransformerLM {
        embedding: EmbeddingConfig::new(vocab_size, d_model).init(&device),
        transformer: transformer_config.init(&device),
        head: LinearConfig::new(d_model, vocab_size).with_bias(false).init(&device),
        vocab_size,
        d_model,
        num_layers,
    };

    let num_params = model.num_params();
    println!("Total parameters: {} ({:.2} M)\n", num_params, num_params as f64 / 1e6);

    if model_exists {
        println!("Cargando pesos del modelo...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
    } else {
        println!("No se encontró modelo previo. Iniciando desde cero.");
    }

    if modo_inferencia {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║     MODO INTERACTIVO — Transformer Chat                        ║");
        println!("║     KV Cache + Top-K/P + Repetition Penalty                    ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Comandos:");
        println!("  - Escribe tu semilla para generar texto.");
        println!("  - 'len <n>':    Cambia la cantidad de tokens.");
        println!("  - 'temp <f>':   Cambia la temperatura.");
        println!("  - 'topk <n>':   Cambia el Top-K.");
        println!("  - 'topp <f>':   Cambia el Top-P.");
        println!("  - 'rpen <f>':   Cambia el Repetition Penalty.");
        println!("  - 'salir' o 'exit' para terminar.\n");

        let mut current_len = 50;
        let mut session_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
        let mut session_offset = 0;

        loop {
            print!("Chat [len:{} t:{} k:{} p:{} rp:{}] > ", 
                current_len, temperature, top_k, top_p, repetition_penalty);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
                break;
            }

            if input.to_lowercase().starts_with("len ") {
                if let Ok(v) = input[4..].trim().parse::<usize>() {
                    current_len = v;
                    println!("  -> Longitud: {} tokens.\n", current_len);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("temp ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    temperature = v;
                    println!("  -> Temperatura: {}.\n", temperature);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("topk ") {
                if let Ok(v) = input[5..].trim().parse::<usize>() {
                    top_k = v;
                    println!("  -> Top-K: {}.\n", top_k);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("topp ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    top_p = v;
                    println!("  -> Top-P: {}.\n", top_p);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("rpen ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    repetition_penalty = v;
                    println!("  -> Repetition Penalty: {}.\n", repetition_penalty);
                    continue;
                }
            }
            if input.eq_ignore_ascii_case("reset") {
                session_caches = (0..num_layers).map(|_| None).collect();
                session_offset = 0;
                println!("  -> Memoria de sesión reiniciada.\n");
                continue;
            }

            if input.is_empty() { continue; }

            println!("\n--- TEXTO GENERADO (KV Cache Persistente) ---");
            let (text, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                &model.valid(), &tokenizer, input, current_len, &device,
                temperature, top_k, top_p, repetition_penalty,
                session_caches, session_offset,
            );
            session_caches = updated_caches.into_iter().map(Some).collect();
            session_offset = updated_offset;

            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("---");
            println!("Tokens: {} | Tiempo: {:.2}s | Velocidad: {:.2} tok/s | Offset Total: {}\n",
                tokens_count, elapsed, tps, session_offset);
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let stride = seq_len;

    let text_path = Path::new(&text_file);

    println!("Iniciando entrenamiento con streaming...");
    println!("  batch_size: {} | seq_len: {} | stride: {} | grad_accum: {}x\n", batch_size, seq_len, stride, gradient_accumulation_steps);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let start_epoch = Instant::now();

        let fragments = FileFragmentIterator::new(text_path, 1)?;

        for (frag_idx, fragment) in fragments.enumerate() {
            let tokens = tokenizer.encode(&fragment);
            let tokens_per_batch = batch_size * seq_len;
            let num_batches = tokens.len() / tokens_per_batch;
            if num_batches == 0 { continue; }

            for b in (0..num_batches).step_by(gradient_accumulation_steps) {
                let mut accum_loss = Tensor::<MyBackend, 1>::zeros([1], &device);
                let mut micro_steps = 0usize;

                for m in 0..gradient_accumulation_steps {
                    let micro_idx = b + m;
                    if micro_idx >= num_batches { break; }
                    let start_idx = micro_idx * tokens_per_batch;

                    let (x, y) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_len, stride, &device);

                    let logits = model.forward(x);
                    let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
                    let targets_flat = y.reshape([batch_size * seq_len]);

                    let loss = loss_fn.forward(logits_flat, targets_flat);
                    let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

                    if current_loss.is_nan() {
                        println!("\n[!] Loss NaN en Fragmento {} micro-batch {} (macro {}). Abortando.", frag_idx, m, micro_idx);
                        return Ok(());
                    }

                    total_loss += current_loss;
                    accum_loss = accum_loss + loss;
                    micro_steps += 1;
                }

                if micro_steps == 0 { continue; }

                accum_loss = accum_loss / micro_steps as f32;
                let grads = accum_loss.backward();
                let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads_p);
                batch_count += 1;

                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = (batch_count * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s            ",
                    epoch + 1, frag_idx, batch_count, (num_batches + gradient_accumulation_steps - 1) / gradient_accumulation_steps,
                    total_loss / (batch_count as f32 * gradient_accumulation_steps as f32), tps);
                io::stdout().flush().unwrap();
            }
        }

        let avg_loss = total_loss / batch_count.max(1) as f32 / gradient_accumulation_steps.max(1) as f32;
        println!("\nEpoch {} completa en {:.2}s. Loss: {:.4}",
            epoch + 1, start_epoch.elapsed().as_secs_f32(), avg_loss);

        let recorder = CompactRecorder::new();
        model.clone().save_file(model_path, &recorder)?;

        if (epoch + 1) % 5 == 0 {
            let ckpt = format!("{}_epoch_{}", model_path, epoch + 1);
            model.clone().save_file(&ckpt, &recorder)?;
            println!("  -> Checkpoint: {}.mpk", ckpt);
        }

        if (epoch + 1) % 2 == 0 {
            println!("--- Generación de prueba (KV Cache Persistente) ---");
            let empty_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
            let (_, tokens_count, elapsed, _, _) = generate_text_cached(
                &model.clone().valid(), &tokenizer, "The world ", 30, &device,
                temperature, top_k, top_p, repetition_penalty,
                empty_caches, 0,
            );
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("[{:.1} tok/s]\n---------------------------", tps);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn main() {
    if let Err(e) = transformer_chat() {
        eprintln!("Error: {}", e);
    }
}
