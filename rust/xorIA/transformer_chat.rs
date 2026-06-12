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
use std::io::{self, Write};
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

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Chat — GQA + RoPE + SwiGLU                   ║");
    println!("║     BPE-Level Language Model (Hugging Face)                  ║");
    println!("║     + KV Cache + Top-K/P + Repetition Penalty               ║");
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

    let text = fs::read_to_string(&text_file)?;
    
    let target_vocab_size = 2000;
    let tokenizer = if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer BPE desde {}...", tokenizer_file);
        Tokenizer::load(&tokenizer_file)?
    } else {
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

    let mut modo_inferencia = false;
    if model_exists {
        loop {
            print!("¡Modelo Transformer encontrado! ¿Deseas (e)ntrenar o (i)nferir? [e/i]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();
            match choice.as_str() {
                "i" => { modo_inferencia = true; break; }
                "e" => { break; }
                _ => {
                    if choice.is_empty() { continue; }
                    println!("  → Escribe 'e' para entrenar o 'i' para inferencia.");
                }
            }
        }
    }

    let tokens = tokenizer.encode(&text);
    let device = Default::default();

    let d_model = 512;
    let num_layers = 8;
    let num_heads = 8;
    let num_kv_groups = 2; 

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
            max_seq_len: 1024,
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
        println!("║     MODO INTERACTIVO — Transformer Chat                      ║");
        println!("║     KV Cache + Top-K/P + Repetition Penalty                  ║");
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
    let batch_size = 16;
    let seq_len = 64;
    let stride = 64;
    let num_batches = (tokens.len().saturating_sub(seq_len) / stride).div_ceil(batch_size);
    let num_epochs = 50;

    println!("Iniciando entrenamiento BPE...");
    println!("  batch_size: {} | seq_len: {} | batches/epoch: {}\n", batch_size, seq_len, num_batches);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let start_epoch = Instant::now();

        for b in 0..num_batches {
            let start_idx = b * batch_size * stride;
            if start_idx + batch_size * stride + seq_len >= tokens.len() { break; }

            let (x, y) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_len, stride, &device);

            let logits = model.forward(x);
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = y.reshape([batch_size * seq_len]);

            let loss = loss_fn.forward(logits_flat, targets_flat);
            let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

            if current_loss.is_nan() {
                println!("\n[!] Loss NaN en Batch {}. Abortando.", b);
                return Ok(());
            }

            total_loss += current_loss;
            batch_count += 1;

            let grads = loss.backward();
            let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(3e-4, model, grads_p);

            if b % 10 == 0 {
                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = ((b + 1) * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {}/{} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s",
                    epoch + 1, num_epochs, b, num_batches,
                    total_loss / batch_count as f32, tps);
                io::stdout().flush().unwrap();
            }
        }

        let avg_loss = total_loss / batch_count.max(1) as f32;
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
