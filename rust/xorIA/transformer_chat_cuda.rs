#![recursion_limit = "256"]
// ─── Transformer Chat CUDA: BPE-Level Language Model ─────────────────────────
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
//   - CUDA backend (GPU acceleration via burn-cuda)
//
// Usage:
//   cargo run --bin transformer_chat_cuda --release -- xorIA/input.txt

use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::LrScheduler;
use burn::lr_scheduler::composed::{ComposedLrScheduler, ComposedLrSchedulerConfig};
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{Module, AutodiffModule, Param},
    optim::{AdamConfig, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::Backend, TensorData, Int},
    nn::loss::CrossEntropyLossConfig,
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
};
use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};
use xlstm::blocks::trasformer_bit::ops::{apply_rope_partial, repeat_kv, apply_causal_mask, apply_causal_mask_with_offset};
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

// Use CUDA backend with Autodiff
type MyBackend = Autodiff<Cuda<f32, i32>>;

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
        let temp_file = "temp_train_transformer_cuda.txt";
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
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));
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

// ─── Language Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerLM<B: Backend> {
    pub embedding: Embedding<B>,
    pub transformer: Transformer<B>,
    pub head: Linear<B>,
    pub x0_lambdas: Option<Param<Tensor<B, 2>>>,  // [1, num_layers]
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

    /// Forward with partial RoPE for training stability + x0 injection.
    pub fn forward_train_partial_rope(
        &self,
        input: Tensor<B, 2, Int>,
        rotary_pct: f64,
    ) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let [batch, seq_len, _d] = x.dims();
        let device = x.device();
        let x0 = x.clone();

        let mut h = x;
        for (i, layer) in self.transformer.layers.iter().enumerate() {
            let residual = h.clone();
            let h_norm = layer.attn_norm.forward(h);

            let (q, k, v) = layer.attention.qkv.forward(h_norm);
            let (q, k) = apply_rope_partial(q, k, 0, rotary_pct);

            let k = repeat_kv(k, layer.attention.num_heads, layer.attention.num_kv_groups);
            let v = repeat_kv(v, layer.attention.num_heads, layer.attention.num_kv_groups);

            let q = q.swap_dims(1, 2);
            let k = k.swap_dims(1, 2);
            let v = v.swap_dims(1, 2);

            let scale = (layer.attention.head_dim as f64).sqrt();
            let mut scores = q.matmul(k.transpose()) / scale;
            if seq_len > 1 {
                scores = apply_causal_mask(scores, seq_len);
            }
            let attn = softmax(scores, 3);
            let attn = layer.attention.dropout.forward(attn);
            let h_attn = attn.matmul(v);
            let h_attn = h_attn.swap_dims(1, 2);
            let h_attn = layer.attention.o_proj.forward(h_attn);
            h = residual + h_attn;

            let residual = h.clone();
            let h_norm = layer.ffn_norm.forward(h);
            let h_ffn = layer.ffn.forward(h_norm);
            h = residual + h_ffn;

            if let Some(ref lambdas) = self.x0_lambdas {
                let lam = lambdas.val().slice([0..1, i..(i+1)]).unsqueeze_dim::<3>(2);
                h = h + lam * x0.clone();
            }
        }

        let h = self.transformer.final_norm.forward(h);
        self.head.forward(h)
    }

    /// Forward with KV cache (for efficient autoregressive generation)
    ///
    /// Returns: (logits, updated_caches)
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

    /// Forward with KV cache + partial RoPE (optional inference toggle).
    /// Manually replicates the cached attention loop using `apply_rope_partial`.
    pub fn forward_with_cache_partial(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<Option<KVCache<B>>>,
        rotary_pct: f64,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let device = input.device();
        let x = self.embedding.forward(input);
        let mut h = x;
        let mut new_caches = Vec::with_capacity(self.num_layers);

        for (layer, cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            let residual = h.clone();
            let h_norm = layer.attn_norm.forward(h);

            let (q, k_new, v_new) = layer.attention.qkv.forward(h_norm);
            let (q, k_new) = apply_rope_partial(q, k_new, offset, rotary_pct);

            let (k_full, v_full) = if let Some(prev) = cache {
                let k_cat = Tensor::cat(vec![prev.cached_k, k_new.clone()], 1);
                let v_cat = Tensor::cat(vec![prev.cached_v, v_new.clone()], 1);
                (k_cat, v_cat)
            } else {
                (k_new.clone(), v_new.clone())
            };

            let new_cache = KVCache { cached_k: k_full.clone(), cached_v: v_full.clone() };

            let k_expanded = repeat_kv(k_full, layer.attention.num_heads, layer.attention.num_kv_groups);
            let v_expanded = repeat_kv(v_full, layer.attention.num_heads, layer.attention.num_kv_groups);

            let q = q.swap_dims(1, 2);
            let k = k_expanded.swap_dims(1, 2);
            let v = v_expanded.swap_dims(1, 2);

            let scale = (layer.attention.head_dim as f64).sqrt();
            let mut scores = q.matmul(k.transpose()) / scale;

            if let Some(cap) = layer.attention.attn_logit_cap {
                scores = scores.div_scalar(cap).tanh().mul_scalar(cap);
            }

            let [_, _, q_len, kv_len] = scores.dims();
            if layer.attention.causal && q_len > 1 {
                scores = apply_causal_mask_with_offset(scores, q_len, kv_len);
            }

            let attn_weights = softmax(scores, 3);
            let attn_weights = layer.attention.dropout.forward(attn_weights);
            let attn_output = attn_weights.matmul(v);
            let attn_output = attn_output.swap_dims(1, 2);
            let h_attn = layer.attention.o_proj.forward(attn_output);
            h = residual + h_attn;

            let residual = h.clone();
            let h_norm = layer.ffn_norm.forward(h);
            let h_ffn = layer.ffn.forward(h_norm);
            h = residual + h_ffn;

            new_caches.push(new_cache);
        }

        let h = self.transformer.final_norm.forward(h);
        (self.head.forward(h), new_caches)
    }
}

// ─── Batch Creation (for training) ──────────────────────────────────────────

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

// ─── Advanced Sampling ──────────────────────────────────────────────────────

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

    // Apply repetition penalty
    if repetition_penalty != 1.0 {
        for (id, prob) in probs_vec.iter_mut() {
            if previous_tokens.contains(id) {
                *prob /= repetition_penalty;
            }
        }
    }

    // Sort by probability (descending)
    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-K + Top-P filtering
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

    // Greedy if temperature is ~0
    if temperature <= 1e-6 {
        return indices[0];
    }

    // Apply temperature to log-probs
    for p in weights.iter_mut() {
        *p = (p.max(1e-10).ln() / temperature).exp();
    }

    // Weighted random sampling
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

// ─── Text Generation with KV Cache ─────────────────────────────────────────

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
    rotary_pct: f64,
) -> (String, usize, f32, Vec<KVCache<B>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }

    let start_gen = Instant::now();
    let use_partial = rotary_pct < 0.999;

    // ── PHASE 1: Prefill ─ Process new seed sequence ──
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]),
        device,
    );

    let (logits, updated_caches) = if use_partial {
        model.forward_with_cache_partial(input, current_offset, caches, rotary_pct)
    } else {
        model.forward_with_cache(input, current_offset, caches)
    };
    let mut caches = updated_caches;

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim])
        .reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;
    // Trim rule: if cache length > threshold, remove `remove_count` oldest tokens
    if current_offset >= 70 {
        let threshold = 70usize;
        let remove_count = 30usize; // remove 30 oldest when threshold exceeded
        if let Some(first) = caches.get(0) {
            let mut dims = first.cached_k.dims();
            let mut seq = dims[1];
            if seq > threshold {
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

    // ── PHASE 2: Autoregressive generation ──
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
        let (logits, new_caches) = if use_partial {
            model.forward_with_cache_partial(input, current_offset, cache_input, rotary_pct)
        } else {
            model.forward_with_cache(input, current_offset, cache_input)
        };
        caches = new_caches;
        current_offset += 1;

        // Trim rule during generation: if cache length > threshold, remove `remove_count` oldest tokens
        if current_offset >= 70 {
            let threshold = 70usize;
            let remove_count = 30usize; // remove 30 oldest when threshold exceeded
            if let Some(first) = caches.get(0) {
                let mut dims = first.cached_k.dims();
                let mut seq = dims[1];
                if seq > threshold {
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

pub fn transformer_chat_cuda() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Chat CUDA — GQA + RoPE + SwiGLU             ║");
    println!("║     BPE-Level Language Model (Hugging Face) [CUDA]          ║");
    println!("║     + KV Cache + Top-K/P + Repetition Penalty               ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 {
        args[1].clone()
    } else {
        "xorIA/input.txt".to_string()
    };

    let model_path = "transformer_chat_cuda";
    let model_file = format!("{}.mpk", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let model_exists = Path::new(&model_file).exists();

    let mut use_custom_tokenizer: bool = false;
    let mut custom_tokenizer_path: String = "tokenizer.json".to_string();

    // ── Sampling configuration ──
    let mut temperature = 0.8;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.95;
    let mut repetition_penalty: f32 = 1.1;

    // Parámetros ajustables
    let mut d_model: usize = 720;
    let mut num_layers: usize = 24;
    let mut num_heads: usize = 8;
    let mut lr: f64 = 4e-5;
    let mut num_epochs: usize = 50;
    let mut batch_size: usize = 24;
    let mut seq_len: usize = 128;
    let mut rotary_pct: f64 = 1.0;
    let mut use_x0: bool = true;
    let mut residual_dropout: f64 = 0.0;
    let mut use_burn_lr: bool = false;
    let mut use_partial_rope_infer: bool = false;
    let mut gradient_accumulation_steps: usize = 1;

    let mut modo_inferencia = false;
    if model_exists {
        loop {
            println!("\n--- CONFIGURACIÓN ACTUAL (CUDA) ---");
            println!("  (1) d_model: {}", d_model);
            println!("  (2) Num layers: {}", num_layers);
            println!("  (3) Heads:   {}", num_heads);
            println!("  (4) LR:      {}", lr);
            println!("  (5) Épocas:  {}", num_epochs);
            println!("  (6) Batch:   {}", batch_size);
            println!("  (7) Temp:    {}", temperature);
            println!("  (8) R-Pen:   {}", repetition_penalty);
            println!("  (9) RoPE%:   {}%", rotary_pct * 100.0);
            println!("  (10) x0:     {}", if use_x0 { "Si" } else { "No" });
            println!("  (11) ResDrop: {}", residual_dropout);
            println!("  (12) LR Sched: {}", if use_burn_lr { "Burn" } else { "Manual" });
            println!("  (13) Inf RoPE%: {}", if use_partial_rope_infer { format!("{}%", rotary_pct * 100.0) } else { "100% (full)".to_string() });
            println!("  (14) seq_len: {} | stride: {}", seq_len, seq_len);
            println!("  (15) Grad Accum: {}x (eff batch: {})", gradient_accumulation_steps, batch_size * gradient_accumulation_steps);
            println!("  (16) Tokenizer: {}", if use_custom_tokenizer { format!("Custom ({})", custom_tokenizer_path) } else { "BPE (entrenado)".to_string() });
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
                print!("RoPE % [{}]: ", rotary_pct * 100.0); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { rotary_pct = (v / 100.0).clamp(0.0, 1.0); }
                print!("x0 injection (s/n) [{}]: ", if use_x0 { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_x0 = true, "n" | "no" | "" => use_x0 = false, _ => {} }
                print!("Residual Dropout [{}]: ", residual_dropout); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { residual_dropout = v.clamp(0.0, 1.0); }
                print!("LR scheduler (m=Manual, b=Burn) [{}]: ", if use_burn_lr { "b" } else { "m" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "b" | "burn" => use_burn_lr = true, "m" | "manual" | "" => use_burn_lr = false, _ => {} }
                print!("Partial RoPE en inferencia (s/n) [{}]: ", if use_partial_rope_infer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_partial_rope_infer = true, "n" | "no" | "" => use_partial_rope_infer = false, _ => {} }
                print!("Seq len [{}]: ", seq_len); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { seq_len = v; } }
                print!("Gradient accumulation steps [{}]: ", gradient_accumulation_steps); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { gradient_accumulation_steps = v; } }
                print!("Tokenizer custom (s/n) [{}]: ", if use_custom_tokenizer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_custom_tokenizer = true, "n" | "no" | "" => use_custom_tokenizer = false, _ => {} }
                if use_custom_tokenizer { print!("Ruta tokenizer.json [{}]: ", custom_tokenizer_path); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if !input.trim().is_empty() { custom_tokenizer_path = input.trim().to_string(); } }
            }
        }
    } else {
        loop {
            println!("\n--- NUEVO MODELO — CONFIGURACIÓN (CUDA) ---");
            println!("  (1) d_model: {}", d_model);
            println!("  (2) Num layers: {}", num_layers);
            println!("  (3) Heads:   {}", num_heads);
            println!("  (4) LR:      {}", lr);
            println!("  (5) Épocas:  {}", num_epochs);
            println!("  (6) Batch:   {}", batch_size);
            println!("  (7) RoPE%:   {}%", rotary_pct * 100.0);
            println!("  (8) x0:     {}", if use_x0 { "Si" } else { "No" });
            println!("  (9) ResDrop: {}", residual_dropout);
            println!("  (10) LR Sched: {}", if use_burn_lr { "Burn" } else { "Manual" });
            println!("  (11) seq_len: {} | stride: {}", seq_len, seq_len);
            println!("  (12) Grad Accum: {}x (eff batch: {})", gradient_accumulation_steps, batch_size * gradient_accumulation_steps);
            println!("  (13) Tokenizer: {}", if use_custom_tokenizer { format!("Custom ({})", custom_tokenizer_path) } else { "BPE (entrenado)".to_string() });
            println!("--------------------------------------------");
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
                print!("RoPE % [{}]: ", rotary_pct * 100.0); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { rotary_pct = (v / 100.0).clamp(0.0, 1.0); }
                print!("x0 injection (s/n) [{}]: ", if use_x0 { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_x0 = true, "n" | "no" | "" => use_x0 = false, _ => {} }
                print!("Residual Dropout [{}]: ", residual_dropout); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { residual_dropout = v.clamp(0.0, 1.0); }
                print!("LR scheduler (m=Manual, b=Burn) [{}]: ", if use_burn_lr { "b" } else { "m" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "b" | "burn" => use_burn_lr = true, "m" | "manual" | "" => use_burn_lr = false, _ => {} }
                print!("Partial RoPE en inferencia (s/n) [{}]: ", if use_partial_rope_infer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_partial_rope_infer = true, "n" | "no" | "" => use_partial_rope_infer = false, _ => {} }
                print!("Seq len [{}]: ", seq_len); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { seq_len = v; } }
                print!("Gradient accumulation steps [{}]: ", gradient_accumulation_steps); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<usize>() { if v > 0 { gradient_accumulation_steps = v; } }
                print!("Tokenizer custom (s/n) [{}]: ", if use_custom_tokenizer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_custom_tokenizer = true, "n" | "no" | "" => use_custom_tokenizer = false, _ => {} }
                if use_custom_tokenizer { print!("Ruta tokenizer.json [{}]: ", custom_tokenizer_path); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if !input.trim().is_empty() { custom_tokenizer_path = input.trim().to_string(); } }
            }
        }
    }

    let text = fs::read_to_string(&text_file)?;

    let target_vocab_size = 16000;
    let tokenizer = if use_custom_tokenizer {
        println!("Cargando tokenizer custom desde {}...", custom_tokenizer_path);
        let config_path = {
            let p = std::path::Path::new(&custom_tokenizer_path);
            let parent = p.parent().unwrap_or(std::path::Path::new("."));
            parent.join("tokenizer_config.json")
        };
        if config_path.exists() {
            if let Ok(cfg_str) = fs::read_to_string(&config_path) {
                if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) {
                    println!("  tokenizer_class: {}", cfg.get("tokenizer_class").and_then(|v| v.as_str()).unwrap_or("unknown"));
                    println!("  eos_token: {:?}", cfg.get("eos_token"));
                    println!("  model_max_length: {:?}", cfg.get("model_max_length"));
                }
            }
        } else {
            println!("  (tokenizer_config.json no encontrado en {:?})", config_path);
        }
        Tokenizer::load_pretrained(&custom_tokenizer_path)?
    } else if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer BPE desde {}...", tokenizer_file);
        Tokenizer::load(&tokenizer_file)?
    } else {
        println!("Entrenando tokenizer BPE (vocab_size={})...", target_vocab_size);
        let tok = Tokenizer::from_text(&text, target_vocab_size)?;
        tok.save(&tokenizer_file)?;
        tok
    };

    let vocab_size = tokenizer.vocab_size();
    let tok_type = if use_custom_tokenizer { "Custom" } else { "BPE" };
    println!("Vocab size ({}): {}", tok_type, vocab_size);

    let tokens = tokenizer.encode(&text);
    let device = CudaDevice::default();

    let num_kv_groups = 4;

    println!("\n── Configuración del Transformer (CUDA) ──");
    println!("  d_model:       {}", d_model);
    println!("  num_layers:    {}", num_layers);
    println!("  num_heads:     {} (query)", num_heads);
    println!("  num_kv_groups: {} (key/value)", num_kv_groups);
    println!("  heads/group:   {}", num_heads / num_kv_groups);
    println!("  head_dim:      {}", d_model / num_heads);
    println!("  FFN:           SwiGLU");
    println!("  Positional:    RoPE ({:.0}%)", rotary_pct * 100.0);
    println!("  x0 injection:  {}", if use_x0 { "Si" } else { "No" });
    println!("  ResDrop:       {}", residual_dropout);
    println!("  LR scheduler:  {}", if use_burn_lr { "Burn (Composed)" } else { "Manual (warmup+cosine)" });
    println!("  Backend:       CUDA");
    println!("  KV Cache:      Enabled");
    println!("  Inf RoPE:      {}", if use_partial_rope_infer { format!("Partial ({:.0}%)", rotary_pct * 100.0) } else { "Full (100%)".to_string() });
    println!("  Sampling:      Top-K={}, Top-P={}, Temp={}, RepPen={}\n",
        top_k, top_p, temperature, repetition_penalty);

    let transformer_config = TransformerConfig {
        num_layers,
        layer: TransformerLayerConfig {
            d_model,
            num_heads,
            num_kv_groups,
            head_dim: None, 
            ffn_expansion: 4.0,
            // KV-cache sizing note:
            // - For this model: num_layers=8, num_heads=8, d_model=512 -> head_dim=64
            // - K+V per token per layer = 2 * num_heads * head_dim floats
            // - bytes_per_token (float32) = 8 * 2 * 8 * 64 * 4 = 32,768 bytes (~32 KiB)
            // - tokens_max = VRAM_bytes / 32,768  (batch=1, KV-cache only)
            // Example: max_seq_len = 1024 (configured below) -> KV-cache ≈ 1024 * 32,768 ≈ 32 MiB
            // If using float16, these numbers are roughly halved. Overheads (weights/activations) reduce usable VRAM.
            use_swiglu: true,
            max_seq_len: 1024,
            rope_base: 10000.0,
            rope_scaling: 1.0,
            causal: true,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
            residual_dropout,
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
        x0_lambdas: Some(Param::from_tensor(Tensor::zeros([1, num_layers], &device))),
        vocab_size,
        d_model,
        num_layers,
    };

    let num_params = model.num_params();
    println!("Total parameters: {} ({:.2} M)\n", num_params, num_params as f64 / 1e6);

    if model_exists {
        println!("Cargando pesos del modelo CUDA...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
        if use_x0 && model.x0_lambdas.is_none() {
            model.x0_lambdas = Some(Param::from_tensor(Tensor::zeros([1, num_layers], &device)));
            println!("  -> x0_lambdas inicializado (checkpoint anterior sin este campo)");
        }
    } else {
        println!("No se encontró modelo previo. Iniciando desde cero.");
    }

    if modo_inferencia {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║     MODO INTERACTIVO — Transformer Chat CUDA                 ║");
        println!("║     KV Cache + Top-K/P + Repetition Penalty                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Comandos:");
        println!("  - Escribe tu semilla para generar texto.");
        println!("  - 'len <n>':    Cambia la cantidad de tokens.");
        println!("  - 'temp <f>':   Cambia la temperatura (ej: temp 0.7).");
        println!("  - 'topk <n>':   Cambia el Top-K (ej: topk 20).");
        println!("  - 'topp <f>':   Cambia el Top-P (ej: topp 0.9).");
        println!("  - 'rpen <f>':   Cambia el Repetition Penalty (ej: rpen 1.2).");
        println!("  - 'salir' o 'exit' para terminar.\n");

        let mut current_len = 50;
        let mut session_caches: Vec<Option<KVCache<Cuda<f32, i32>>>> = (0..num_layers).map(|_| None).collect();
        let mut session_offset = 0;

        loop {
            print!("CUDA [len:{} t:{} k:{} p:{} rp:{}] > ",
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

            println!("\n--- TEXTO GENERADO (CUDA + KV Cache Persistente) ---");
            let inf_rotary = if use_partial_rope_infer { rotary_pct } else { 1.0 };
            let (text, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                &model.valid(), &tokenizer, input, current_len, &device,
                temperature, top_k, top_p, repetition_penalty,
                session_caches, session_offset, inf_rotary,
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

    // ── Training Mode ──
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let stride = seq_len;
    let num_batches = (tokens.len().saturating_sub(seq_len) / stride).div_ceil(batch_size);

    let total_steps = num_batches * num_epochs;
    let warmup_steps = 50.min(total_steps / 10);
    let mut step_count = 0;

    let mut burn_scheduler = if use_burn_lr {
        Some(ComposedLrSchedulerConfig::new()
            .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, warmup_steps))
            .cosine(CosineAnnealingLrSchedulerConfig::new(lr, total_steps - warmup_steps).with_min_lr(lr * 0.2))
            .init()
            .unwrap())
    } else {
        None
    };

    println!("Iniciando entrenamiento BPE (CUDA)...");
    println!("  batch_size: {} | seq_len: {} | stride: {} | batches/epoch: {}", batch_size, seq_len, stride, num_batches);
    println!("  Grad accum: {}x → eff batch: {}", gradient_accumulation_steps, batch_size * gradient_accumulation_steps);
    println!("  LR: {:.0e} | warmup {} steps + cosine decay to 20% over {} steps | scheduler: {}\n",
        lr, warmup_steps, total_steps, if use_burn_lr { "Burn" } else { "Manual" });

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let mut step_count_epoch = 0;
        let start_epoch = Instant::now();

        for b in (0..num_batches).step_by(gradient_accumulation_steps) {
            let mut accum_loss = Tensor::<MyBackend, 1>::zeros([1], &device);
            let mut micro_steps = 0usize;

            for m in 0..gradient_accumulation_steps {
                let micro_idx = b + m;
                if micro_idx >= num_batches { break; }
                let start_idx = micro_idx * batch_size * stride;
                if start_idx + batch_size * stride + seq_len >= tokens.len() { break; }

                let (x, y) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_len, stride, &device);

                let logits = if rotary_pct < 0.999 || use_x0 {
                    model.forward_train_partial_rope(x, rotary_pct)
                } else {
                    model.forward(x)
                };
                let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
                let targets_flat = y.reshape([batch_size * seq_len]);

                let loss = loss_fn.forward(logits_flat, targets_flat);
                let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

                if current_loss.is_nan() {
                    println!("\n[!] Loss NaN en micro-batch {} (macro {}). Abortando.", m, micro_idx);
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
            step_count += 1;
            step_count_epoch += 1;
            let current_lr = if let Some(ref mut sched) = burn_scheduler {
                sched.step()
            } else if step_count < warmup_steps {
                lr * step_count as f64 / warmup_steps as f64
            } else if step_count < total_steps {
                let t = (step_count - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                lr * (0.2 + 0.8 * (1.0 + (t * std::f64::consts::PI).cos()) / 2.0)
            } else {
                lr * 0.1
            };
            model = optim.step(current_lr, model, grads_p);
            batch_count += 1;

            if step_count_epoch % 10 == 0 {
                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = (batch_count * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {}/{} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s            ",
                    epoch + 1, num_epochs, step_count_epoch, num_batches / gradient_accumulation_steps,
                    total_loss / (batch_count * gradient_accumulation_steps) as f32, tps);
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
            println!("--- Generación de prueba (CUDA + KV Cache Persistente) ---");
            let empty_caches: Vec<Option<KVCache<Cuda<f32, i32>>>> = (0..num_layers).map(|_| None).collect();
            let (_, tokens_count, elapsed, _, _) = generate_text_cached(
                &model.clone().valid(), &tokenizer, "The world ", 30, &device,
                temperature, top_k, top_p, repetition_penalty,
                empty_caches, 0, 1.0,
            );
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("[{:.1} tok/s]\n---------------------------", tps);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn main() {
    if let Err(e) = transformer_chat_cuda() {
        eprintln!("Error: {}", e);
    }
}
