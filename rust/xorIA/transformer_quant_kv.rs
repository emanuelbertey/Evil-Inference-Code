// ─── Transformer Quant KV: CPU Transformer + TurboQuant KV Cache ─────────────
//
// Same architecture as transformer_chat (GQA + RoPE + SwiGLU) but with
// TurboQuant-compressed KV cache for inference (2-4 bits per element).
//
// Architecture:
//   Embedding → Transformer(N layers × GQA+RoPE+SwiGLU) → Linear → logits
//
// Features:
//   - Standard FP32 training (same as transformer_chat)
//   - TurboQuant KV cache for low-memory inference
//   - Top-K / Top-P sampling
//   - Repetition Penalty
//
// Usage:
//   cargo run --bin transformer_quant_kv --release -- xorIA/input.txt

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
use burn_flex::Flex;
use std::error::Error;
use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

use xlstm::blocks::trasformer::layer::{
    Transformer, TransformerConfig, TransformerLayerConfig,
};
use xlstm::blocks::trasformer::attention::KVCache;
use xlstm::blocks::trasformer_bit::cache::{KuantKVCache, MAX_CACHE_LEN};
use xlstm::blocks::trasformer_bit::ops::{apply_rope_fused, apply_rope_partial, apply_rope_fused_partial, repeat_kv, apply_causal_mask, apply_causal_mask_with_offset};
use xlstm::blocks::trasformer_bit::model::{Tokenizer, FileFragmentIterator, sample_from_logits, create_batch};

type MyBackend = Autodiff<Flex<f32>>;

// ─── Language Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerLM<B: Backend> {
    pub embedding: Embedding<B>,
    pub transformer: Transformer<B>,
    pub head: Linear<B>,
    /// x0 injection: learned scalar per layer, initialized to 0.
    /// Optional — old checkpoints without it default to None (no injection).
    pub x0_lambdas: Option<Param<Tensor<B, 2>>>,  // Some([1, num_layers])
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
}

impl<B: Backend> TransformerLM<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer.forward(x, 0);
        self.head.forward(x)
    }

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

    /// Forward pass with partial RoPE for training stability.
    /// Rotates only `rotary_pct` of head dimensions (0.0–1.0).
    /// Preserves autodiff for gradient flow.
    pub fn forward_train_partial_rope(
        &self,
        input: Tensor<B, 2, Int>,
        rotary_pct: f64,
    ) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let [batch, seq_len, _d] = x.dims();
        let device = x.device();
        let x0 = x.clone();  // for x0 injection

        let mut h = x;
        for (i, layer) in self.transformer.layers.iter().enumerate() {
            // Pre-Norm → Attention
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
            let h_attn = layer.residual_dropout.forward(h_attn);
            h = residual + h_attn;

            // Pre-Norm → FFN
            let residual = h.clone();
            let h_norm = layer.ffn_norm.forward(h);
            let h_ffn = layer.ffn.forward(h_norm);
            let h_ffn = layer.residual_dropout.forward(h_ffn);
            h = residual + h_ffn;

            // x0 injection: h += lambda_i * x0 (skip if None = old checkpoint)
            if let Some(ref lambdas) = self.x0_lambdas {
                let lam = lambdas.val().slice([0..1, i..(i+1)]).unsqueeze_dim::<3>(2);
                h = h + lam * x0.clone();
            }
        }

        let h = self.transformer.final_norm.forward(h);
        self.head.forward(h)
    }

    /// Forward with KV cache + partial RoPE (regular cache, not quant).
    pub fn forward_with_cache_partial(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<Option<KVCache<B>>>,
        rotary_pct: f64,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
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

    /// Forward pass with TurboQuant KV cache.
    /// Only supports batch=1. Uses partial RoPE when `rotary_pct < 1.0`.
    pub fn forward_with_kuant_cache(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<KuantKVCache>,
        rotary_pct: f64,
    ) -> (Tensor<B, 3>, Vec<KuantKVCache>) {
        let device = input.device();
        let mut h = self.embedding.forward(input);
        let [batch, _seq, _d] = h.dims();
        assert_eq!(batch, 1, "KuantKVCache requires batch=1");

        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (layer, mut cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            // Pre-Norm → Attention
            let residual = h.clone();
            let h_norm = layer.attn_norm.forward(h);

            let (q, k, v) = layer.attention.qkv.forward(h_norm);
            let (q, k_rot) = if rotary_pct < 0.999 {
                apply_rope_fused_partial(q, k, offset, rotary_pct)
            } else {
                apply_rope_fused(q, k, offset)
            };

            let old_len = cache.current_len;
            cache.append(k_rot, v);
            let attn_output = cache.attend(q, old_len, layer.attention.num_heads, &device);
            let h_attn = layer.attention.o_proj.forward(attn_output);
            h = residual + h_attn;

            // Pre-Norm → FFN
            let residual = h.clone();
            let h_norm = layer.ffn_norm.forward(h);
            let h_ffn = layer.ffn.forward(h_norm);
            h = residual + h_ffn;

            new_caches.push(cache);
        }
        let h = self.transformer.final_norm.forward(h);
        (self.head.forward(h), new_caches)
    }
}

// ─── Text Generation (TurboQuant) ───────────────────────────────────────────

fn generate_kuant_cached<B: Backend>(
    model: &TransformerLM<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    rotary_pct: f64,
    caches: Vec<KuantKVCache>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<KuantKVCache>, usize, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() {
        return (seed_text.to_string(), 0, 0.0, caches, current_offset, 0);
    }

    let device: B::Device = Default::default();
    let start_gen = Instant::now();
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]),
        &device,
    );

    let (logits, mut caches) =
        model.forward_with_kuant_cache(input, current_offset, caches, rotary_pct);

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits
        .slice([0..1, (s_len - 1)..s_len, 0..v_dim])
        .reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    let mut next_id =
        sample_from_logits(last_logits, temperature, top_k, top_p, repetition_penalty, &history);

    let mut total_model_time = 0.0f32;
    let mut total_other_time = 0.0f32;

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "<|endoftext|>" { break; }
        }

        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('\u{2581}', " ").replace('▁', " ").replace(' ', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let t0 = Instant::now();
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![next_id as i64], [1, 1]), &device);
        let (next_logits, new_caches) =
            model.forward_with_kuant_cache(input, current_offset, caches, rotary_pct);
        let model_time = t0.elapsed().as_secs_f32();
        total_model_time += model_time;

        let t1 = Instant::now();
        caches = new_caches;
        current_offset += 1;

        let [_, _, v] = next_logits.dims();
        let logits_2d = next_logits.reshape([1, v]);
        next_id = sample_from_logits(
            logits_2d, temperature, top_k, top_p, repetition_penalty, &history);
        total_other_time += t1.elapsed().as_secs_f32();
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    if !generated.is_empty() {
        let n = generated.len() as f32;
        println!(
            "[DEBUG] Modelo: {:.3}s ({:.1} ms/tok) | Other: {:.3}s ({:.1} ms/tok) | Kernel+attn: ~{:.1}%",
            total_model_time,
            total_model_time * 1000.0 / n,
            total_other_time,
            total_other_time * 1000.0 / n,
            total_model_time / elapsed.max(0.001) * 100.0
        );
    }
    let cache_bytes = caches.iter().map(KuantKVCache::compressed_bytes).sum();
    (text, generated.len(), elapsed, caches, current_offset, cache_bytes)
}

// ─── Text Generation (regular KV Cache) ─────────────────────────────────────

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
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]), device);

    let (logits, updated_caches) = if use_partial {
        model.forward_with_cache_partial(input, current_offset, caches, rotary_pct)
    } else {
        model.forward_with_cache(input, current_offset, caches)
    };
    let mut caches = updated_caches;
    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim]).reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    if current_offset >= 255 {
        if let Some(first) = caches.get(0) {
            let mut seq = first.cached_k.dims()[1];
            if seq > 70 {
                let remove = 160usize.min(seq);
                let keep = seq - remove;
                for c in caches.iter_mut() { *c = c.keep_last(keep); }
                current_offset = current_offset.saturating_sub(remove);
                seq = keep;
            }
        }
    }

    let mut next_id = sample_from_logits(
        last_logits, temperature, top_k, top_p, repetition_penalty, &history);

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
            TensorData::new(vec![next_id as i64], [1, 1]), device);
        let cache_input: Vec<Option<KVCache<B>>> = caches.into_iter().map(|c| Some(c)).collect();
        let (logits, new_caches) = if use_partial {
            model.forward_with_cache_partial(input, current_offset, cache_input, rotary_pct)
        } else {
            model.forward_with_cache(input, current_offset, cache_input)
        };
        caches = new_caches;
        current_offset += 1;

        if current_offset >= 255 {
            if let Some(first) = caches.get(0) {
                let seq = first.cached_k.dims()[1];
                if seq > 70 {
                    let remove = 160usize.min(seq);
                    let keep = seq - remove;
                    for c in caches.iter_mut() { *c = c.keep_last(keep); }
                    current_offset = current_offset.saturating_sub(remove);
                }
            }
        }

        let [_, _, v] = logits.dims();
        let logits_2d = logits.reshape([1, v]);
        next_id = sample_from_logits(
            logits_2d, temperature, top_k, top_p, repetition_penalty, &history);
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

// ─── Main ───────────────────────────────────────────────────────────────────

pub fn transformer_quant_kv() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Transformer Quant KV — CPU + TurboQuant KV Cache             ║");
    println!("║  BPE-Level Language Model (Hugging Face)                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 { args[1].clone() } else { "xorIA/input.txt".to_string() };

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

    let mut d_model: usize = 720;
    let mut num_layers: usize = 24;
    let mut num_heads: usize = 8;
    let mut lr: f64 = 3e-4;
    let mut num_epochs: usize = 50;
    let mut batch_size: usize = 16;
    let mut rotary_pct: f64 = 1.0;
    let mut use_x0: bool = true;
    let mut residual_dropout: f64 = 0.0;
    let mut use_burn_lr: bool = false;
    let mut use_partial_rope_infer: bool = false;

    let mut modo_inferencia = false;
    let mut modo_kuant = false;

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
            println!("  (9) RoPE%:   {}%", rotary_pct * 100.0);
            println!("  (10) x0:     {}", if use_x0 { "Si" } else { "No" });
            println!("  (11) ResDrop: {}", residual_dropout);
            println!("  (12) LR Sched: {}", if use_burn_lr { "Burn" } else { "Manual" });
            println!("  (13) Inf RoPE%: {}", if use_partial_rope_infer { format!("{}%", rotary_pct * 100.0) } else { "100% (full)".to_string() });
            println!("----------------------------");
            print!("¿Entrenar (e), Inferir (i), Inferir TurboQuant (t) o Ajustar (s)? [e/i/t/s]: ");
            io::stdout().flush()?;

            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();

            if choice == "i" { modo_inferencia = true; break; }
            if choice == "t" { modo_inferencia = true; modo_kuant = true; break; }
            if choice == "e" { break; }
            if choice == "s" {
                println!("\nAjustar parámetros (Enter para mantener actual):");
                print!("d_model [{}]: ", d_model); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { d_model = v; }
                print!("Num layers [{}]: ", num_layers); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_layers = v; }
                print!("Heads [{}]: ", num_heads); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_heads = v; }
                print!("Learning Rate [{}]: ", lr); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { lr = v; }
                print!("Épocas [{}]: ", num_epochs); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { num_epochs = v; }
                print!("Batch Size [{}]: ", batch_size); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { batch_size = v; }
                print!("Temperatura [{}]: ", temperature); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { temperature = v; }
                print!("Repetition Penalty [{}]: ", repetition_penalty); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse() { repetition_penalty = v; }
                print!("RoPE % [{}]: ", rotary_pct * 100.0); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { rotary_pct = (v / 100.0).clamp(0.0, 1.0); }
                print!("x0 injection (s/n) [{}]: ", if use_x0 { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_x0 = true, "n" | "no" | "" => use_x0 = false, _ => {} }
                print!("Residual Dropout [{}]: ", residual_dropout); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { residual_dropout = v.clamp(0.0, 1.0); }
                print!("LR scheduler (m=Manual, b=Burn) [{}]: ", if use_burn_lr { "b" } else { "m" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "b" | "burn" => use_burn_lr = true, "m" | "manual" | "" => use_burn_lr = false, _ => {} }
                print!("Partial RoPE en inferencia (s/n) [{}]: ", if use_partial_rope_infer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_partial_rope_infer = true, "n" | "no" | "" => use_partial_rope_infer = false, _ => {} }
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
            println!("  (7) RoPE%:   {}%", rotary_pct * 100.0);
            println!("  (8) x0:     {}", if use_x0 { "Si" } else { "No" });
            println!("  (9) ResDrop: {}", residual_dropout);
            println!("  (10) LR Sched: {}", if use_burn_lr { "Burn" } else { "Manual" });
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
                print!("RoPE % [{}]: ", rotary_pct * 100.0); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { rotary_pct = (v / 100.0).clamp(0.0, 1.0); }
                print!("x0 injection (s/n) [{}]: ", if use_x0 { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_x0 = true, "n" | "no" | "" => use_x0 = false, _ => {} }
                print!("Residual Dropout [{}]: ", residual_dropout); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; if let Ok(v) = input.trim().parse::<f64>() { residual_dropout = v.clamp(0.0, 1.0); }
                print!("LR scheduler (m=Manual, b=Burn) [{}]: ", if use_burn_lr { "b" } else { "m" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "b" | "burn" => use_burn_lr = true, "m" | "manual" | "" => use_burn_lr = false, _ => {} }
                print!("Partial RoPE en inferencia (s/n) [{}]: ", if use_partial_rope_infer { "s" } else { "n" }); io::stdout().flush()?; let mut input = String::new(); io::stdin().read_line(&mut input)?; match input.trim().to_lowercase().as_str() { "s" | "si" | "y" | "yes" => use_partial_rope_infer = true, "n" | "no" | "" => use_partial_rope_infer = false, _ => {} }
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
    println!("  head_dim:      {}", d_model / num_heads);
    println!("  FFN:           SwiGLU");
    println!("  Positional:    RoPE ({:.0}%)", rotary_pct * 100.0);
    println!("  x0 injection:  {}", if use_x0 { "Si" } else { "No" });
    println!("  ResDrop:       {}", residual_dropout);
    println!("  LR scheduler:  {}", if use_burn_lr { "Burn (Composed)" } else { "Manual (warmup+cosine)" });
    println!("  Inf RoPE:      {}", if use_partial_rope_infer { format!("Partial ({:.0}%)", rotary_pct * 100.0) } else { "Full (100%)".to_string() });
    println!("  KV Cache:      {} \n", if modo_kuant { "TurboQuant" } else { "Regular" });

    let transformer_config = TransformerConfig {
        num_layers,
        layer: TransformerLayerConfig {
            d_model,
            num_heads,
            num_kv_groups,
            head_dim: None,
            ffn_expansion: 4.0,
            use_swiglu: true,
            max_seq_len: MAX_CACHE_LEN,
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
        println!("Cargando pesos del modelo desde {}...", model_file);
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
        let model_v = model.valid();

        if modo_kuant {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!("║  MODO TURBOQUANT — Transformer + KV cuantizado                ║");
            println!("╚════════════════════════════════════════════════════════════════╝\n");

            let kv_seed = 42u64;
            let mut kv_bits = loop {
                print!("TurboQuant bits (2/3/4) [3]: ");
                io::stdout().flush()?;
                let mut line = String::new();
                io::stdin().read_line(&mut line)?;
                let line = line.trim();
                if line.is_empty() { break 3usize; }
                match line.parse::<usize>() {
                    Ok(2 | 3 | 4) => break line.parse().unwrap(),
                    _ => println!("  Opciones: 2, 3 o 4"),
                }
            };

            let mut current_len = 50usize;
            let num_kv_groups_actual = if num_kv_groups == 0 { num_heads } else { num_kv_groups };
            let head_dim = d_model / num_heads;
            let mut session_caches: Vec<KuantKVCache> = (0..num_layers).map(|idx| {
                KuantKVCache::new(num_kv_groups_actual, head_dim, kv_bits, kv_seed.wrapping_add(idx as u64))
            }).collect();
            let mut session_offset = 0usize;

            println!("Comandos: 'len <n>', 'temp <f>', 'topk <n>', 'topp <f>', 'rpen <f>', 'quant <n>', 'reset', 'salir'\n");

            loop {
                print!("Chat [kuant:{}b len:{} t:{} k:{} p:{} rp:{}] > ",
                    kv_bits, current_len, temperature, top_k, top_p, repetition_penalty);
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();

                if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }
                if input.to_lowercase().starts_with("len ") {
                    if let Ok(v) = input[4..].trim().parse::<usize>() { current_len = v; println!("  -> Longitud: {}\n", current_len); continue; }
                }
                if input.to_lowercase().starts_with("temp ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { temperature = v; println!("  -> Temperatura: {}\n", temperature); continue; }
                }
                if input.to_lowercase().starts_with("topk ") {
                    if let Ok(v) = input[5..].trim().parse::<usize>() { top_k = v; continue; }
                }
                if input.to_lowercase().starts_with("topp ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { top_p = v; continue; }
                }
                if input.to_lowercase().starts_with("rpen ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { repetition_penalty = v; continue; }
                }
                if input.to_lowercase().starts_with("quant ") {
                    if let Ok(v) = input[6..].trim().parse::<usize>() {
                        match v { 2 | 3 | 4 => { kv_bits = v; session_caches = (0..num_layers).map(|idx| KuantKVCache::new(num_kv_groups_actual, head_dim, kv_bits, kv_seed.wrapping_add(idx as u64))).collect(); session_offset = 0; println!("  -> TurboQuant cambiado a {} bits, cache reiniciada.\n", kv_bits); }, _ => println!("  Opciones: 2, 3 o 4\n"), }
                        continue;
                    }
                }
                if input.eq_ignore_ascii_case("reset") {
                    session_caches = (0..num_layers).map(|idx| KuantKVCache::new(num_kv_groups_actual, head_dim, kv_bits, kv_seed.wrapping_add(idx as u64))).collect();
                    session_offset = 0;
                    println!("  -> Cache TurboQuant reiniciada.\n");
                    continue;
                }
                if input.is_empty() { continue; }

                println!("\n--- TEXTO GENERADO (TurboQuant KV) ---");
                let (_, tokens_count, elapsed, updated_caches, updated_offset, cache_bytes) =
                    generate_kuant_cached(
                        &model_v, &tokenizer, input, current_len,
                        temperature, top_k, top_p, repetition_penalty, rotary_pct,
                        session_caches, session_offset,
                    );
                session_caches = updated_caches;
                session_offset = updated_offset;
                let tps = tokens_count as f32 / elapsed.max(0.001);
                println!("---");
                println!("Tokens: {} | Tiempo: {:.2}s | {:.2} tok/s | Offset: {} | KV comprimida: {:.2} KB\n",
                    tokens_count, elapsed, tps, session_offset, cache_bytes as f32 / 1024.0);
            }
        } else {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!("║  MODO INTERACTIVO — KV Cache regular                          ║");
            println!("╚════════════════════════════════════════════════════════════════╝\n");
            println!("Comandos: 'len <n>', 'temp <f>', 'topk <n>', 'topp <f>', 'rpen <f>', 'salir'\n");

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

                if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }
                if input.to_lowercase().starts_with("len ") {
                    if let Ok(v) = input[4..].trim().parse::<usize>() { current_len = v; continue; }
                }
                if input.to_lowercase().starts_with("temp ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { temperature = v; continue; }
                }
                if input.to_lowercase().starts_with("topk ") {
                    if let Ok(v) = input[5..].trim().parse::<usize>() { top_k = v; continue; }
                }
                if input.to_lowercase().starts_with("topp ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { top_p = v; continue; }
                }
                if input.to_lowercase().starts_with("rpen ") {
                    if let Ok(v) = input[5..].trim().parse::<f32>() { repetition_penalty = v; continue; }
                }
                if input.is_empty() { continue; }

                println!("\n--- TEXTO GENERADO (KV Cache Regular) ---");
                let inf_rotary = if use_partial_rope_infer { rotary_pct } else { 1.0 };
                let (text, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                    &model_v, &tokenizer, input, current_len, &device,
                    temperature, top_k, top_p, repetition_penalty,
                    session_caches, session_offset, inf_rotary,
                );
                session_caches = updated_caches.into_iter().map(Some).collect();
                session_offset = updated_offset;
                let tps = tokens_count as f32 / elapsed.max(0.001);
                println!("---");
                println!("Tokens: {} | Tiempo: {:.2}s | {:.2} tok/s | Offset: {}\n",
                    tokens_count, elapsed, tps, session_offset);
            }
        }
        return Ok(());
    }

    // ─── Training ──────────────────────────────────────────────────────────

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let seq_len = 64;
    let stride = 64;

    println!("Iniciando entrenamiento con streaming...");
    println!("  batch_size: {} | seq_len: {} | stride: {}", batch_size, seq_len, stride);

    let warmup_steps = 500;
    let cosine_period = 10000;
    let mut step_count = 0;

    let mut burn_scheduler = if use_burn_lr {
        Some(ComposedLrSchedulerConfig::new()
            .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, warmup_steps))
            .cosine(CosineAnnealingLrSchedulerConfig::new(lr, cosine_period).with_min_lr(lr * 0.1))
            .init()
            .unwrap())
    } else {
        None
    };

    println!("  LR: {:.0e} | warmup {} steps + cosine decay to 10% over {} steps | scheduler: {}\n",
        lr, warmup_steps, cosine_period, if use_burn_lr { "Burn" } else { "Manual" });

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let start_epoch = Instant::now();

        let fragments = FileFragmentIterator::new(Path::new(&text_file), 1)?;

        for (frag_idx, fragment) in fragments.enumerate() {
            let tokens = tokenizer.encode(&fragment);
            let tokens_per_batch = batch_size * seq_len;
            let num_batches = tokens.len() / tokens_per_batch;
            if num_batches == 0 { continue; }

            for b in 0..num_batches {
                let start_idx = b * tokens_per_batch;
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
                    println!("\n[!] Loss NaN en Fragmento {} Batch {}. Abortando.", frag_idx, b);
                    return Ok(());
                }

                total_loss += current_loss;
                batch_count += 1;

                let grads = loss.backward();
                let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
                step_count += 1;
                let current_lr = if let Some(ref mut sched) = burn_scheduler {
                    sched.step()
                } else if step_count < warmup_steps {
                    lr * step_count as f64 / warmup_steps as f64
                } else {
                    let t = ((step_count - warmup_steps) as f64 / cosine_period as f64).min(1.0);
                    lr * (0.1 + 0.9 * (1.0 + (t * std::f64::consts::PI).cos()) / 2.0)
                };
                model = optim.step(current_lr, model, grads_p);

                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = (batch_count * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s",
                    epoch + 1, frag_idx, b + 1, num_batches,
                    total_loss / batch_count.max(1) as f32, tps);
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
    }

    Ok(())
}

#[allow(dead_code)]
fn main() {
    if let Err(e) = transformer_quant_kv() {
        eprintln!("Error: {}", e);
    }
}
