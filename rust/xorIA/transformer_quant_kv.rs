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
use std::time::Instant;

use xlstm::blocks::trasformer::layer::{
    Transformer, TransformerConfig, TransformerLayerConfig,
};
use xlstm::blocks::trasformer::attention::KVCache;
use xlstm::blocks::trasformer_bit::cache::{KuantKVCache, MAX_CACHE_LEN};
use xlstm::blocks::trasformer_bit::ops::apply_rope_fused;
use xlstm::blocks::trasformer_bit::model::{Tokenizer, FileFragmentIterator, sample_from_logits, create_batch};

type MyBackend = Autodiff<Flex<f32>>;

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

    /// Forward pass with TurboQuant KV cache.
    /// Only supports batch=1.
    pub fn forward_with_kuant_cache(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        mut caches: Vec<KuantKVCache>,
    ) -> (Tensor<B, 3>, Vec<KuantKVCache>) {
        let device = input.device();
        let x = self.embedding.forward(input);
        let [batch, _seq, _d] = x.dims();
        assert_eq!(batch, 1, "KuantKVCache requires batch=1");

        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (layer, cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            // Pre-Norm → Attention
            let residual = x.clone();
            let h = layer.attn_norm.forward(x);

            let (q, k, v) = layer.attention.qkv.forward(h);
            let (q, k_rot) = apply_rope_fused(q, k, offset);

            let old_len = cache.current_len;
            cache.append(k_rot, v);
            let attn_output = cache.attend(q, old_len, layer.attention.num_heads, &device);
            let h = layer.attention.o_proj.forward(attn_output);
            let h = layer.residual_dropout.forward(h);
            let x = residual + h;

            // Pre-Norm → FFN
            let residual = x.clone();
            let h = layer.ffn_norm.forward(x);
            let h = layer.ffn.forward(h);
            let h = layer.residual_dropout.forward(h);
            x = residual + h;

            new_caches.push(cache);
        }
        let x = self.transformer.final_norm.forward(x);
        (self.head.forward(x), new_caches)
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
        model.forward_with_kuant_cache(input, current_offset, caches);

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
            model.forward_with_kuant_cache(input, current_offset, caches);
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
) -> (String, usize, f32, Vec<KVCache<B>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }

    let start_gen = Instant::now();
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]), device);

    let (logits, updated_caches) = model.forward_with_cache(input, current_offset, caches);
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
        let (logits, new_caches) = model.forward_with_cache(input, current_offset, cache_input);
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
    println!("  Positional:    RoPE");
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
        println!("Cargando pesos del modelo desde {}...", model_file);
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
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
                        temperature, top_k, top_p, repetition_penalty,
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
                let (text, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                    &model_v, &tokenizer, input, current_len, &device,
                    temperature, top_k, top_p, repetition_penalty,
                    session_caches, session_offset,
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
    println!("  batch_size: {} | seq_len: {} | stride: {}\n", batch_size, seq_len, stride);

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

                let logits = model.forward(x);
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
                model = optim.step(lr, model, grads_p);

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
