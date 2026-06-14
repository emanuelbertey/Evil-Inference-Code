// ─── Transformer Bit2: BitLinear (1.58-bit) CPU Training + I2S Kernel Inference
//
// Entrenamiento CPU con STE + inferencia con kernel I2S (ternary).
// Para entrenamiento GPU: usar transformer_bit2_cuda.
//
// Usage: cargo run --bin transformer_bit2 --release -- xorIA/input.txt

mod model;

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{Module, AutodiffModule},
    optim::{AdamConfig, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData, Int, backend::Backend},
    nn::loss::CrossEntropyLossConfig,
    nn::EmbeddingConfig,
};
use burn_autodiff::Autodiff;
use burn_flex::Flex;
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use xlstm::blocks::bitlinear::layer::BitLinearConfig;
use model::{
    Tokenizer, FileFragmentIterator, BitLinearQKVProjection, BitLinearOutputProjection,
    BitLinearSwiGLUFeedForward, BitLinearTransformerLayer, BitLinearRMSNorm,
    BitLinearTransformerStack, TransformerBitLinearLM, KVCache,
    create_batch, sample_from_logits,
};

type MyBackend = Autodiff<Flex<f32>>;

// ─── Text Generation with I2S Kernel Inference ─────────────────────────────

fn generate_text_cached<B: Backend>(
    model: &TransformerBitLinearLM<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    caches: Vec<Option<KVCache<B>>>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<Option<KVCache<B>>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }

    let device: B::Device = Default::default();
    let start_gen = Instant::now();
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]), &device,
    );

    let (logits, updated_caches) = model.forward_with_cache_inference(input, current_offset, caches, &device);
    let mut caches = updated_caches.into_iter().map(Some).collect::<Vec<_>>();

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim]).reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    // Trim rule
    if current_offset >= 255 {
        if let Some(Some(first)) = caches.get(0) {
            let seq = first.cached_k.dims()[1];
            if seq > 70 {
                let keep = seq - 160.min(seq);
                for c in caches.iter_mut() { if let Some(ref kv) = c { *c = Some(kv.keep_last(keep)); } }
                current_offset = current_offset.saturating_sub(160);
            }
        }
    }

    let mut next_id = sample_from_logits(last_logits, temperature, top_k, top_p, repetition_penalty, &history);

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "eos" { break; }
        }

        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('\u{2581}', " ").replace(' ', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_id as i64], [1, 1]), &device);
        let cache_input: Vec<Option<KVCache<B>>> = caches.into_iter().collect();
        let (logits, new_caches) = model.forward_with_cache_inference(input, current_offset, cache_input, &device);
        caches = new_caches.into_iter().map(Some).collect();
        current_offset += 1;

        // Trim rule during generation
        if current_offset >= 255 {
            if let Some(Some(first)) = caches.get(0) {
                let seq = first.cached_k.dims()[1];
                if seq > 70 {
                    let keep = seq - 160.min(seq);
                    for c in caches.iter_mut() { if let Some(ref kv) = c { *c = Some(kv.keep_last(keep)); } }
                    current_offset = current_offset.saturating_sub(160);
                }
            }
        }

        let [_, _, v] = logits.dims();
        let logits_2d = logits.reshape([1, v]);
        next_id = sample_from_logits(logits_2d, temperature, top_k, top_p, repetition_penalty, &history);
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Bit2 — BitLinear CPU + I2S Kernel             ║");
    println!("║     GQA + RoPE + SwiGLU + KV Cache + Ternary Inference       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 { args[1].clone() } else { "xorIA/input.txt".to_string() };

    let model_path = "transformer_bit2";
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
    let mut d_model: usize = 512;
    let mut num_layers: usize = 6;
    let mut num_heads: usize = 8;
    let mut lr: f64 = 3e-4;
    let mut num_epochs: usize = 10;
    let mut batch_size: usize = 8;

    let mut modo_inferencia = false;
    if model_exists {
        loop {
            println!("\n--- CONFIGURACIÓN ACTUAL ---");
            println!("  (1) d_model: {}  (2) Layers: {}  (3) Heads: {}", d_model, num_layers, num_heads);
            println!("  (4) LR: {}  (5) Épocas: {}  (6) Batch: {}", lr, num_epochs, batch_size);
            println!("  (7) Temp: {}  (8) R-Pen: {}", temperature, repetition_penalty);
            println!("----------------------------");
            print!("¿Entrenar (e), Inferir con I2S (i) o Ajustar (s)? [e/i/s]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();
            if choice == "i" { modo_inferencia = true; break; }
            else if choice == "e" { break; }
            else if choice == "s" {
                macro_rules! read_param { ($label:expr, $val:expr) => { print!("{} [{}]: ", $label, $val); io::stdout().flush()?; let mut buf = String::new(); io::stdin().read_line(&mut buf)?; if let Ok(v) = buf.trim().parse() { $val = v; } }; }
                read_param!("d_model", d_model);
                read_param!("Layers", num_layers);
                read_param!("Heads", num_heads);
                read_param!("LR", lr);
                read_param!("Épocas", num_epochs);
                read_param!("Batch", batch_size);
                read_param!("Temp", temperature);
                read_param!("R-Pen", repetition_penalty);
            }
        }
    }

    let device = Default::default();
    let num_kv_groups = 4;
    let head_dim = d_model / num_heads;
    let ffn_expansion = 4.0;
    let ffn_dim = ((ffn_expansion * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;

    println!("\n── Configuración ──");
    println!("  d_model={} | layers={} | heads={} | kv_groups={}", d_model, num_layers, num_heads, num_kv_groups);
    println!("  head_dim={} | ffn_dim={} | SwiGLU | RoPE | I2S Kernel\n", head_dim, ffn_dim);

    let layers = (0..num_layers).map(|_| {
        BitLinearTransformerLayer {
            attn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
            qkv: BitLinearQKVProjection {
                q_proj: BitLinearConfig { in_features: d_model, out_features: num_heads * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                k_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                v_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                num_heads, num_kv_groups, head_dim,
            },
            o_proj: BitLinearOutputProjection {
                o_proj: BitLinearConfig { in_features: num_heads * head_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                num_heads, head_dim,
            },
            ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: BitLinearConfig { in_features: d_model, out_features: 2 * ffn_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                down_proj: BitLinearConfig { in_features: ffn_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                dropout: burn::nn::DropoutConfig::new(0.1).init(),
                intermediate_dim: ffn_dim,
            },
            residual_dropout: burn::nn::DropoutConfig::new(0.1).init(),
        }
    }).collect();

    let mut model: TransformerBitLinearLM<MyBackend> = TransformerBitLinearLM {
        embedding: EmbeddingConfig::new(vocab_size, d_model).init(&device),
        transformer: BitLinearTransformerStack { final_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device), num_layers, d_model, layers },
        head: BitLinearConfig { in_features: d_model, out_features: vocab_size, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
        vocab_size, d_model, num_layers,
    };

    let param_count = (d_model * d_model * 4 + d_model * ffn_dim * 3) as f64 * num_layers as f64;
    println!("Total parameters (approx): {:.2} M\n", param_count / 1e6);

    if model_exists {
        println!("Cargando pesos del modelo...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
    } else {
        println!("No se encontró modelo previo. Iniciando desde cero.");
    }

    if modo_inferencia {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║     MODO INFERENCIA — I2S Kernel (Ternary CPU)                ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Comandos: 'len <n>', 'temp <f>', 'topk <n>', 'topp <f>', 'rpen <f>', 'reset', 'salir'\n");

        let mut current_len = 50;
        let mut session_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
        let mut session_offset = 0;

        loop {
            print!("Chat [len:{} t:{} k:{} p:{} rp:{}] > ", current_len, temperature, top_k, top_p, repetition_penalty);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }
            if input.to_lowercase().starts_with("len ") { if let Ok(v) = input[4..].trim().parse::<usize>() { current_len = v; println!("  -> Longitud: {}\n", current_len); continue; } }
            if input.to_lowercase().starts_with("temp ") { if let Ok(v) = input[5..].trim().parse::<f32>() { temperature = v; println!("  -> Temperatura: {}\n", temperature); continue; } }
            if input.to_lowercase().starts_with("topk ") { if let Ok(v) = input[5..].trim().parse::<usize>() { top_k = v; println!("  -> Top-K: {}\n", top_k); continue; } }
            if input.to_lowercase().starts_with("topp ") { if let Ok(v) = input[5..].trim().parse::<f32>() { top_p = v; println!("  -> Top-P: {}\n", top_p); continue; } }
            if input.to_lowercase().starts_with("rpen ") { if let Ok(v) = input[5..].trim().parse::<f32>() { repetition_penalty = v; println!("  -> R-Pen: {}\n", repetition_penalty); continue; } }
            if input.eq_ignore_ascii_case("reset") { session_caches = (0..num_layers).map(|_| None).collect(); session_offset = 0; println!("  -> Cache reiniciada.\n"); continue; }
            if input.is_empty() { continue; }

            println!("\n--- TEXTO GENERADO (I2S Kernel) ---");
            let (_, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                &model.valid(), &tokenizer, input, current_len,
                temperature, top_k, top_p, repetition_penalty, session_caches, session_offset,
            );
            session_caches = updated_caches;
            session_offset = updated_offset;
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("---");
            println!("Tokens: {} | Tiempo: {:.2}s | {:.2} tok/s | Offset: {}\n", tokens_count, elapsed, tps, session_offset);
        }
        return Ok(());
    }

    // Training
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let seq_len = 64;
    let stride = 64;
    let text_path = Path::new(&text_file);

    println!("Iniciando entrenamiento CPU...");
    println!("  batch_size: {} | seq_len: {} | stride: {} | epochs: {}\n", batch_size, seq_len, stride, num_epochs);

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

            for b in 0..num_batches {
                let start_idx = b * tokens_per_batch;
                let (x, y) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_len, stride, &device);
                let logits = model.forward(x);
                let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
                let targets_flat = y.reshape([batch_size * seq_len]);
                let loss = loss_fn.forward(logits_flat, targets_flat);
                let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                if current_loss.is_nan() { println!("\n[!] Loss NaN. Abortando."); return Ok(()); }
                total_loss += current_loss;
                batch_count += 1;
                let grads = loss.backward();
                let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads_p);
                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = (batch_count * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {}/{} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s",
                    epoch + 1, num_epochs, frag_idx, b + 1, num_batches, total_loss / batch_count as f32, tps);
                io::stdout().flush().unwrap();
            }
        }

        let avg_loss = total_loss / batch_count.max(1) as f32;
        println!("\nEpoch {} completa en {:.2}s. Loss: {:.4}", epoch + 1, start_epoch.elapsed().as_secs_f32(), avg_loss);

        let recorder = CompactRecorder::new();
        model.clone().save_file(model_path, &recorder)?;

        if (epoch + 1) % 2 == 0 {
            println!("--- Generación de prueba (I2S Kernel) ---");
            let empty_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
            let (_, tokens_count, elapsed, _, _) = generate_text_cached(
                &model.valid(), &tokenizer, "The world ", 30,
                temperature, top_k, top_p, repetition_penalty, empty_caches, 0,
            );
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("[{:.1} tok/s]\n---------------------------", tps);
        }
    }

    Ok(())
}
