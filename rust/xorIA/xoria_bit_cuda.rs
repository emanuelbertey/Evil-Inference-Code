// ─── xoria_bit_cuda — BitLinear GPU Training ──────────────────────────────
//
// Entrenamiento GPU con CUDA. Modelo guardado como .mpk compatible con I2S CPU.
// Para inferencia I2S: usar xoria_bit (CPU).
//
// Usage: cargo run --bin xoria_bit_cuda --release -- xorIA/input.txt

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{Module, AutodiffModule},
    optim::{AdamConfig, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData, Int},
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
use xlstm::blocks::bitlinear::kernel::KernelKind;
use xlstm::blocks::trasformer_bit::model::{
    Tokenizer, FileFragmentIterator, BitLinearQKVProjection, BitLinearOutputProjection,
    BitLinearSwiGLUFeedForward, BitLinearTransformerLayer, BitLinearRMSNorm,
    BitLinearTransformerStack, TransformerBitLinearLM, TransformerInferenceState, KVCache,
    create_batch, sample_from_logits,
};
use xlstm::blocks::trasformer_bit::bitnet_export::{export_bitnet, load_bitnet, is_bitnet_file, compare_models};

type MyBackend = Autodiff<burn_cuda::Cuda<f32>>;

fn generate_text_cached<B: burn::tensor::backend::Backend>(
    model: &TransformerBitLinearLM<B>,
    inf_state: &TransformerInferenceState,
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
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]), &device);
    let (logits, updated_caches) = model.forward_with_cache_inference(input, current_offset, caches, inf_state);
    let mut caches = updated_caches.into_iter().map(Some).collect::<Vec<_>>();

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim]).reshape([1, v_dim]);
    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    if current_offset >= 200 {
        if current_offset >= 200 {
            for c in caches.iter_mut() { if let Some(ref mut kv) = c { kv.keep_last(70); } }
            current_offset = 70;
        }
    }

    let mut next_id = sample_from_logits(last_logits, temperature, top_k, top_p, repetition_penalty, &history);
    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) { if token == "eos" { break; } }
        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }
        let clean_str = tokenizer.id_to_token(next_id).unwrap_or_default().replace('\u{2581}', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_id as i64], [1, 1]), &device);
        let cache_input: Vec<Option<KVCache<B>>> = caches.into_iter().collect();
        let (logits, new_caches) = model.forward_with_cache_inference(input, current_offset, cache_input, inf_state);
        caches = new_caches.into_iter().map(Some).collect();
        current_offset += 1;

        if current_offset >= 200 {
            for c in caches.iter_mut() { if let Some(ref mut kv) = c { kv.keep_last(70); } }
            current_offset = 70;
        }

        let [_, _, v] = logits.dims();
        next_id = sample_from_logits(logits.reshape([1, v]), temperature, top_k, top_p, repetition_penalty, &history);
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

pub fn xoria_cuda() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Bit2 CUDA — BitLinear GPU Training             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 { args[1].clone() } else { "xorIA/input.txt".to_string() };

    let model_path = "transformer_bit2";
    let model_file = format!("{}.mpk", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let bitnet_file = format!("{}.bitnet", model_path);
    let model_exists = Path::new(&model_file).exists();
    let bitnet_exists = Path::new(&bitnet_file).exists();

    // ─── CLI Modes ─────────────────────────────────────────────────────
    if args.iter().any(|a| a == "--export") {
        if !model_exists { println!("No hay modelo .mpk para exportar."); return Ok(()); }
        let device = Default::default();
        let d_model: usize = 512;
        let num_layers: usize = 6;
        let num_heads: usize = 8;
        let num_kv_groups: usize = 4;
        let head_dim = d_model / num_heads;
        let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;
        let vocab_size = 16000;
        type FlexModel = TransformerBitLinearLM<Flex<f32>>;
        let layers: Vec<BitLinearTransformerLayer<Flex<f32>>> = (0..num_layers).map(|_| {
            BitLinearTransformerLayer {
                attn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
                qkv: BitLinearQKVProjection {
                    q_proj: BitLinearConfig { in_features: d_model, out_features: num_heads * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                    k_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                    v_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                    num_heads, num_kv_groups, head_dim,
                },
                o_proj: BitLinearOutputProjection { o_proj: BitLinearConfig { in_features: num_heads * head_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device), num_heads, head_dim },
                ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
                ffn: BitLinearSwiGLUFeedForward {
                    gate_up_proj: BitLinearConfig { in_features: d_model, out_features: 2 * ffn_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                    down_proj: BitLinearConfig { in_features: ffn_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                    dropout: burn::nn::DropoutConfig::new(0.0).init(), intermediate_dim: ffn_dim,
                },
                residual_dropout: burn::nn::DropoutConfig::new(0.0).init(),
            }
        }).collect();
        let mut model: FlexModel = TransformerBitLinearLM {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(&device),
            transformer: BitLinearTransformerStack { final_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device), num_layers, d_model, layers },
            head: BitLinearConfig { in_features: d_model, out_features: vocab_size, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
            vocab_size, d_model, num_layers,
        };
        println!("Cargando modelo MPK para exportar...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
        export_bitnet(&model, &bitnet_file)?;
        return Ok(());
    }

    if args.iter().any(|a| a == "--compare") {
        if !model_exists || !bitnet_exists {
            println!("Necesitás ambos archivos: {} y {}", model_file, bitnet_file);
            return Ok(());
        }
        let device = Default::default();
        compare_models::<Flex<f32>>(&model_file, &bitnet_file, &device)?;
        return Ok(());
    }

    let target_vocab_size = 16000;
    let tokenizer = if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer...");
        Tokenizer::load(&tokenizer_file)?
    } else {
        println!("Leyendo primeros 50MB para entrenar tokenizer...");
        let mut frag_iter = FileFragmentIterator::new(Path::new(&text_file), 50)?;
        let text = frag_iter.next().unwrap_or_default();
        let tok = Tokenizer::from_text(&text, target_vocab_size)?;
        tok.save(&tokenizer_file)?;
        tok
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {}", vocab_size);

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
            println!("  (7) Temp: {}  (8) Top-K: {}  (9) Top-P: {}  (10) R-Pen: {}", temperature, top_k, top_p, repetition_penalty);
            println!("----------------------------");
            print!("¿Entrenar (e), Inferir con I2S (i) o Ajustar (s)? [e/i/s]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            match choice.trim().to_lowercase().as_str() {
                "i" => { modo_inferencia = true; break; }
                "e" => break,
                "s" => {
                    macro_rules! rp { ($l:expr, $v:expr) => { print!("{} [{}]: ", $l, $v); io::stdout().flush().unwrap(); let mut b = String::new(); io::stdin().read_line(&mut b).unwrap(); if let Ok(v) = b.trim().parse() { $v = v; } }; }
                    rp!("d_model", d_model); rp!("Layers", num_layers); rp!("Heads", num_heads);
                    rp!("LR", lr); rp!("Épocas", num_epochs); rp!("Batch", batch_size);
                    rp!("Temp", temperature); rp!("Top-K", top_k); rp!("Top-P", top_p); rp!("R-Pen", repetition_penalty);
                }
                _ => continue,
            }
        }
    }

    let device = Default::default();
    let num_kv_groups = 4;
    let head_dim = d_model / num_heads;
    let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;

    println!("\n── Configuración (CUDA) ──");
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
            o_proj: BitLinearOutputProjection { o_proj: BitLinearConfig { in_features: num_heads * head_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device), num_heads, head_dim },
            ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: BitLinearConfig { in_features: d_model, out_features: 2 * ffn_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                down_proj: BitLinearConfig { in_features: ffn_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 }.init(&device),
                dropout: burn::nn::DropoutConfig::new(0.1).init(), intermediate_dim: ffn_dim,
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

    if modo_inferencia && bitnet_exists && is_bitnet_file(&bitnet_file) {
        println!("Cargando modelo BitNet directamente...");
        let (loaded_model, warnings) = load_bitnet::<MyBackend>(&bitnet_file, &device)?;
        model = loaded_model;
        for w in &warnings { println!("  {}", w); }
    } else if model_exists {
        println!("Cargando modelo MPK...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
    } else {
        println!("No se encontró modelo previo.");
    }

    if modo_inferencia {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║     MODO INFERENCIA — I2S Kernel (Ternary CPU)                ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Pre-computando kernels ternarios...");
        let inf_start = Instant::now();
        let mut model_v = model.valid();
        let inf_state = model_v.build_inference_state(&device, KernelKind::I2S, KernelKind::I2S);
        model_v.release_all_weights(&device);
        println!("Kernels listos en {:.2}s (RAM 16-bit liberada)\n", inf_start.elapsed().as_secs_f32());
        println!("Comandos: 'len <n>', 'temp <f>', 'topk <n>', 'topp <f>', 'rpen <f>', 'reset', 'salir'\n");

        let mut current_len = 50;
        let mut session_caches: Vec<Option<KVCache<burn_cuda::Cuda<f32>>>> = (0..num_layers).map(|_| None).collect();
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
            let (_, tokens_count, elapsed, caches, offset) = generate_text_cached(&model_v, &inf_state, &tokenizer, input, current_len, temperature, top_k, top_p, repetition_penalty, session_caches, session_offset);
            session_caches = caches; session_offset = offset;
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("---");
            println!("Tokens: {} | Tiempo: {:.2}s | {:.2} tok/s | Offset: {}\n", tokens_count, elapsed, tps, session_offset);
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let seq_len = 64;
    let stride = 64;

    println!("Entrenando en GPU...");
    println!("  batch_size: {} | seq_len: {} | stride: {} | epochs: {}\n", batch_size, seq_len, stride, num_epochs);
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let start_epoch = Instant::now();
        let fragments = FileFragmentIterator::new(Path::new(&text_file), 1)?;

        for (frag_idx, fragment) in fragments.enumerate() {
            let tokens = tokenizer.encode(&fragment);
            let tpb = batch_size * seq_len;
            let nb = tokens.len() / tpb;
            if nb == 0 { continue; }

            for b in 0..nb {
                let (x, y) = create_batch::<MyBackend>(&tokens, b * tpb, batch_size, seq_len, stride, &device);
                let logits = model.forward(x);
                let loss = loss_fn.forward(logits.reshape([batch_size * seq_len, vocab_size]), y.reshape([batch_size * seq_len]));
                let cl = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                if cl.is_nan() { println!("\n[!] NaN"); return Ok(()); }
                total_loss += cl; batch_count += 1;
                let grads = loss.backward();
                let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads_p);
                let tps = (batch_count * batch_size * seq_len) as f32 / start_epoch.elapsed().as_secs_f32();
                print!("\rEpoch {}/{} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s", epoch+1, num_epochs, frag_idx, b+1, nb, total_loss/batch_count as f32, tps);
                io::stdout().flush().unwrap();
            }
        }

        println!("\nEpoch {} Loss: {:.4} ({:.2}s)", epoch+1, total_loss/batch_count.max(1) as f32, start_epoch.elapsed().as_secs_f32());
        let recorder = CompactRecorder::new();
        model.clone().save_file(&model_file, &recorder)?;

        // Auto-export bitnet ternary format
        if let Err(e) = export_bitnet(&model, &bitnet_file) {
            println!("  ⚠ Error exportando .bitnet: {}", e);
        }

        if (epoch+1) % 2 == 0 {
            let inf_state = model.valid().build_inference_state(&device, KernelKind::I2S, KernelKind::I2S);
            let empty: Vec<Option<KVCache<burn_cuda::Cuda<f32>>>> = (0..num_layers).map(|_| None).collect();
            let (_, tc, el, _, _) = generate_text_cached(&model.clone().valid(), &inf_state, &tokenizer, "The world ", 30, temperature, top_k, top_p, repetition_penalty, empty, 0);
            println!("[{:.1} tok/s]", tc as f32 / el.max(0.001));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn main() {
    if let Err(e) = xoria_cuda() {
        eprintln!("Error: {}", e);
    }
}
