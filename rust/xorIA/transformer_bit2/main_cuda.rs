// ─── Transformer Bit2 CUDA — BitLinear GPU Training ────────────────────────
//
// Entrenamiento GPU con CUDA. Modelo guardado como .mpk compatible con I2S CPU.
// Para inferencia I2S: usar transformer_bit2 (CPU).
//
// Usage: cargo run --bin transformer_bit2_cuda --release -- xorIA/input.txt

mod model;

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

type MyBackend = Autodiff<burn_cuda::Cuda<f32>>;

fn generate_text_cached(
    model: &TransformerBitLinearLM<MyBackend>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    caches: Vec<Option<KVCache<MyBackend>>>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<Option<KVCache<MyBackend>>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }
    let device = Default::default();
    let start_gen = Instant::now();
    let seed_len = ids.len();
    let input = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]), &device);
    let (logits, updated_caches) = model.forward_with_cache(input, current_offset, caches);
    let mut caches = updated_caches.into_iter().map(Some).collect::<Vec<_>>();

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim]).reshape([1, v_dim]);
    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    if current_offset >= 255 {
        if let Some(Some(first)) = caches.get(0) {
            let seq = first.cached_k.dims()[1];
            if seq > 70 { let keep = seq - 160.min(seq); for c in caches.iter_mut() { if let Some(ref kv) = c { *c = Some(kv.keep_last(keep)); } }; current_offset = current_offset.saturating_sub(160); }
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

        let input = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(vec![next_id as i64], [1, 1]), device);
        let cache_input: Vec<Option<KVCache<MyBackend>>> = caches.into_iter().collect();
        let (logits, new_caches) = model.forward_with_cache(input, current_offset, cache_input);
        caches = new_caches.into_iter().map(Some).collect();
        current_offset += 1;

        if current_offset >= 255 {
            if let Some(Some(first)) = caches.get(0) {
                let seq = first.cached_k.dims()[1];
                if seq > 70 { let keep = seq - 160.min(seq); for c in caches.iter_mut() { if let Some(ref kv) = c { *c = Some(kv.keep_last(keep)); } }; current_offset = current_offset.saturating_sub(160); }
            }
        }

        let [_, _, v] = logits.dims();
        next_id = sample_from_logits(logits.reshape([1, v]), temperature, top_k, top_p, repetition_penalty, &history);
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Bit2 CUDA — BitLinear GPU Training             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 { args[1].clone() } else { "xorIA/input.txt".to_string() };

    let model_path = "transformer_bit2";
    let model_file = format!("{}.mpk", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let model_exists = Path::new(&model_file).exists();

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
            println!("\n--- CONFIG ---");
            println!("  d_model={} | layers={} | heads={}", d_model, num_layers, num_heads);
            print!("¿Entrenar (e), Inferir (i) o Ajustar (s)? [e/i/s]: ");
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
                    rp!("Temp", temperature); rp!("R-Pen", repetition_penalty);
                }
                _ => continue,
            }
        }
    }

    let device = Default::default();
    let num_kv_groups = 4;
    let head_dim = d_model / num_heads;
    let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;

    println!("\n── Config (CUDA) ──");
    println!("  GPU Training | d_model={} | layers={} | heads={}\n", d_model, num_layers, num_heads);

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

    if model_exists {
        println!("Cargando modelo...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
    }

    if modo_inferencia {
        let mut session_caches: Vec<Option<KVCache<MyBackend>>> = (0..num_layers).map(|_| None).collect();
        let mut session_offset = 0;
        loop {
            print!("Chat > "); io::stdout().flush()?;
            let mut input = String::new(); io::stdin().read_line(&mut input)?;
            let input = input.trim();
            if input == "salir" || input == "exit" { break; }
            if input.is_empty() { continue; }
            let (_, _, _, caches, offset) = generate_text_cached(&model.valid(), &tokenizer, input, 50, temperature, top_k, top_p, repetition_penalty, session_caches, session_offset);
            session_caches = caches; session_offset = offset;
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let seq_len = 64;
    let stride = 64;

    println!("Entrenando en GPU...\n");
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
                model = optim.step(lr, model, burn::optim::GradientsParams::from_grads(grads, &model));
                let tps = (batch_count * batch_size * seq_len) as f32 / start_epoch.elapsed().as_secs_f32();
                print!("\rEpoch {} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s", epoch+1, frag_idx, b+1, nb, total_loss/batch_count as f32, tps);
                io::stdout().flush().unwrap();
            }
        }

        println!("\nEpoch {} Loss: {:.4} ({:.2}s)", epoch+1, total_loss/batch_count.max(1) as f32, start_epoch.elapsed().as_secs_f32());
        CompactRecorder::new().clone().save_file(&model_file, &model.clone())?;

        if (epoch+1) % 2 == 0 {
            let empty: Vec<Option<KVCache<MyBackend>>> = (0..num_layers).map(|_| None).collect();
            let (_, tc, el, _, _) = generate_text_cached(&model.clone().valid(), &tokenizer, "The world ", 30, temperature, top_k, top_p, repetition_penalty, empty, 0);
            println!("[{:.1} tok/s]", tc as f32 / el.max(0.001));
        }
    }
    Ok(())
}
