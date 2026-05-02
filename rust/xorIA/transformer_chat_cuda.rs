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
// Backend: CUDA (GPU acceleration via burn-cuda)
//
// Usage:
//   cargo run --bin transformer_chat_cuda --release -- xorIA/input.txt

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
use burn_cuda::{Cuda, CudaDevice};
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
}

// ─── Language Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerLM<B: Backend> {
    pub embedding: Embedding<B>,
    pub transformer: Transformer<B>,
    pub head: Linear<B>,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl<B: Backend> TransformerLM<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer.forward(x, 0);
        self.head.forward(x)
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

fn sample_from_logits<B: Backend>(logits: Tensor<B, 2>, temperature: f32) -> usize {
    let probs = softmax(logits / temperature, 1);
    let probs_vec: Vec<f32> = probs.into_data().as_slice::<f32>().unwrap().to_vec();

    let mut rng = rand::rng();
    use rand::Rng;
    let sample: f32 = rng.random::<f32>();
    let mut cumulative = 0.0;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative { return i; }
    }
    0
}

// ─── Text Generation ────────────────────────────────────────────────────────

fn generate_text<B: Backend>(
    model: &TransformerLM<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
) -> String {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return seed_text.to_string(); }

    let mut context: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    let max_ctx = 512; 

    let mut generated = Vec::new();

    println!("--- Generando {} tokens ---", length);
    let start_gen = Instant::now();

    for _ in 0..length {
        let ctx_start = if context.len() > max_ctx { context.len() - max_ctx } else { 0 };
        let ctx = &context[ctx_start..];
        let ctx_len = ctx.len();

        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ctx.to_vec(), [1, ctx_len]),
            device,
        );

        let logits = model.forward(input);
        let last_logits = logits.slice([0..1, (ctx_len - 1)..ctx_len, 0..tokenizer.vocab_size()])
            .reshape([1, tokenizer.vocab_size()]);

        let next_id = sample_from_logits(last_logits, 0.7);
        generated.push(next_id);
        context.push(next_id as i64);

        let token = tokenizer.decode(&[next_id]);
        print!("{}", token);
        io::stdout().flush().unwrap();
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let tps = length as f32 / elapsed;
    println!("\n\n[Velocidad: {:.2} tokens/s | Tiempo: {:.2}s]", tps, elapsed);
    tokenizer.decode(&generated)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Chat CUDA — GQA + RoPE + SwiGLU             ║");
    println!("║     BPE-Level Language Model (Hugging Face) [CUDA]          ║");
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

    let text = fs::read_to_string(&text_file)?;
    
    let target_vocab_size = 2000; // Configurable BPE vocab size
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

    let mut modo_inferencia = false;
    if model_exists {
        loop {
            print!("¡Modelo Transformer CUDA encontrado! ¿Deseas (e)ntrenar o (i)nferir? [e/i]: ");
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
    let device = CudaDevice::default();

    let d_model = 256;
    let num_layers = 4;
    let num_heads = 8;
    let num_kv_groups = 2; 

    println!("\n── Configuración del Transformer (CUDA) ──");
    println!("  d_model:       {}", d_model);
    println!("  num_layers:    {}", num_layers);
    println!("  num_heads:     {} (query)", num_heads);
    println!("  num_kv_groups: {} (key/value)", num_kv_groups);
    println!("  heads/group:   {}", num_heads / num_kv_groups);
    println!("  head_dim:      {}", d_model / num_heads);
    println!("  FFN:           SwiGLU");
    println!("  Positional:    RoPE");
    println!("  Backend:       CUDA\n");

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
    };

    let num_params = model.num_params();
    println!("Total parameters: {} ({:.2} M)\n", num_params, num_params as f64 / 1e6);

    if model_exists {
        println!("Cargando pesos del modelo CUDA...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
    } else {
        println!("No se encontró modelo previo. Iniciando desde cero.");
    }

    if modo_inferencia {
        println!("\n╔════════════════════════════════════════════════════════╗");
        println!("║     MODO INTERACTIVO — Transformer Chat CUDA         ║");
        println!("╚════════════════════════════════════════════════════════╝\n");
        println!("Comandos:");
        println!("  - Escribe tu semilla para generar texto.");
        println!("  - 'len <n>': Cambia la cantidad de tokens.");
        println!("  - 'salir' o 'exit' para terminar.\n");

        let mut current_len = 50; // Fewer tokens needed for BPE
        loop {
            print!("Semilla CUDA [len: {}] > ", current_len);
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
                    println!("  -> Longitud: {} tokens.\n", current_len);
                    continue;
                }
            }

            if input.is_empty() { continue; }

            println!("\n--- TEXTO GENERADO (CUDA) ---");
            generate_text(&model.valid(), &tokenizer, input, current_len, &device);
            println!("----------------------\n");
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let batch_size = 16;
    let seq_len = 64; // BPE handles more content per token
    let stride = 64;
    let num_batches = (tokens.len().saturating_sub(seq_len) / stride).div_ceil(batch_size);
    let num_epochs = 50;

    println!("Iniciando entrenamiento BPE (CUDA)...");
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
            println!("--- Generación de prueba (CUDA) ---");
            generate_text(&model.clone().valid(), &tokenizer, "The world ", 30, &device);
            println!("\n---------------------------");
        }
    }

    Ok(())
}
