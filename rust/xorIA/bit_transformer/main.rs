mod model;

use burn::prelude::*;
use burn::optim::{AdamConfig, Optimizer};
use burn::module::{Module, AutodiffModule};
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{Tensor, Int, TensorData};
use burn::nn::loss::CrossEntropyLossConfig;
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

use model::{BitTransformerLM, BitTransformer, BitTransformerLayer, BitAttention, BitFFN, BitLinearConfig, BitLinearMode, RMSNorm};

type MyBackend = burn_autodiff::Autodiff<burn_flex::Flex<f32>>;

// ─── Tokenizer ──────────────────────────────────────────────────────────────

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let model = BPE::builder().byte_fallback(true).build().map_err(|e| format!("Error building BPE: {}", e))?;
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', PrependScheme::Always, false)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('\u{2581}', PrependScheme::Always, false)));
        let special_token = "eos";
        tokenizer.add_special_tokens(&[AddedToken::from(special_token, true)]);
        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![AddedToken::from(special_token, true)])
            .build();
        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_bit_tok.txt";
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
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('\u{2581}', PrependScheme::Always, false)));
        Ok(Self { tokenizer })
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text, false).unwrap().get_ids().iter().map(|&id| id as usize).collect()
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

// ─── Model Factory ───────────────────────────────────────────────────────────

fn create_model<B: Backend>(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, device: &B::Device) -> BitTransformerLM<B> {
    let head_dim = d_model / num_heads;
    let layers = (0..num_layers).map(|_| {
        BitTransformerLayer {
            attention: BitAttention {
                q_proj: BitLinearConfig::new(d_model, d_model).init(device),
                k_proj: BitLinearConfig::new(d_model, d_model).init(device),
                v_proj: BitLinearConfig::new(d_model, d_model).init(device),
                o_proj: BitLinearConfig::new(d_model, d_model).init(device),
                num_heads,
                head_dim,
            },
            ffn: BitFFN {
                up: BitLinearConfig::new(d_model, d_model * 4).init(device),
                gate: BitLinearConfig::new(d_model, d_model * 4).init(device),
                down: BitLinearConfig::new(d_model * 4, d_model).init(device),
            },
            norm1: RMSNorm::new(d_model, 1e-5, device),
            norm2: RMSNorm::new(d_model, 1e-5, device),
        }
    }).collect();
    BitTransformerLM {
        embedding: burn::nn::EmbeddingConfig::new(vocab_size, d_model).init(device),
        transformer: BitTransformer { layers, norm_final: RMSNorm::new(d_model, 1e-5, device) },
        head: BitLinearConfig::new(d_model, vocab_size).init(device),
        mode: BitLinearMode::Training,
    }
}

// ─── Entropy Regularization ──────────────────────────────────────────────────

fn entropy_loss<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2> {
    let probs = burn::tensor::activation::softmax(logits.clone(), 1);
    let log_probs = probs.clone().clamp_min(1e-10).log();
    let entropy = probs * log_probs * -1.0;
    entropy.sum_dim(1)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     BitTransformer Chat — Ternary LLM (STE Training)         ║");
    println!("║     1.58-bit Weights | Per-Group Quantization | Entropy Reg  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_path = if args.len() >= 2 {
        args[1].clone()
    } else {
        "xorIA/input.txt".to_string()
    };

    let model_path = "bit_transformer_chat";
    let model_file = format!("{}.mpk", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let model_exists = Path::new(&model_file).exists();

    // ─── Load tokenizer ──────────────────────────────────────────────
    let start_tok = Instant::now();
    let tokenizer = if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer BPE desde {}...", tokenizer_file);
        Tokenizer::load(&tokenizer_file)?
    } else {
        println!("Leyendo dataset para entrenar tokenizer...");
        let text_full = std::fs::File::open(&text_path)
            .map(|f| {
                use std::io::Read;
                let mut reader = std::io::BufReader::new(f);
                let mut buf = vec![0u8; 50 * 1024 * 1024]; // 50MB max for tokenizer
                let n = reader.read(&mut buf).unwrap_or(0);
                buf.truncate(n);
                String::from_utf8(buf).unwrap_or_default()
            })
            .unwrap_or_default();
        println!("Entrenando tokenizer BPE (vocab_size=16000)...");
        let tok = Tokenizer::from_text(&text_full, 16000)?;
        tok.save(&tokenizer_file)?;
        tok
    };
    let tok_elapsed = start_tok.elapsed().as_secs_f32();
    println!("Tokenizer listo en {:.2}s", tok_elapsed);

    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size (BPE): {}", vocab_size);

    // ─── Tokenize dataset ────────────────────────────────────────────
    println!("Leyendo y tokenizando dataset...");
    let start_load = Instant::now();
    let text = fs::read_to_string(&text_path)?;
    let load_elapsed = start_load.elapsed().as_secs_f32();
    let file_size_mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Dataset: {:.2} MB leido en {:.2}s ({:.1} MB/s)", file_size_mb, load_elapsed, file_size_mb / load_elapsed.max(0.001) as f64);

    let start_tok2 = Instant::now();
    let tokens = tokenizer.encode(&text);
    let tok2_elapsed = start_tok2.elapsed().as_secs_f32();
    println!("Tokenizado: {} tokens en {:.2}s ({:.0} tok/s)", tokens.len(), tok2_elapsed, tokens.len() as f32 / tok2_elapsed.max(0.001));
    drop(text);

    // ─── Model config ────────────────────────────────────────────────
    let device = Default::default();
    let d_model = 512;
    let num_layers = 6;
    let num_heads = 8;
    let head_dim = d_model / num_heads;
    let ffn_dim = d_model * 4;
    let batch_size = 8;
    let seq_len = 128;
    let num_epochs = 5;
    let lr = 3e-4;
    let entropy_weight = 0.01;

    let tokens_per_batch = batch_size * seq_len;
    let num_batches = (tokens.len() - 1) / tokens_per_batch;

    println!("\n── Configuracion del BitTransformer ──");
    println!("  d_model:       {}", d_model);
    println!("  num_layers:    {}", num_layers);
    println!("  num_heads:     {} (query)", num_heads);
    println!("  head_dim:      {}", head_dim);
    println!("  ffn_dim:       {} (SwiGLU)", ffn_dim);
    println!("  Quantization:  1.58-bit Ternary (Per-Group, GS=128)");
    println!("  Training:      STE (Straight-Through Estimator)");
    println!("  Entropy Reg:   {}\n", entropy_weight);

    let mut model = create_model::<MyBackend>(vocab_size, d_model, num_layers, num_heads, &device);
    let param_count = (d_model * d_model * 4 + d_model * ffn_dim * 3) as f64 * num_layers as f64
        + (vocab_size * d_model) as f64 + (d_model * vocab_size) as f64;
    println!("Total parameters: {:.2} M", param_count / 1e6);

    if model_exists {
        println!("Cargando pesos del modelo...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
    } else {
        println!("No se encontro modelo previo. Iniciando desde cero.");
    }

    // ─── Interactive loop ────────────────────────────────────────────
    loop {
        model.print_info();
        println!("\nOptions: (t)rain, (i)nfer, (m)ode, (q)uit");
        print!("> ");
        io::stdout().flush()?;
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        match choice.trim() {
            "t" => {
                model.mode = BitLinearMode::Training;
                train(&mut model, &tokens, vocab_size, &device, entropy_weight, batch_size, seq_len, num_epochs, lr, num_batches)?;
                let recorder = CompactRecorder::new();
                model.clone().save_file(&model_file, &recorder)?;
                println!("Modelo guardado en {}", model_file);
            }
            "i" => {
                println!("Modo Inferencia: {:?}", model.mode);
                generate(&model.clone().valid(), &tokenizer, &device);
            }
            "m" => {
                println!("Modo actual: {:?}", model.mode);
                println!("Elegir: (1) Ternary, (2) Full16");
                let mut m_choice = String::new();
                io::stdin().read_line(&mut m_choice)?;
                match m_choice.trim() {
                    "1" => { model.mode = BitLinearMode::Ternary; println!("Modo: Ternary Inference."); }
                    "2" => { model.mode = BitLinearMode::Full16; println!("Modo: Full16 Inference."); }
                    _ => println!("Opcion invalida."),
                }
            }
            "q" => break,
            _ => continue,
        }
    }
    Ok(())
}

// ─── Training ───────────────────────────────────────────────────────────────

fn train<B: AutodiffBackend>(
    model: &mut BitTransformerLM<B>,
    tokens: &[usize],
    vocab_size: usize,
    device: &B::Device,
    entropy_weight: f32,
    batch_size: usize,
    seq_len: usize,
    num_epochs: usize,
    lr: f64,
    num_batches: usize,
) -> Result<(), Box<dyn Error>> {
    let mut optim = AdamConfig::new().init::<B, BitTransformerLM<B>>();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let tokens_per_batch = batch_size * seq_len;

    println!("\nIniciando entrenamiento BitTransformer...");
    println!("  batch_size: {} | seq_len: {} | batches/epoch: {} | epochs: {}\n", batch_size, seq_len, num_batches, num_epochs);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let start_epoch = Instant::now();

        for b in 0..num_batches {
            let start_idx = b * tokens_per_batch;
            if start_idx + tokens_per_batch + 1 >= tokens.len() { break; }

            let mut x_vec = Vec::with_capacity(batch_size * seq_len);
            let mut y_vec = Vec::with_capacity(batch_size * seq_len);
            for i in 0..batch_size {
                let s = start_idx + i * seq_len;
                for j in 0..seq_len {
                    x_vec.push(tokens[s + j] as i64);
                    y_vec.push(tokens[s + j + 1] as i64);
                }
            }

            let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_vec, [batch_size, seq_len]), device);
            let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_vec, [batch_size, seq_len]), device);

            let logits = model.forward(x);
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = y.reshape([batch_size * seq_len]);

            let ce_loss = loss_fn.forward(logits_flat.clone(), targets_flat);

            // Entropy regularization: penalize overconfident predictions
            let ent = entropy_loss(logits_flat);
            let ent_mean = ent.mean();
            let loss = ce_loss - ent_mean * entropy_weight;

            let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

            if current_loss.is_nan() {
                println!("\n[!] Loss NaN en Batch {}. Abortando.", b);
                return Ok(());
            }

            total_loss += current_loss;
            batch_count += 1;

            let grads = loss.backward();
            let grads_p = burn::optim::GradientsParams::from_grads(grads, &*model);
            *model = optim.step(lr, model.clone(), grads_p);

            if b % 10 == 0 {
                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = ((b + 1) * tokens_per_batch) as f32 / elapsed;
                print!("\rEpoch {}/{} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s",
                    epoch + 1, num_epochs, b, num_batches,
                    total_loss / batch_count as f32, tps);
                io::stdout().flush().unwrap();
            }
        }

        let avg_loss = total_loss / batch_count.max(1) as f32;
        let epoch_elapsed = start_epoch.elapsed().as_secs_f32();
        let tps = (batch_count * tokens_per_batch) as f32 / epoch_elapsed;
        println!("\nEpoch {} completa en {:.2}s. Loss: {:.4} | {:.1} tok/s",
            epoch + 1, epoch_elapsed, avg_loss, tps);
    }
    Ok(())
}

// ─── Generation ─────────────────────────────────────────────────────────────

fn generate<B: Backend>(model: &BitTransformerLM<B>, tokenizer: &Tokenizer, device: &B::Device) {
    print!("Enter seed: ");
    io::stdout().flush().unwrap();
    let mut seed = String::new();
    io::stdin().read_line(&mut seed).unwrap();
    let seed = seed.trim();
    if seed.is_empty() { return; }

    let mut ids = tokenizer.encode(seed);
    println!("\n--- TEXTO GENERADO ---");

    let start_gen = Instant::now();

    for _ in 0..100 {
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), [1, ids.len()]),
            device,
        );
        let logits = model.forward(input);
        let [_, s, v] = logits.dims();
        let last_logits = logits.slice([0..1, (s-1)..s, 0..v]).reshape([1, v]);

        let next_id = last_logits.argmax(1).into_scalar().elem::<i64>() as usize;
        ids.push(next_id);

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('\u{2581}', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        if tokenizer.decode(&[next_id]) == "eos" { break; }
        if ids.len() > 128 { ids.remove(0); }
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let gen_count = ids.len().saturating_sub(tokenizer.encode(seed).len());
    let tps = gen_count as f32 / elapsed.max(0.001);
    println!("\n---");
    println!("Tokens: {} | Tiempo: {:.2}s | Velocidad: {:.2} tok/s\n", gen_count, elapsed, tps);
}
