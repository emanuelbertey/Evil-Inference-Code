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
use tokenizers::AddedToken;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use model::{BitTransformerLM, BitTransformer, BitTransformerLayer, BitAttention, BitFFN, BitLinearConfig, BitLinearMode, RMSNorm};

type MyBackend = burn_autodiff::Autodiff<burn_flex::Flex<f32>>;

// ─── Professional Tokenizer (Borrowed from transformer_chat.rs) ─────────────

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let model = BPE::builder().byte_fallback(true).build().map_err(|e| format!("{}", e))?;
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, false)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, false)));
        let special_token = "<|endoftext|>";
        tokenizer.add_special_tokens(&[AddedToken::from(special_token, true)]);
        let trainer = BpeTrainerBuilder::default()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![AddedToken::from(special_token, true)])
            .build();
        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_bit_tok.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("{}", e))?;
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
    pub fn encode(&self, text: &str) -> Vec<usize> { self.tokenizer.encode(text, false).unwrap().get_ids().iter().map(|&id| id as usize).collect() }
    pub fn decode(&self, indices: &[usize]) -> String { 
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap() 
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

// ─── Main Logic ──────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   BitTransformer — 1.58-bit Ternary LLM Implementation         ║");
    println!("║   Training in 16-bit (STE) | Inference: Ternary or Full Choice ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let device = Default::default();
    let text_path = "xorIA/input.txt";
    let text = fs::read_to_string(text_path)?;
    
    let tokenizer_path = "xorIA/bit_transformer/tokenizer.json";
    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Training tokenizer...");
        let tok = Tokenizer::from_text(&text, 2000)?;
        tok.save(tokenizer_path)?;
        tok
    };

    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.tokenizer.get_vocab_size(true);
    let d_model = 256;
    let num_layers = 4;
    let num_heads = 8;

    let mut model = create_model::<MyBackend>(vocab_size, d_model, num_layers, num_heads, &device);
    let model_file = "xorIA/bit_transformer/model.mpk";

    if Path::new(model_file).exists() {
        println!("Loading existing model weights...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
    }

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
                train(&mut model, &tokens, vocab_size, &device)?;
                let recorder = CompactRecorder::new();
                model.clone().save_file(model_file, &recorder)?;
                println!("Model saved to {}", model_file);
            }
            "i" => {
                println!("Inference Mode: {:?}", model.mode);
                generate(&model.clone().valid(), &tokenizer, &device);
            }
            "m" => {
                println!("Current Mode: {:?}", model.mode);
                println!("Choose mode: (1) Ternary, (2) Full16");
                let mut m_choice = String::new();
                io::stdin().read_line(&mut m_choice)?;
                match m_choice.trim() {
                    "1" => { model.mode = BitLinearMode::Ternary; println!("Switched to Ternary Inference."); }
                    "2" => { model.mode = BitLinearMode::Full16; println!("Switched to Full16 Inference."); }
                    _ => println!("Invalid choice."),
                }
            }
            "q" => break,
            _ => continue,
        }
    }

    Ok(())
}

fn train<B: AutodiffBackend>(model: &mut BitTransformerLM<B>, tokens: &[usize], vocab_size: usize, device: &B::Device) -> Result<(), Box<dyn Error>> {
    let mut optim = AdamConfig::new().init::<B, BitTransformerLM<B>>();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let batch_size = 8;
    let seq_len = 64;
    let epochs = 5;

    println!("Starting training...");
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let num_batches = 100; // Small subset for demo
        for b in 0..num_batches {
            let start = b * batch_size * seq_len % (tokens.len() - seq_len - 1);
            let mut x_vec = Vec::new();
            let mut y_vec = Vec::new();
            for i in 0..batch_size {
                let s = start + i * seq_len;
                for j in 0..seq_len {
                    x_vec.push(tokens[s + j] as i64);
                    y_vec.push(tokens[s + j + 1] as i64);
                }
            }
            let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_vec, [batch_size, seq_len]), device);
            let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_vec, [batch_size, seq_len]), device);

            let logits = model.forward(x);
            let loss = loss_fn.forward(logits.reshape([batch_size * seq_len, vocab_size]), y.reshape([batch_size * seq_len]));
            
            total_loss += loss.clone().into_scalar().elem::<f32>();
            
            let grads = loss.backward();
            let grads_p = burn::optim::GradientsParams::from_grads(grads, &*model);
            *model = optim.step(3e-4, model.clone(), grads_p);

            if b % 20 == 0 {
                print!("\rEpoch {} | Batch {}/{} | Loss: {:.4}", epoch, b, num_batches, total_loss / (b + 1) as f32);
                io::stdout().flush()?;
            }
        }
        println!("\nEpoch {} Loss: {:.4}", epoch, total_loss / num_batches as f32);
    }
    Ok(())
}

fn generate<B: Backend>(model: &BitTransformerLM<B>, tokenizer: &Tokenizer, device: &B::Device) {
    print!("Enter seed: ");
    io::stdout().flush().unwrap();
    let mut seed = String::new();
    io::stdin().read_line(&mut seed).unwrap();
    let seed = seed.trim();
    if seed.is_empty() { return; }

    let mut ids = tokenizer.encode(seed);
    println!("\nGenerating...");
    for _ in 0..50 {
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), [1, ids.len()]), device);
        let logits = model.forward(input);
        let [_, s, v] = logits.dims();
        let last_logits = logits.slice([0..1, (s-1)..s, 0..v]).reshape([1, v]);
        
        // Greedy sampling for simplicity
        let next_id = last_logits.argmax(1).into_scalar().elem::<i64>() as usize;
        ids.push(next_id);
        
        let token = tokenizer.decode(&[next_id]);
        print!("{}", token.replace('▁', " "));
        io::stdout().flush().unwrap();
        if token == "<|endoftext|>" { break; }
        if ids.len() > 64 { ids.remove(0); }
    }
    println!("\n");
}
