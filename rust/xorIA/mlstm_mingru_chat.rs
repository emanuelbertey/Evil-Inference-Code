#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using MLSTM and MINGRU alternating layers.
This version uses a custom vector for the block order.

Author: Based on xlstm-rs project
Date: February 2026
*/

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::AutodiffModule,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::{AutodiffBackend, Backend}},
    nn::loss::CrossEntropyLossConfig,
};
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use burn_flex::Flex;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::{LearningRateConfig, LstmType, XLstm, XLstmconfig, BlockType};
type MyBackend = Autodiff<Flex<f32>>;

/// Tokenizador profesional usando la librería 'tokenizers' de Hugging Face
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
        
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '▁', 
            PrependScheme::Always,
            true,
        )));

        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);

        let temp_file = "temp_train_mlstm_mingru.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true)
            .map_err(|e| format!("Error al guardar: {}", e))?;
        println!("Tokenizador guardado en: {}", path);
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| format!("Error al cargar: {}", e))?;
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));
        println!("Tokenizador cargado desde: {}", path);
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

fn create_batch<B: AutodiffBackend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    vocab_size: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2, burn::tensor::Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i * stride;
        for j in 0..seq_length {
            x_indices.push(tokens[current_start + j] as i64);
            y_indices.push(tokens[current_start + j + 1] as i64);
        }
    }

    let eye = Tensor::<B::InnerBackend, 2>::eye(vocab_size, device);
    let indices_inner = Tensor::<B::InnerBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(x_indices, [batch_size * seq_length]),
        device,
    );

    let x = Tensor::<B, 3>::from_inner(
        eye.select(0, indices_inner)
           .reshape([batch_size, seq_length, vocab_size])
    );
    
    let y = Tensor::<B, 2, burn::tensor::Int>::from_data(
        TensorData::new(y_indices, [batch_size, seq_length]),
        device,
    );

    (x, y)
}

fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, 
    temperature: f32,
    top_k: usize,
    top_p: f32
) -> usize
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive,
{
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<<B as Backend>::FloatElem>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, num_traits::ToPrimitive::to_f32(&x).unwrap_or(0.0)))
        .collect();

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs: Vec<(usize, f32)> = Vec::with_capacity(k);
    
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

fn generate_text<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    vocab_size: usize,
    device: &B::Device,
) -> String
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let mut generated_ids = tokenizer.encode(seed_text);
    let seed_tokens = generated_ids.clone();
    
    if seed_tokens.is_empty() {
        return seed_text.to_string();
    }

    let eye = Tensor::<B, 2>::eye(vocab_size, device);
    let mut current_state = None;
    let mut current_tokens = seed_tokens.clone();

    for i in 0..length {
        let tokens_to_process = if i == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };

        let seq_len = tokens_to_process.len();
        let indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(tokens_to_process.iter().map(|&t| t as i64).collect(), [seq_len]),
            device,
        );

        let input = eye.clone()
            .select(0, indices)
            .reshape([1, seq_len, vocab_size]);

        let (output, next_state) = model.forward(input, current_state);
        current_state = Some(next_state);

        let dims = output.dims();
        let last_logits = output
            .slice([0..1, (dims[1] - 1)..dims[1], 0..dims[2]])
            .reshape([1, dims[2]]);

        let next_token = sample_from_logits(last_logits, 0.4, 40, 0.9);

        current_tokens.push(next_token);
        generated_ids.push(next_token);
    }

    tokenizer.decode(&generated_ids[seed_tokens.len()..])
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM MLSTM + MINGRU Hybrid Text Generation");
    println!("==========================================\n");

    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin mlstm_mingru_chat -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer_hybrid.json";
    let model_path = "mlstm_mingru_chat_model";

    let target_vocab_size = 1024;

    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Entrenando nuevo tokenizador BPE...");
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.vocab_size();
    
    // Hiperparámetros
    let hidden_size = 256;
    let num_layers = 1;
    let dropout = 0.1;
    let seq_length = 128;
    let batch_size = 16;
    let stride = 128;
    let num_epochs = 50;
    let num_heads = 4;

    // DEFINICIÓN DEL ORDEN DE BLOQUES
    let block_layout = vec![
        BlockType::MLSTM, 
        BlockType::MINGRU, 
        BlockType::MLSTM, 
        BlockType::MINGRU, 
        BlockType::MLSTM,
        BlockType::MINGRU, 
        ];
    let num_blocks = block_layout.len();

    let lr_config = LearningRateConfig::per_block_type(
        1e-4, // sLSTM
        1e-3, // mLSTM (más rápido)
        1e-3, // minGRU
        1e-4, // Others
    );

 

    let device = Default::default();

    let config = XLstmconfig::new(vocab_size, hidden_size, num_layers, num_blocks, vocab_size)
        .with_dropout(dropout)
        .with_num_heads(num_heads)
        .with_use_conv(false)
        .with_use_mlp(false)
        .with_lstm_type(LstmType::Custom(block_layout))
        .with_use_projection(true);   

    let model_file = format!("{}.mpk", model_path);
    let existe_modelo = Path::new(&model_file).exists();
    
    let mut continuar_entrenamiento = false;
    if existe_modelo {
        print!("¿Deseas seguir entrenando el modelo cargado? (s/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "s" {
            continuar_entrenamiento = true;
        }
    }

    let mut model = if existe_modelo && !continuar_entrenamiento {
        println!("Cargando pesos para generación...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        config.init::<MyBackend>(&device).load_record(record)
    } else {
        if continuar_entrenamiento {
            println!("Cargando para continuar...");
            let record = CompactRecorder::new().load(model_file.into(), &device)?;
            config.init::<MyBackend>(&device).load_record(record)
        } else {
            println!("Iniciando desde cero...");
            config.init::<MyBackend>(&device)
        }
    };

    if !continuar_entrenamiento && existe_modelo {
        // Modo generación directa
    } else {
        model.print_architecture();
        let mut optim = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0)))
            .init();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let num_actual_sequences = (tokens.len().saturating_sub(seq_length) + stride - 1) / stride;
        let num_batches = num_actual_sequences.div_ceil(batch_size);

        for epoch in 0..num_epochs {
            let mut total_loss = 0.0f32;
            let mut correct = 0;
            let mut total = 0;
            let mut current_state = None;

            for batch_idx in 0..num_batches {
                let batch_start = Instant::now();
                let start_seq = batch_idx * batch_size;
                let current_batch_size = batch_size.min(num_actual_sequences - start_seq);
                
                // Si el último batch no está completo, lo saltamos para evitar errores de forma (reshape/states)
                if current_batch_size < batch_size { break; }

                let (input, target) = create_batch::<MyBackend>(&tokens, start_seq * stride, current_batch_size, seq_length, stride, vocab_size, &device);
                
                if batch_idx == 0 { current_state = None; }
                
                let (logits, next_state) = model.forward(input, current_state);
                
                // Truncated BPTT: Detach the state
                current_state = Some(
                    next_state
                        .into_iter()
                        .map(|opt_state| opt_state.map(|s| s.detach()))
                        .collect()
                );

                let logits_flat = logits.reshape([current_batch_size * seq_length, vocab_size]);
                let target_flat = target.reshape::<1, _>([current_batch_size * seq_length]);

                let loss = loss_fn.forward(logits_flat.clone(), target_flat.clone());
                total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];

                correct += logits_flat.argmax(1).reshape([current_batch_size * seq_length]).equal(target_flat).int().sum().into_data().as_slice::<i64>().unwrap()[0] as usize;
                total += current_batch_size * seq_length;

                let grads = loss.backward();
                model = model.optimizer_step(&lr_config, &mut optim, grads);

                if batch_idx % 1 == 0 || batch_idx == num_batches - 1 {
                    let elapsed = batch_start.elapsed().as_secs_f32();
                    print!("\r  -> Batch [{}/{}] Loss: {:.4}  Acc: {:.2}% ({:.2}s)    ", 
                        batch_idx + 1, num_batches, total_loss / (batch_idx + 1) as f32,
                        100.0 * correct as f32 / total as f32, elapsed);
                    io::stdout().flush()?;
                }
            }
            println!("\nGuardando época {}...", epoch + 1);
            model.clone().save_file(model_path, &CompactRecorder::new())?;
            
            let seed = "king ";
            let gen_start = Instant::now();
            let generated = generate_text(&model.valid(), &tokenizer, seed, 100, vocab_size, &device);
            println!("  Generado ({:.2}s): {}\n", gen_start.elapsed().as_secs_f32(), generated);
        }
    }

    // Modo interactivo simplificado
    println!("\n--- MODO INTERACTIVO ---");
    loop {
        print!("Semilla > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input == "exit" { break; }
        let generated = generate_text(&model.valid(), &tokenizer, input, 200, vocab_size, &device);
        println!("\n{}\n", generated);
    }

    Ok(())
}
