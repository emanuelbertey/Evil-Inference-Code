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

use xlstm::{MinGru, MinGruConfig, MinGruState};
use xlstm::components::conv::{CausalConv1d, CausalConv1dConfig};
use xlstm::blocks::xlstm_large::components::RMSNorm;

type MyBackend = Autodiff<Cuda<f32, i32>>;

/// Tokenizador de caracteres simple (Igual al de Jupyter)
pub struct CharTokenizer {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars = BTreeSet::new();
        for c in text.chars() {
            chars.insert(c);
        }
        
        let char_list: Vec<char> = chars.into_iter().collect();
        let mut char_to_idx = HashMap::new();
        let mut idx_to_char = HashMap::new();
        
        for (i, &c) in char_list.iter().enumerate() {
            char_to_idx.insert(c, i);
            idx_to_char.insert(i, c);
        }
        
        let vocab_size = char_list.len();
        Self { char_to_idx, idx_to_char, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_idx.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        indices.iter().map(|i| *self.idx_to_char.get(i).unwrap_or(&' ')).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub l1: Linear<B>,
    pub l2: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.l1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.l2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct LanguageModelLayer<B: Backend> {
    pub conv: CausalConv1d<B>,
    pub norm1: RMSNorm<B>,
    pub mingru: MinGru<B>,
    pub norm2: RMSNorm<B>,
    pub mlp: MLP<B>,
    pub dropout: burn::nn::Dropout,
}

impl<B: Backend> LanguageModelLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, state: Option<Vec<MinGruState<B>>>) -> (Tensor<B, 3>, Vec<MinGruState<B>>) {
        // Estabilizado: Estilo Pre-Norm Residual (Sync con mingruchat.rs)
        // 1. Conv + Residual
        let x = self.conv.forward(x.clone()) + x;
        
        // 2. Norm -> MinGru -> Residual
        let x_norm1 = self.norm1.forward(x.clone());
        let (output, next_state) = self.mingru.forward(x_norm1, state);
        let x = x + output;
        
        // 3. Norm -> MLP -> Residual
        let x_norm2 = self.norm2.forward(x.clone());
        let x = x + self.mlp.forward(x_norm2);
        
        // 4. Dropout final de capa
        let x = self.dropout.forward(x);
        
        (x, next_state)
    }

    pub fn step(&self, x_t: Tensor<B, 3>, conv_state: Tensor<B, 3>, mingru_state: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // Modo secuencial estabilizado (Sync con mingruchat.rs)
        let x_t_2d = x_t.clone().reshape([x_t.dims()[0], x_t.dims()[2]]);
        let (y_conv, next_conv_state) = self.conv.step(x_t_2d, conv_state);
        let x = y_conv.unsqueeze_dim(1) + x_t;
        
        let x_norm1 = self.norm1.forward(x.clone());
        let (y_mingru, next_mingru_state) = self.mingru.sequential_mode(x_norm1, mingru_state);
        let x = x + y_mingru;
        
        let x_norm2 = self.norm2.forward(x.clone());
        let x = x + self.mlp.forward(x_norm2);
        let x = self.dropout.forward(x);
        
        (x, next_conv_state, next_mingru_state)
    }
}

#[derive(Module, Debug)]
pub struct MinGruChatModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<LanguageModelLayer<B>>,
    pub norm: RMSNorm<B>,
    pub head: Linear<B>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

impl<B: Backend> MinGruChatModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>, _states: Option<Vec<Vec<MinGruState<B>>>>) -> (Tensor<B, 3>, Vec<Vec<MinGruState<B>>>) {
        let mut x = self.embedding.forward(input);
        let mut next_states = Vec::new();
        
        for layer in self.layers.iter() {
            let (out, ns) = layer.forward(x, None);
            x = out;
            next_states.push(ns);
        }
        
        x = self.norm.forward(x);
        let logits = self.head.forward(x);
        (logits, next_states)
    }

    pub fn step(&self, input: Tensor<B, 1, Int>, conv_states: &mut Vec<Tensor<B, 3>>, mingru_states: &mut Vec<Tensor<B, 3>>) -> Tensor<B, 2> {
        let [b] = input.dims();
        let mut x = self.embedding.forward(input.reshape([b, 1]));
        
        for i in 0..self.num_layers {
            let (out, next_conv, next_mingru) = self.layers[i].step(x, conv_states[i].clone(), mingru_states[i].clone());
            x = out;
            conv_states[i] = next_conv;
            mingru_states[i] = next_mingru;
        }
        
        x = self.norm.forward(x);
        self.head.forward(x).reshape([b, self.vocab_size])
    }
}

fn get_batches<B: Backend>(
    tokens: &[usize],
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Vec<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
    let total_length = tokens.len();
    let num_batches_total = (total_length - 1) / (batch_size * seq_length);
    let slice_len = num_batches_total * batch_size * seq_length;
    let data = &tokens[..slice_len];
    
    let cols = slice_len / batch_size;
    let mut reshaped = vec![vec![0usize; cols]; batch_size];
    for b in 0..batch_size {
        for c in 0..cols {
            reshaped[b][c] = data[b * cols + c];
        }
    }

    let mut batches = Vec::new();
    for i in (0..cols - seq_length).step_by(seq_length) {
        let mut x_indices = Vec::with_capacity(batch_size * seq_length);
        let mut y_indices = Vec::with_capacity(batch_size * seq_length);
        for b in 0..batch_size {
            for j in 0..seq_length {
                x_indices.push(reshaped[b][i + j] as i64);
                y_indices.push(reshaped[b][i + j + 1] as i64);
            }
        }
        let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), device);
        let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), device);
        batches.push((x, y));
    }
    batches
}

fn sample_from_logits<B: Backend>(logits: Tensor<B, 2>, temperature: f32) -> usize 
where <B as Backend>::FloatElem: num_traits::ToPrimitive {
    let probs = softmax(logits / temperature, 1);
    let data = probs.into_data();
    let probs_vec: Vec<f32> = data.as_slice::<<B as Backend>::FloatElem>().unwrap().iter().map(|&x| num_traits::ToPrimitive::to_f32(&x).unwrap()).collect();
    
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

fn generate_text<B: Backend>(
    model: &MinGruChatModel<B>,
    tokenizer: &CharTokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
) -> String
where <B as Backend>::FloatElem: num_traits::ToPrimitive {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return seed_text.to_string(); }

    let mut conv_states = Vec::new();
    let b = 1;

    for i in 0..model.num_layers {
        conv_states.push(model.layers[i].conv.empty_state(b, device));
    }
    let mut mingru_states = Vec::new();
    for _ in 0..model.num_layers {
        mingru_states.push(Tensor::<B, 3>::zeros([b, 1, model.hidden_size * 2], device));
    }

    // Prefill context
    for &id in &ids {
        let input = Tensor::<B, 1, Int>::from_ints(vec![id as i32].as_slice(), device);
        let _ = model.step(input, &mut conv_states, &mut mingru_states);
    }

    let mut generated = Vec::new();
    let mut last_id = *ids.last().unwrap();
    
    println!("--- Generando {} tokens en CUDA ---", length);
    let start_gen = Instant::now();

    for _ in 0..length {
        let input = Tensor::<B, 1, Int>::from_ints(vec![last_id as i32].as_slice(), device);
        let logits = model.step(input, &mut conv_states, &mut mingru_states);
        
        let next_id = sample_from_logits(logits, 0.7);
        generated.push(next_id);
        last_id = next_id;
        
        let token = tokenizer.decode(&[next_id]);
        print!("{}", token);
        io::stdout().flush().unwrap();
    }
    
    let elapsed = start_gen.elapsed().as_secs_f32();
    let tps = length as f32 / elapsed;
    println!("\n\n[Velocidad CUDA: {:.2} tokens/s | Tiempo: {:.2}s]", tps, elapsed);
    tokenizer.decode(&generated)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut text_file = String::new();

    if args.len() >= 2 {
        text_file = args[1].clone();
    }

    println!("=== MinGRU Chat CUDA System (Interactive Menu) ===");
    println!("1. Entrenar nuevo modelo o continuar entrenamiento");
    println!("2. Solo Generación");
    print!("Seleccione una opción: ");
    io::stdout().flush().unwrap();
    
    let mut input_op = String::new();
    io::stdin().read_line(&mut input_op).unwrap();
    let option = input_op.trim();

    if text_file.is_empty() {
        print!("Ruta del archivo de texto (ej: input.txt): ");
        io::stdout().flush().unwrap();
        let mut text_file_input = String::new();
        io::stdin().read_line(&mut text_file_input).unwrap();
        text_file = text_file_input.trim().to_string();
    }

    let mut gen_len = 0;
    let mut seed = "The ".to_string();

    if option == "2" {
        print!("Longitud a generar: ");
        io::stdout().flush().unwrap();
        let mut len_str = String::new();
        io::stdin().read_line(&mut len_str).unwrap();
        gen_len = len_str.trim().parse().unwrap_or(100);

        print!("Semilla de texto (Enter para 'The '): ");
        io::stdout().flush().unwrap();
        let mut seed_str = String::new();
        io::stdin().read_line(&mut seed_str).unwrap();
        let trimmed_seed = seed_str.trim();
        if !trimmed_seed.is_empty() {
            seed = trimmed_seed.to_string();
        }
    }

    let model_path = "mingru_cuda_stable";
    let text = fs::read_to_string(text_file)?;
    
    let tokenizer = CharTokenizer::from_text(&text);
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size (Characters): {}", vocab_size);

    let tokens = tokenizer.encode(&text);
    let hidden_size = 256;
    let num_layers = 3;
    let mlp_expansion = 4;
    let device = CudaDevice::default();

    let mut layers = Vec::new();
    for _ in 0..num_layers {
        layers.push(LanguageModelLayer {
            conv: CausalConv1dConfig::new(hidden_size, 4).init(&device),
            norm1: RMSNorm::init(hidden_size, true, false, 1e-4, true, &device),
            mingru: MinGruConfig { input_features: hidden_size, expansion_factor: 2 }.init(&device),
            norm2: RMSNorm::init(hidden_size, true, false, 1e-4, true, &device),
            mlp: MLP {
                l1: LinearConfig::new(hidden_size, hidden_size * mlp_expansion).init(&device),
                l2: LinearConfig::new(hidden_size * mlp_expansion, hidden_size).init(&device),
            },
            dropout: burn::nn::DropoutConfig::new(0.2).init(),
        });
    }

    let mut model: MinGruChatModel<MyBackend> = MinGruChatModel {
        embedding: EmbeddingConfig::new(vocab_size, hidden_size).init(&device),
        layers,
        norm: RMSNorm::init(hidden_size, true, false, 1e-4, true, &device),
        head: LinearConfig::new(hidden_size, vocab_size).with_bias(false).init(&device),
        vocab_size,
        hidden_size,
        num_layers,
    };

    let model_file = format!("{}.mpk", model_path);
    let mut model_exists = false;
    if Path::new(&model_file).exists() {
        println!("Cargando modelo...");
        let record = CompactRecorder::new().load(model_file.into(), &device)?;
        model = model.load_record(record);
        model_exists = true;
    }

    if gen_len > 0 {
        if !model_exists {
            println!("[!] Error: No hay modelo guardado para generar.");
            return Ok(());
        }
        generate_text(&model.valid(), &tokenizer, &seed, gen_len, &device);
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let batch_size = 64;
    let seq_len = 512;

    let batches = get_batches::<MyBackend>(&tokens, batch_size, seq_len, &device);
    let num_batches = batches.len();

    println!("Iniciando entrenamiento en CUDA (Batch: 64, Seq: 512, LR: 2e-3)...");
    for epoch in 0..50 {
        let mut total_loss = 0.0;
        let start_epoch = Instant::now();
        
        for (b, (x, y)) in batches.iter().enumerate() {
            let (logits, _) = model.forward(x.clone(), None);
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = y.clone().reshape([batch_size * seq_len]);
            
            let loss = loss_fn.forward(logits_flat, targets_flat);
            let current_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            
            if current_loss.is_nan() {
                println!("\n[!] FATAL: Loss es NaN en epoch {} batch {}. Abortando.", epoch+1, b);
                //std::process::exit(1);
            }

            total_loss += current_loss;
            
            let grads = loss.backward();
            let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(2e-3, model, grads_p); // Subido LR a 2e-3 como pediste
            
            if b % 2 == 0 {
                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = ((b + 1) * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {}/50 | Batch {}/{} | Loss: {:.4} | Speed: {:.1} tok/s", epoch+1, b, num_batches, total_loss / (b+1) as f32, tps);
                io::stdout().flush().unwrap();
            }
        }
        
        println!("\nEpoch {} completa en {:.2}s. Loss promedio: {:.4}", epoch+1, start_epoch.elapsed().as_secs_f32(), total_loss / num_batches as f32);
        
        let recorder = CompactRecorder::new();
        model.clone().save_file(model_path, &recorder)?;

        if (epoch + 1) % 5 == 0 {
            let checkpoint_name = format!("{}_epoch_{}", model_path, epoch + 1);
            model.clone().save_file(&checkpoint_name, &recorder)?;
            println!(" -> Checkpoint guardado: {}.mpk", checkpoint_name);
        }

        if epoch % 1 == 0 {
            println!("--- Generación de prueba ---");
            generate_text(&model.clone().valid(), &tokenizer, "The ", 100, &device);
            println!("\n---------------------------");
        }
    }

    Ok(())
}
