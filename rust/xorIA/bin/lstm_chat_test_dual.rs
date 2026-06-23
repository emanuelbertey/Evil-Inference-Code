// ─── LSTM Chat Test Dual: Normal vs BitLinear Comparison ───────────────────
//
// This test compares two character-level LSTM models:
//   1. Normal LSTM: Uses standard nn::Linear layers for gates.
//   2. BitLinear LSTM: Uses BitLinear ternary {-1, 0, +1} layers for gates.
//
// Both models are trained on the same text data to predict the next character.
// We compare their convergence, loss, and text generation quality.

use burn::prelude::*;
use burn::tensor::{Tensor, Int, TensorData};
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use burn::module::{Module, AutodiffModule};
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};
use burn::optim::{AdamConfig, Optimizer};
use burn::tensor::activation::{softmax, sigmoid, tanh};
use std::collections::{HashMap, BTreeSet};

use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearConfig};

type MyBackend = Autodiff<Flex<f32>>;

// ─── Char Tokenizer ──────────────────────────────────────────────────────────

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

// ─── Custom LSTM Cell Implementation ────────────────────────────────────────

#[derive(Module, Debug)]
pub struct LstmCellNormal<B: Backend> {
    pub input_gate: Linear<B>,
    pub hidden_gate: Linear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> LstmCellNormal<B> {
    pub fn forward(&self, x: Tensor<B, 2>, h: Tensor<B, 2>, c: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Concatenated gates for efficiency
        let gates = self.input_gate.forward(x) + self.hidden_gate.forward(h);
        let chunks = gates.chunk(4, 1);
        
        let i = sigmoid(chunks[0].clone());
        let f = sigmoid(chunks[1].clone());
        let g = tanh(chunks[2].clone());
        let o = sigmoid(chunks[3].clone());
        
        let c_next = f * c + i * g;
        let h_next = o * tanh(c_next.clone());
        
        (h_next, c_next)
    }
}

#[derive(Module, Debug)]
pub struct LstmCellBit<B: Backend> {
    pub input_gate: BitLinear<B>,
    pub hidden_gate: BitLinear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> LstmCellBit<B> {
    pub fn forward(&self, x: Tensor<B, 2>, h: Tensor<B, 2>, c: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // BitLinear's forward_2d for the step
        let gates = self.input_gate.forward_2d(x) + self.hidden_gate.forward_2d(h);
        let chunks = gates.chunk(4, 1);
        
        let i = sigmoid(chunks[0].clone());
        let f = sigmoid(chunks[1].clone());
        let g = tanh(chunks[2].clone());
        let o = sigmoid(chunks[3].clone());
        
        let c_next = f * c + i * g;
        let h_next = o * tanh(c_next.clone());
        
        (h_next, c_next)
    }
}

// ─── Model Wrapper ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct LstmModel<B: Backend, Cell: Module<B>> {
    pub embedding: Embedding<B>,
    pub cell: Cell,
    pub head: Linear<B>,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl<B: Backend> LstmModel<B, LstmCellNormal<B>> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();
        let x = self.embedding.forward(input);
        
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &device);
        
        let mut outputs = Vec::new();
        
        for t in 0..seq_len {
            let x_t = x.clone().slice([0..batch_size, t..t+1, 0..self.hidden_size]).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_t, h, c);
            h = h_next;
            c = c_next;
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }
        
        let full_output = Tensor::cat(outputs, 1);
        self.head.forward(full_output)
    }

    pub fn generate(&self, seed: Tensor<B, 2, Int>, length: usize) -> Vec<usize> {
        let [batch_size, _] = seed.dims();
        let device = seed.device();
        let x = self.embedding.forward(seed);
        
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &device);
        
        // Prefill
        let seq_len = x.dims()[1];
        let mut last_h = h.clone();
        for t in 0..seq_len {
            let x_t = x.clone().slice([0..batch_size, t..t+1, 0..self.hidden_size]).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_t, h, c);
            h = h_next;
            c = c_next;
            last_h = h.clone();
        }
        
        let mut generated = Vec::new();
        let mut current_h = last_h;
        
        let mut logits = self.head.forward(current_h.clone().unsqueeze_dim::<3>(1)).reshape::<2, _>([batch_size, self.vocab_size]);
        
        for _ in 0..length {
            let probs = softmax(logits / 0.8, 1); 
            
            let probs_vec: Vec<f32> = probs.into_data().as_slice::<f32>().unwrap().to_vec();
            let next_id = sample_from_probs(probs_vec);
            generated.push(next_id);
            
            let next_token = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next_id as i32], [1]), &device);
            let x_next = self.embedding.forward(next_token.reshape::<2, _>([batch_size, 1])).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_next, current_h, c);
            current_h = h_next;
            c = c_next;
            logits = self.head.forward(current_h.clone().unsqueeze_dim::<3>(1)).reshape::<2, _>([batch_size, self.vocab_size]);
        }
        
        generated
    }
}

impl<B: Backend> LstmModel<B, LstmCellBit<B>> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();
        let x = self.embedding.forward(input);
        
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &device);
        
        let mut outputs = Vec::new();
        
        for t in 0..seq_len {
            let x_t = x.clone().slice([0..batch_size, t..t+1, 0..self.hidden_size]).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_t, h, c);
            h = h_next;
            c = c_next;
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }
        
        let full_output = Tensor::cat(outputs, 1);
        self.head.forward(full_output)
    }

    pub fn generate(&self, seed: Tensor<B, 2, Int>, length: usize) -> Vec<usize> {
        let [batch_size, _] = seed.dims();
        let device = seed.device();
        let x = self.embedding.forward(seed);
        
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &device);
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &device);
        
        // Prefill
        let seq_len = x.dims()[1];
        let mut last_h = h.clone();
        for t in 0..seq_len {
            let x_t = x.clone().slice([0..batch_size, t..t+1, 0..self.hidden_size]).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_t, h, c);
            h = h_next;
            c = c_next;
            last_h = h.clone();
        }
        
        let mut generated = Vec::new();
        let mut current_h = last_h;
        
        let mut logits = self.head.forward(current_h.clone().unsqueeze_dim::<3>(1)).reshape::<2, _>([batch_size, self.vocab_size]);
        
        for _ in 0..length {
            let probs = softmax(logits / 0.8, 1); 
            
            let probs_vec: Vec<f32> = probs.into_data().as_slice::<f32>().unwrap().to_vec();
            let next_id = sample_from_probs(probs_vec);
            generated.push(next_id);
            
            let next_token = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next_id as i32], [1]), &device);
            let x_next = self.embedding.forward(next_token.reshape::<2, _>([batch_size, 1])).reshape::<2, _>([batch_size, self.hidden_size]);
            let (h_next, c_next) = self.cell.forward(x_next, current_h, c);
            current_h = h_next;
            c = c_next;
            logits = self.head.forward(current_h.clone().unsqueeze_dim::<3>(1)).reshape::<2, _>([batch_size, self.vocab_size]);
        }
        
        generated
    }
}

// ─── Training Helpers ──────────────────────────────────────────────────────

fn sample_from_probs(probs: Vec<f32>) -> usize {
    let mut rng = rand::rng();
    use rand::Rng;
    let sample: f32 = rng.random::<f32>();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative { return i; }
    }
    0
}

fn create_batch<B: Backend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i;
        for j in 0..seq_length {
            x_indices.push(tokens[current_start + j] as i64);
            y_indices.push(tokens[current_start + j + 1] as i64);
        }
    }

    let x = Tensor::<B, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), device);
    let y = Tensor::<B, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), device);
    (x, y)
}

fn cross_entropy_loss<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
    let targets_flat = targets.reshape([batch_size * seq_len]);
    
    let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
    let target_log_probs = log_probs.gather(1, targets_flat.unsqueeze_dim(1));
    target_log_probs.mean().neg().reshape([1])
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          LSTM DUAL TEST: Normal vs BitLinear (Ternary)         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let device = Default::default();
    
    let text = std::fs::read_to_string("xorIA/input.txt")
        .unwrap_or_else(|_| "Failed to find xorIA/input.txt, using fallback text.".to_string());
    
    // Take a significant slice for testing if it's too large
    let text_slice = if text.len() > 100_000 { &text[0..100_000] } else { &text };
                
    let tokenizer = CharTokenizer::from_text(text_slice);
    let vocab_size = tokenizer.vocab_size();
    let tokens = tokenizer.encode(text_slice);
    
    println!("Vocab Size: {} | Total Tokens: {}", vocab_size, tokens.len());

    let hidden_size = 256;
    let seq_len = 64;
    let batch_size = 20;
    let steps = 1000;
    let lr = 8e-4;

    // 1. Normal Model
    let mut normal_model: LstmModel<MyBackend, LstmCellNormal<MyBackend>> = LstmModel {
        embedding: EmbeddingConfig::new(vocab_size, hidden_size).init(&device),
        cell: LstmCellNormal {
            input_gate: LinearConfig::new(hidden_size, 4 * hidden_size).with_bias(true).init(&device),
            hidden_gate: LinearConfig::new(hidden_size, 4 * hidden_size).with_bias(true).init(&device),
            hidden_size,
        },
        head: LinearConfig::new(hidden_size, vocab_size).with_bias(false).init(&device),
        hidden_size,
        vocab_size,
    };
    
    // 2. Bit Model
    let mut bit_model: LstmModel<MyBackend, LstmCellBit<MyBackend>> = LstmModel {
        embedding: EmbeddingConfig::new(vocab_size, hidden_size).init(&device),
        cell: LstmCellBit {
            input_gate: BitLinearConfig {
                in_features: hidden_size,
                out_features: 4 * hidden_size,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
                quantized: true,
            }.init(&device),
            hidden_gate: BitLinearConfig {
                in_features: hidden_size,
                out_features: 4 * hidden_size,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
                quantized: true,
            }.init(&device),
            hidden_size,
        },
        head: LinearConfig::new(hidden_size, vocab_size).with_bias(false).init(&device),
        hidden_size,
        vocab_size,
    };

    let mut normal_optim = AdamConfig::new().init();
    let mut bit_optim = AdamConfig::new().init();

    println!("\n━━━ Training Both Models ━━━");
    
    let mut normal_losses = Vec::new();
    let mut bit_losses = Vec::new();

    for step in 1..=steps {
        // Use a random starting point for each batch
        let max_start = tokens.len().saturating_sub(seq_len + 1);
        if max_start == 0 { break; }
        
        let mut rng = rand::rng();
        use rand::Rng;
        let start_idx = rng.random_range(0..max_start);
        let (x, y) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_len, &device);
        
        // Train Normal
        let logits_n = normal_model.forward(x.clone());
        let loss_n = cross_entropy_loss(logits_n, y.clone());
        let loss_n_val: f32 = loss_n.clone().into_scalar().elem();
        
        let grads_n = loss_n.backward();
        let gp_n = burn::optim::GradientsParams::from_grads(grads_n, &normal_model);
        normal_model = normal_optim.step(lr, normal_model, gp_n);
        normal_losses.push(loss_n_val);
        
        // Train Bit
        let logits_b = bit_model.forward(x.clone());
        let loss_b = cross_entropy_loss(logits_b, y.clone());
        let loss_b_val: f32 = loss_b.clone().into_scalar().elem();
        
        let grads_b = loss_b.backward();
        let gp_b = burn::optim::GradientsParams::from_grads(grads_b, &bit_model);
        bit_model = bit_optim.step(lr, bit_model, gp_b);
        bit_losses.push(loss_b_val);
        
        if step % 100 == 0 || step == 1 {
            println!("Step {:4}/{:4} | Normal Loss: {:.6} | Bit Loss: {:.6}", 
                step, steps, loss_n_val, loss_b_val);
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     GENERATION COMPARISON                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let seed_text = "The ";
    let seed_tokens = tokenizer.encode(seed_text);
    let seed_tensor = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::new(seed_tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(), [1, seed_tokens.len()]),
        &device
    );

    println!("Seed: '{}'", seed_text);
    
    let gen_n = normal_model.valid().generate(seed_tensor.clone().inner(), 30);
    println!("\n[Normal Model Prediction]:");
    println!("  {}", tokenizer.decode(&gen_n));
    
    let gen_b = bit_model.valid().generate(seed_tensor.clone().inner(), 30);
    println!("\n[BitLinear Model Prediction]:");
    println!("  {}", tokenizer.decode(&gen_b));

    println!("\n━━━ Final Statistics ━━━");
    println!("Normal Final Loss: {:.6}", normal_losses.last().unwrap());
    println!("Bit Final Loss:    {:.6}", bit_losses.last().unwrap());
    
    let diff = (bit_losses.last().unwrap() - normal_losses.last().unwrap()).abs();
    println!("Loss Difference:   {:.6}", diff);
    
    if bit_losses.last().unwrap() < normal_losses.last().unwrap() {
        println!("🏆 BitLinear achieved LOWER loss in this run!");
    } else {
        println!("🏆 Normal Linear achieved LOWER loss in this run!");
    }

    println!("\n═══ TEST COMPLETE ═══");
}
