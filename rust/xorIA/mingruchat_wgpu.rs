#![recursion_limit = "256"]
use burn::prelude::*;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{EmbeddingConfig, LinearConfig, DropoutConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::grad_clipping::GradientClippingConfig;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_autodiff::Autodiff;
use std::fs;
use std::io::{self, Write};
use std::error::Error;
use std::time::Instant;

use xlstm::blocks::minrnn::mingru::{MinGru, MinGruConfig, MinGruState};
use xlstm::components::conv::{CausalConv1d, CausalConv1dConfig};
use xlstm::blocks::xlstm_large::RMSNorm;
use burn::tensor::activation;

type MyBackend = Autodiff<Wgpu<f32, i32>>;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub l1: nn::Linear<B>,
    pub l2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.l1.forward(x);
        let x = activation::gelu(x);
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
    pub dropout: nn::Dropout,
}

impl<B: Backend> LanguageModelLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, h0: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x_res = self.conv.forward(x.clone()) + x;
        let x_norm = self.norm1.forward(x_res.clone());
        
        let (output, next_states) = self.mingru.forward(x_norm, Some(vec![MinGruState::new(h0)]));
        let h_next = next_states[0].hidden.clone();
        
        let x = output + x_res; 
        let x_norm2 = self.norm2.forward(x);
        let x = self.mlp.forward(x_norm2.clone()) + x_norm2;
        let x = self.dropout.forward(x.clone()) + x;
        
        (x, h_next)
    }
}

#[derive(Module, Debug)]
pub struct MinGruChatModel<B: Backend> {
    pub embedding: nn::Embedding<B>,
    pub layers: Vec<LanguageModelLayer<B>>,
    pub norm: RMSNorm<B>,
    pub head: nn::Linear<B>,
}

impl<B: Backend> MinGruChatModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>, mut h0: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut x = self.embedding.forward(input);
        for i in 0..self.layers.len() {
            let (out, h_next) = self.layers[i].forward(x, h0);
            x = out;
            h0 = h_next;
        }
        x = self.norm.forward(x);
        let logits = self.head.forward(x);
        (logits, h0)
    }

    pub fn generate(&self, tokenizer: &CharTokenizer, seed: &str, length: usize, device: &B::Device) -> String {
        let mut tokens = tokenizer.encode(seed);
        let h_size = self.embedding.weight.dims()[1] * 2;
        for _ in 0..length {
            let input_indices: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let input = Tensor::<B, 2, Int>::from_data(TensorData::new(input_indices, [1, tokens.len()]), device);
            let (logits, _) = self.forward(input, Tensor::zeros([1, 1, h_size], device)); 
            let [_, s, v] = logits.dims();
            let last_logits = logits.slice([0..1, s-1..s, 0..v]).flatten::<1>(0, 2);
            let probs = activation::softmax(last_logits / 0.8, 0).into_data().as_slice::<f32>().unwrap().to_vec();
            let mut r = rand::random::<f32>();
            let mut id = 0;
            for (idx, &p) in probs.iter().enumerate() { r -= p; if r <= 0.0 { id = idx; break; } }
            tokens.push(id);
            if tokens.len() > 512 { break; } 
        }
        tokenizer.decode(&tokens)
    }
}

pub struct CharTokenizer {
    chars: Vec<char>,
    char_to_id: std::collections::HashMap<char, usize>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect(); chars.sort(); chars.dedup();
        let char_to_id = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { chars, char_to_id }
    }
    pub fn vocab_size(&self) -> usize { self.chars.len() }
    pub fn encode(&self, text: &str) -> Vec<usize> { text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect() }
    pub fn decode(&self, ids: &[usize]) -> String { ids.iter().map(|&id| self.chars[id]).collect() }
}

fn get_batches(tokens: &[usize], batch_size: usize, seq_length: usize) -> Vec<(Vec<i32>, Vec<i32>)> {
    let num_batches = (tokens.len() - 1) / (batch_size * seq_length);
    let stream_len = num_batches * seq_length;
    let mut batch_idx = Vec::new();
    for i in (0..stream_len).step_by(seq_length) {
        let mut x_indices = Vec::with_capacity(batch_size * seq_length);
        let mut y_indices = Vec::with_capacity(batch_size * seq_length);
        for b in 0..batch_size {
            let offset = b * (tokens.len() / batch_size);
            for j in 0..seq_length {
                x_indices.push(tokens[offset + i + j] as i32);
                y_indices.push(tokens[offset + i + j + 1] as i32);
            }
        }
        batch_idx.push((x_indices, y_indices));
    }
    batch_idx
}

fn main() -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string("input.txt")?;
    let tokenizer = CharTokenizer::from_text(&text);
    let (device, h_dim, n_layers) = (WgpuDevice::default(), 256, 3);
    
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        layers.push(LanguageModelLayer {
            conv: CausalConv1dConfig::new(h_dim, 4).init(&device),
            norm1: RMSNorm::init(h_dim, true, false, 1e-4, true, &device),
            mingru: MinGruConfig { input_features: h_dim, expansion_factor: 2 }.init(&device),
            norm2: RMSNorm::init(h_dim, true, false, 1e-4, true, &device),
            mlp: MLP { l1: LinearConfig::new(h_dim, h_dim * 4).init(&device), 
                       l2: LinearConfig::new(h_dim * 4, h_dim).init(&device) },
            dropout: DropoutConfig::new(0.2).init(),
        });
    }

    let mut model: MinGruChatModel<MyBackend> = MinGruChatModel {
        embedding: EmbeddingConfig::new(tokenizer.vocab_size(), h_dim).init(&device),
        layers, norm: RMSNorm::init(h_dim, true, false, 1e-4, true, &device),
        head: LinearConfig::new(h_dim, tokenizer.vocab_size()).with_bias(false).init(&device),
    };

    let mut optim = AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let (bs, slen) = (32, 256); // Ajustado para 1GB VRAM aprox.
    let batches = get_batches(&tokenizer.encode(&text), bs, slen);
    let mut h0 = Tensor::zeros([bs, 1, h_dim * 2], &device);

    for epoch in 0..50 {
        let (mut t_loss, e_start, mut b_start) = (0.0, Instant::now(), Instant::now());
        for (b, (xi, yi)) in batches.iter().enumerate() {
            let x = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(xi.clone(), [bs, slen]), &device);
            let y = Tensor::<MyBackend, 2, Int>::from_data(TensorData::new(yi.clone(), [bs, slen]), &device);
            
            let (logits, next_h0) = model.forward(x, h0.detach());
            h0 = next_h0;
            
            let [bd, sd, vd] = logits.dims();
            let loss = loss_fn.forward(logits.reshape([bd * sd, vd]), y.reshape([bd * sd]));
            t_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(2e-3, model, grads); 
            
            if b % 10 == 0 {
                let tps = 10.0 * (bs * slen) as f32 / b_start.elapsed().as_secs_f32();
                print!("\rEp {:>2} | B {:>4}/{} | L: {:.4} | TPS: {:>8.1} | T: {:.1}s", 
                    epoch+1, b, batches.len(), t_loss/(b+1) as f32, tps, e_start.elapsed().as_secs_f32());
                io::stdout().flush()?; b_start = Instant::now();
            }
        }
        println!("\nSample: \"{}\"", model.generate(&tokenizer, "The ", 100, &device).replace('\n', "\\n"));
    }
    Ok(())
}
