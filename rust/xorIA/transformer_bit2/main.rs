// ─── Transformer Bit2: BitLinear (1.58-bit) Transformer Chat ────────────────
//
// Versión del transformer_chat que reemplaza Linear por BitLinear (ternary {-1,0,+1})
// con per-group quantization (GS=128) y Straight-Through Estimator (STE).
//
// Mantiene: GQA + RoPE + SwiGLU + KV Cache + Top-K/P + Repetition Penalty
// Cambia: Linear → BitLinear (RMSNorm + 8-bit act quant + ternary weight quant)
//
// Architecture:
//   Embedding → TransformerBitLinear(N layers × GQA+RoPE+BitLinear_SwiGLU) → BitLinear → logits
//
// Usage:
//   cargo run --bin transformer_bit2 --release -- xorIA/input.txt

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::{Module, AutodiffModule},
    optim::{AdamConfig, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::Backend, TensorData, Int},
    nn::loss::CrossEntropyLossConfig,
    nn::{Embedding, EmbeddingConfig},
};
use burn_autodiff::Autodiff;
use burn_flex::Flex;
use std::error::Error;
use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

use tokenizers::AddedToken;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearConfig};

// ─── Type Alias ──────────────────────────────────────────────────────────────

type MyBackend = Autodiff<Flex<f32>>;

// ─── BPE Tokenizer ──────────────────────────────────────────────────────────

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
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, false)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, false)));

        let special_token = "eos";
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
        let temp_file = "temp_train_transformer_bit2.txt";
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
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, false)));
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

// ─── File Fragment Iterator (Streaming) ─────────────────────────────────────

struct FileFragmentIterator {
    reader: BufReader<fs::File>,
    buffer_size: usize,
    finished: bool,
}

impl FileFragmentIterator {
    fn new(path: &Path, buffer_size_mb: usize) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            buffer_size: buffer_size_mb * 1024 * 1024,
            finished: false,
        })
    }
}

impl Iterator for FileFragmentIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }

        let mut buffer = vec![0u8; self.buffer_size];
        let mut total_read = 0;

        while total_read < self.buffer_size {
            match self.reader.read(&mut buffer[total_read..]) {
                Ok(0) => { self.finished = true; break; }
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.finished = true; break; }
            }
        }

        if total_read == 0 { return None; }
        buffer.truncate(total_read);

        while !buffer.is_empty() && String::from_utf8(buffer.clone()).is_err() {
            buffer.pop();
        }

        if buffer.is_empty() { return None; }
        String::from_utf8(buffer).ok()
    }
}

// ─── BitLinear Attention Projection (Q/K/V) ────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearQKVProjection<B: Backend> {
    pub q_proj: BitLinear<B>,
    pub k_proj: BitLinear<B>,
    pub v_proj: BitLinear<B>,
    pub num_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
}

impl<B: Backend> BitLinearQKVProjection<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();

        let q = self.q_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);

        (q, k, v)
    }
}

// ─── BitLinear Output Projection ────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearOutputProjection<B: Backend> {
    pub o_proj: BitLinear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> BitLinearOutputProjection<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        let x_merged = x.reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(x_merged)
    }
}

// ─── BitLinear SwiGLU FeedForward ───────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearSwiGLUFeedForward<B: Backend> {
    pub gate_up_proj: BitLinear<B>,
    pub down_proj: BitLinear<B>,
    pub dropout: burn::nn::Dropout,
    pub intermediate_dim: usize,
}

impl<B: Backend> BitLinearSwiGLUFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward(x);

        // Split into gate and up projections
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();

        // SwiGLU activation: SiLU(gate) * up
        let h = burn::tensor::activation::silu(gate) * up;
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }
}

// ─── BitLinear Transformer Layer ────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearTransformerLayer<B: Backend> {
    pub attn_norm: BitLinearRMSNorm<B>,
    pub qkv: BitLinearQKVProjection<B>,
    pub o_proj: BitLinearOutputProjection<B>,
    pub ffn_norm: BitLinearRMSNorm<B>,
    pub ffn: BitLinearSwiGLUFeedForward<B>,
    pub residual_dropout: burn::nn::Dropout,
}

impl<B: Backend> BitLinearTransformerLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        // 1. Pre-Norm → Attention → Residual
        let residual = x.clone();
        let h = self.attn_norm.forward(x);
        let h = self.attention_forward(h, offset);
        let h = self.residual_dropout.forward(h);
        let x = residual + h;

        // 2. Pre-Norm → FFN → Residual
        let residual = x.clone();
        let h = self.ffn_norm.forward(x);
        let h = self.ffn.forward(h);
        let h = self.residual_dropout.forward(h);
        residual + h
    }

    fn attention_forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_batch, seq_len, _d] = x.dims();

        // 1. Project to Q, K, V with per-head shapes
        let (q, k, v) = self.qkv.forward(x);

        // 2. Apply RoPE to Q and K
        let (q, k) = apply_rope(q, k, offset);

        // 3. Repeat KV groups to match num_heads (GQA broadcast)
        let k = repeat_kv(k, self.qkv.num_heads, self.qkv.num_kv_groups);
        let v = repeat_kv(v, self.qkv.num_heads, self.qkv.num_kv_groups);

        // 4. Transpose for attention: (B, num_heads, S, head_dim)
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // 5. Scaled dot-product attention
        let scale = (self.qkv.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;

        // 6. Causal mask
        if seq_len > 1 {
            scores = apply_causal_mask(scores, seq_len);
        }

        // 7. Softmax + Dropout
        let attn_weights = softmax(scores, 3);

        // 8. Weighted sum of values
        let attn_output = attn_weights.matmul(v);

        // 9. Transpose back and project output
        let attn_output = attn_output.swap_dims(1, 2);
        self.o_proj.forward(attn_output)
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        offset: usize,
        cache: Option<KVCache<B>>,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        // 1. Pre-Norm → Attention with cache → Residual
        let residual = x.clone();
        let h = self.attn_norm.forward(x);
        let (h, new_cache) = self.attention_with_cache(h, offset, cache);
        let h = self.residual_dropout.forward(h);
        let x = residual + h;

        // 2. Pre-Norm → FFN → Residual
        let residual = x.clone();
        let h = self.ffn_norm.forward(x);
        let h = self.ffn.forward(h);
        let h = self.residual_dropout.forward(h);
        (residual + h, new_cache)
    }

    fn attention_with_cache(
        &self,
        x: Tensor<B, 3>,
        offset: usize,
        cache: Option<KVCache<B>>,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        // 1. Project to Q, K, V
        let (q, k_new, v_new) = self.qkv.forward(x);

        // 2. Apply RoPE to Q and K (with offset for position tracking)
        let (q, k_new) = apply_rope(q, k_new, offset);

        // 3. Concatenate with cached K, V if available
        let (k_full, v_full) = if let Some(prev) = cache {
            let k_cat = Tensor::cat(vec![prev.cached_k, k_new.clone()], 1);
            let v_cat = Tensor::cat(vec![prev.cached_v, v_new.clone()], 1);
            (k_cat, v_cat)
        } else {
            (k_new.clone(), v_new.clone())
        };

        // 4. Store the updated cache (before GQA expansion, to save memory)
        let new_cache = KVCache {
            cached_k: k_full.clone(),
            cached_v: v_full.clone(),
        };

        // 5. Expand KV groups for GQA
        let k_expanded = repeat_kv(k_full, self.qkv.num_heads, self.qkv.num_kv_groups);
        let v_expanded = repeat_kv(v_full, self.qkv.num_heads, self.qkv.num_kv_groups);

        // 6. Transpose: (B, S, H, D) → (B, H, S, D)
        let q = q.swap_dims(1, 2);
        let k = k_expanded.swap_dims(1, 2);
        let v = v_expanded.swap_dims(1, 2);

        // 7. Scaled dot-product attention
        let scale = (self.qkv.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;

        // 8. Causal mask (only needed during prefill when new_seq_len > 1)
        let [_, _, q_len, kv_len] = scores.dims();
        if q_len > 1 {
            scores = apply_causal_mask_with_offset(scores, q_len, kv_len);
        }

        // 9. Softmax + Dropout
        let attn_weights = softmax(scores, 3);

        // 10. Weighted sum
        let attn_output = attn_weights.matmul(v);

        // 11. Transpose back and project
        let attn_output = attn_output.swap_dims(1, 2);
        let output = self.o_proj.forward(attn_output);

        (output, new_cache)
    }
}

// ─── BitLinear RMSNorm ─────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearRMSNorm<B: Backend> {
    pub weight: burn::module::Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> BitLinearRMSNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: burn::module::Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = x.clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp_min(self.eps as f32);
        let normed = x / rms;
        normed * self.weight.val().unsqueeze::<2>().unsqueeze::<3>()
    }
}

// ─── KV Cache ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub cached_k: Tensor<B, 4>,
    pub cached_v: Tensor<B, 4>,
}

impl<B: Backend> KVCache<B> {
    pub fn keep_last(&self, keep: usize) -> KVCache<B> {
        let [b, seq, g, d] = self.cached_k.dims();
        if keep == 0 {
            return self.clone();
        }
        let keep = keep.min(seq);
        if keep == seq {
            return self.clone();
        }

        let start = seq - keep;
        let k = self.cached_k.clone().slice([0..b, start..seq, 0..g, 0..d]);
        let v = self.cached_v.clone().slice([0..b, start..seq, 0..g, 0..d]);

        KVCache { cached_k: k, cached_v: v }
    }
}

// ─── RoPE (Rotary Position Embeddings) ─────────────────────────────────────

fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_batch, seq_len, _num_heads, head_dim] = q.dims();

    // Compute theta = 1 / (base^(2i/dim))
    let theta: Vec<f32> = (0..head_dim / 2)
        .map(|i| {
            let exponent = 2.0 * i as f32 / head_dim as f32;
            1.0 / 10000.0f32.powf(exponent)
        })
        .collect();

    let theta_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(theta, [head_dim / 2]),
        &q.device(),
    );

    // Positions: [seq_len]
    let positions: Vec<f32> = (offset..offset + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(positions, [seq_len]),
        &q.device(),
    );

    // Compute angles: [seq_len, head_dim/2]
    let angles = pos_tensor.reshape([seq_len, 1]) * theta_tensor.reshape([1, head_dim / 2]);

    // cos and sin: [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim/2]
    let cos = angles.clone().cos().reshape([1, seq_len, 1, head_dim / 2]);
    let sin = angles.sin().reshape([1, seq_len, 1, head_dim / 2]);

    // Split q into pairs: (B, S, H, D/2) x 2
    let q_chunks = q.chunk(2, 3);
    let q1 = q_chunks[0].clone();
    let q2 = q_chunks[1].clone();

    let k_chunks = k.chunk(2, 3);
    let k1 = k_chunks[0].clone();
    let k2 = k_chunks[1].clone();

    // Apply rotation
    let q_rotated = Tensor::cat(vec![
        q1.clone() * cos.clone() - q2.clone() * sin.clone(),
        q1 * sin.clone() + q2 * cos.clone(),
    ], 3);

    let k_rotated = Tensor::cat(vec![
        k1.clone() * cos.clone() - k2.clone() * sin.clone(),
        k1 * sin + k2 * cos,
    ], 3);

    (q_rotated, k_rotated)
}

// ─── KV Repeat for GQA ─────────────────────────────────────────────────────

pub fn repeat_kv<B: Backend>(
    x: Tensor<B, 4>,
    num_heads: usize,
    num_kv_groups: usize,
) -> Tensor<B, 4> {
    if num_kv_groups == num_heads {
        return x;
    }

    let repeats = num_heads / num_kv_groups;
    let [batch, seq_len, _nkv, head_dim] = x.dims();

    let x = x.unsqueeze_dim::<5>(3);
    let x = x.repeat_dim(3, repeats);
    x.reshape([batch, seq_len, num_heads, head_dim])
}

// ─── Causal Mask ───────────────────────────────────────────────────────────

fn apply_causal_mask<B: Backend>(scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
    let device = scores.device();

    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = 1.0;
        }
    }
    let mask = Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, [seq_len, seq_len]),
            &device,
        )
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);

    let neg_inf = mask.clone() * (-1e9);
    let keep = (mask * (-1.0)) + 1.0;

    scores * keep + neg_inf
}

fn apply_causal_mask_with_offset<B: Backend>(
    scores: Tensor<B, 4>,
    q_len: usize,
    kv_len: usize,
) -> Tensor<B, 4> {
    let device = scores.device();
    let offset = kv_len - q_len;

    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_attend = offset + i;
        for j in (max_attend + 1)..kv_len {
            mask_data[i * kv_len + j] = 1.0;
        }
    }

    let mask = Tensor::<B, 2>::from_data(
        TensorData::new(mask_data, [q_len, kv_len]),
        &device,
    )
    .unsqueeze_dim::<3>(0)
    .unsqueeze_dim::<4>(0);

    let neg_inf = mask.clone() * (-1e9);
    let keep = (mask * (-1.0)) + 1.0;
    scores * keep + neg_inf
}

// ─── BitLinear Transformer Stack ────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearTransformerStack<B: Backend> {
    pub layers: Vec<BitLinearTransformerLayer<B>>,
    pub final_norm: BitLinearRMSNorm<B>,
    pub num_layers: usize,
    pub d_model: usize,
}

// ─── Language Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerBitLinearLM<B: Backend> {
    pub embedding: Embedding<B>,
    pub transformer: BitLinearTransformerStack<B>,
    pub head: BitLinear<B>,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
}

impl<B: Backend> TransformerBitLinearLM<B> {
    /// Standard forward (for training, no cache)
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer_forward(x, 0);
        self.head.forward(x)
    }

    /// Forward with KV cache
    pub fn forward_with_cache(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<Option<KVCache<B>>>,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer_forward_with_cache(x, offset, caches);
        (self.head.forward(x), new_caches)
    }

    fn transformer_forward(&self, mut x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        for layer in &self.transformer.layers {
            x = layer.forward(x, offset);
        }
        self.transformer.final_norm.forward(x)
    }

    fn transformer_forward_with_cache(
        &self,
        mut x: Tensor<B, 3>,
        offset: usize,
        caches: Vec<Option<KVCache<B>>>,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let mut new_caches = Vec::with_capacity(self.num_layers);

        for (layer, cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            let (out, new_cache) = layer.forward_with_cache(x, offset, cache);
            x = out;
            new_caches.push(new_cache);
        }

        (self.transformer.final_norm.forward(x), new_caches)
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

fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    previous_tokens: &[usize],
) -> usize {
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    if repetition_penalty != 1.0 {
        for (id, prob) in probs_vec.iter_mut() {
            if previous_tokens.contains(id) {
                *prob /= repetition_penalty;
            }
        }
    }

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs = Vec::with_capacity(k);
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

// ─── Text Generation ────────────────────────────────────────────────────────

fn generate_text_cached<B: Backend>(
    model: &TransformerBitLinearLM<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    caches: Vec<Option<KVCache<B>>>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<KVCache<B>>, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() { return (seed_text.to_string(), 0, 0.0, Vec::new(), current_offset); }

    let start_gen = Instant::now();

    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]),
        device,
    );

    let (logits, updated_caches) = model.forward_with_cache(input, current_offset, caches);
    let mut caches = updated_caches;

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits.slice([0..1, (s_len - 1)..s_len, 0..v_dim])
        .reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;

    // Trim rule
    if current_offset >= 255 {
        let remove_count = 160usize;
        if let Some(first) = caches.get(0) {
            let dims = first.cached_k.dims();
            let seq = dims[1];
            if seq > 70 {
                let remove = remove_count.min(seq);
                let keep = seq - remove;
                for c in caches.iter_mut() {
                    *c = c.keep_last(keep);
                }
                current_offset = current_offset.saturating_sub(remove);
                println!("(Cache trimmed: removed {} tokens; kept last {} tokens; new offset: {})", remove, keep, current_offset);
            }
        }
    }

    let mut next_id = sample_from_logits(
        last_logits, temperature, top_k, top_p, repetition_penalty, &history,
    );

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "eos" { break; }
        }

        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 { history.remove(0); }

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('▁', " ").replace(' ', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![next_id as i64], [1, 1]),
            device,
        );

        let cache_input: Vec<Option<KVCache<B>>> = caches.into_iter().map(|c| Some(c)).collect();
        let (logits, new_caches) = model.forward_with_cache(input, current_offset, cache_input);
        caches = new_caches;
        current_offset += 1;

        // Trim rule during generation
        if current_offset >= 255 {
            let remove_count = 160usize;
            if let Some(first) = caches.get(0) {
                let dims = first.cached_k.dims();
                let seq = dims[1];
                if seq > 70 {
                    let remove = remove_count.min(seq);
                    let keep = seq - remove;
                    for c in caches.iter_mut() {
                        *c = c.keep_last(keep);
                    }
                    current_offset = current_offset.saturating_sub(remove);
                    println!("(Cache trimmed: removed {} tokens; kept last {} tokens; new offset: {})", remove, keep, current_offset);
                }
            }
        }

        let [_, _, v] = logits.dims();
        let logits_2d = logits.reshape([1, v]);

        next_id = sample_from_logits(
            logits_2d, temperature, top_k, top_p, repetition_penalty, &history,
        );
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    (text, generated.len(), elapsed, caches, current_offset)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Transformer Bit2 — BitLinear (1.58-bit Ternary)           ║");
    println!("║     GQA + RoPE + SwiGLU + KV Cache                            ║");
    println!("║     Per-Group Quantization (GS=128) + STE                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 {
        args[1].clone()
    } else {
        "xorIA/input.txt".to_string()
    };

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
            println!("  (1) d_model: {}", d_model);
            println!("  (2) Num layers: {}", num_layers);
            println!("  (3) Heads:   {}", num_heads);
            println!("  (4) LR:      {}", lr);
            println!("  (5) Épocas:  {}", num_epochs);
            println!("  (6) Batch:   {}", batch_size);
            println!("  (7) Temp:    {}", temperature);
            println!("  (8) R-Pen:   {}", repetition_penalty);
            println!("----------------------------");
            print!("¿Entrenar (e), Inferir (i) o Ajustar parámetros (s)? [e/i/s]: ");
            io::stdout().flush()?;

            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();

            if choice == "i" {
                modo_inferencia = true;
                break;
            } else if choice == "e" {
                break;
            } else if choice == "s" {
                println!("\nAjustar parámetros (Enter para mantener actual):");

                print!("d_model [{}]: ", d_model);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { d_model = v; }

                print!("Num layers [{}]: ", num_layers);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { num_layers = v; }

                print!("Heads [{}]: ", num_heads);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { num_heads = v; }

                print!("Learning Rate [{}]: ", lr);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { lr = v; }

                print!("Épocas [{}]: ", num_epochs);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { num_epochs = v; }

                print!("Batch Size [{}]: ", batch_size);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { batch_size = v; }

                print!("Temperatura [{}]: ", temperature);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { temperature = v; }

                print!("Repetition Penalty [{}]: ", repetition_penalty);
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if let Ok(v) = input.trim().parse() { repetition_penalty = v; }
            }
        }
    }

    let device = Default::default();

    let num_kv_groups = 4; 

    println!("\n── Configuración del Transformer Bit2 ──");
    println!("  d_model:       {}", d_model);
    println!("  num_layers:    {}", num_layers);
    println!("  num_heads:     {} (query)", num_heads);
    println!("  num_kv_groups: {} (key/value)", num_kv_groups);
    println!("  heads/group:   {}", num_heads / num_kv_groups);
    println!("  head_dim:      {}", d_model / num_heads);
    println!("  FFN:           SwiGLU (BitLinear)");
    println!("  Positional:    RoPE");
    println!("  KV Cache:      Enabled");
    println!("  Quantization:  Ternary {{-1,0,+1}} (GS=128)\n");

    // Build BitLinear Transformer
    let head_dim = d_model / num_heads;
    let ffn_expansion = 4.0;
    let ffn_dim = ((ffn_expansion * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;

    let layers = (0..num_layers).map(|_| {
        let qkv = BitLinearQKVProjection {
            q_proj: BitLinearConfig {
                in_features: d_model,
                out_features: num_heads * head_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            k_proj: BitLinearConfig {
                in_features: d_model,
                out_features: num_kv_groups * head_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            v_proj: BitLinearConfig {
                in_features: d_model,
                out_features: num_kv_groups * head_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            num_heads,
            num_kv_groups,
            head_dim,
        };

        let o_proj = BitLinearOutputProjection {
            o_proj: BitLinearConfig {
                in_features: num_heads * head_dim,
                out_features: d_model,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            num_heads,
            head_dim,
        };

        let ffn = BitLinearSwiGLUFeedForward {
            gate_up_proj: BitLinearConfig {
                in_features: d_model,
                out_features: 2 * ffn_dim,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            down_proj: BitLinearConfig {
                in_features: ffn_dim,
                out_features: d_model,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }.init(&device),
            dropout: burn::nn::DropoutConfig::new(0.1).init(),
            intermediate_dim: ffn_dim,
        };

        BitLinearTransformerLayer {
            attn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
            qkv,
            o_proj,
            ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
            ffn,
            residual_dropout: burn::nn::DropoutConfig::new(0.1).init(),
        }
    }).collect();

    let transformer = BitLinearTransformerStack {
        final_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
        num_layers,
        d_model,
        layers,
    };

    let mut model: TransformerBitLinearLM<MyBackend> = TransformerBitLinearLM {
        embedding: EmbeddingConfig::new(vocab_size, d_model).init(&device),
        transformer,
        head: BitLinearConfig {
            in_features: d_model,
            out_features: vocab_size,
            bias: false,
            activation_bits: 8,
            rms_norm_eps: 1e-5,
        }.init(&device),
        vocab_size,
        d_model,
        num_layers,
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
        println!("║     MODO INTERACTIVO — Transformer Bit2 (BitLinear)           ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Comandos:");
        println!("  - Escribe tu semilla para generar texto.");
        println!("  - 'len <n>':    Cambia la cantidad de tokens.");
        println!("  - 'temp <f>':   Cambia la temperatura.");
        println!("  - 'topk <n>':   Cambia el Top-K.");
        println!("  - 'topp <f>':   Cambia el Top-P.");
        println!("  - 'rpen <f>':   Cambia el Repetition Penalty.");
        println!("  - 'salir' o 'exit' para terminar.\n");

        let mut current_len = 50;
        let mut session_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
        let mut session_offset = 0;

        loop {
            print!("Chat [len:{} t:{} k:{} p:{} rp:{}] > ", 
                current_len, temperature, top_k, top_p, repetition_penalty);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
                break;
            }

            if input.to_lowercase().starts_with("len ") {
                if let Ok(v) = input[4..].trim().parse::<usize>() {
                    current_len = v;
                    println!("  -> Longitud: {} tokens.\n", current_len);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("temp ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    temperature = v;
                    println!("  -> Temperatura: {}.\n", temperature);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("topk ") {
                if let Ok(v) = input[5..].trim().parse::<usize>() {
                    top_k = v;
                    println!("  -> Top-K: {}.\n", top_k);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("topp ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    top_p = v;
                    println!("  -> Top-P: {}.\n", top_p);
                    continue;
                }
            }
            if input.to_lowercase().starts_with("rpen ") {
                if let Ok(v) = input[5..].trim().parse::<f32>() {
                    repetition_penalty = v;
                    println!("  -> Repetition Penalty: {}.\n", repetition_penalty);
                    continue;
                }
            }
            if input.eq_ignore_ascii_case("reset") {
                session_caches = (0..num_layers).map(|_| None).collect();
                session_offset = 0;
                println!("  -> Memoria de sesión reiniciada.\n");
                continue;
            }

            if input.is_empty() { continue; }

            println!("\n--- TEXTO GENERADO ---");
            let (_text, tokens_count, elapsed, updated_caches, updated_offset) = generate_text_cached(
                &model.valid(), &tokenizer, input, current_len, &device,
                temperature, top_k, top_p, repetition_penalty,
                session_caches, session_offset,
            );
            session_caches = updated_caches.into_iter().map(Some).collect();
            session_offset = updated_offset;

            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("---");
            println!("Tokens: {} | Tiempo: {:.2}s | Velocidad: {:.2} tok/s | Offset Total: {}\n",
                tokens_count, elapsed, tps, session_offset);
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let seq_len = 64;
    let stride = 64;

    let text_path = Path::new(&text_file);

    println!("Iniciando entrenamiento con streaming...");
    println!("  batch_size: {} | seq_len: {} | stride: {}\n", batch_size, seq_len, stride);

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

                if current_loss.is_nan() {
                    println!("\n[!] Loss NaN en Fragmento {} Batch {}. Abortando.", frag_idx, b);
                    return Ok(());
                }

                total_loss += current_loss;
                batch_count += 1;

                let grads = loss.backward();
                let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads_p);

                let elapsed = start_epoch.elapsed().as_secs_f32();
                let tps = (batch_count * batch_size * seq_len) as f32 / elapsed;
                print!("\rEpoch {} | Frag {} | Batch {}/{} | Loss: {:.4} | {:.1} tok/s",
                    epoch + 1, frag_idx, b + 1, num_batches,
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
            println!("--- Generación de prueba ---");
            let empty_caches: Vec<Option<KVCache<Flex<f32>>>> = (0..num_layers).map(|_| None).collect();
            let (_, tokens_count, elapsed, _, _) = generate_text_cached(
                &model.clone().valid(), &tokenizer, "The world ", 30, &device,
                temperature, top_k, top_p, repetition_penalty,
                empty_caches, 0,
            );
            let tps = tokens_count as f32 / elapsed.max(0.001);
            println!("[{:.1} tok/s]\n---------------------------", tps);
        }
    }

    Ok(())
}
