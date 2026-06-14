// ─── BitLinear Transformer Model ────────────────────────────────────────────
// Shared model structs for transformer_bit2 (CPU & CUDA).

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend, TensorData, Int};
use std::error::Error;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::Path;

use tokenizers::AddedToken;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::blocks::bitlinear::layer::BitLinear;

// ─── BPE Tokenizer ──────────────────────────────────────────────────────────

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let model = BPE::builder().byte_fallback(true).build().map_err(|e| format!("BPE error: {}", e))?;
        let mut tok = HFTokenizer::new(model);
        tok.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', PrependScheme::Always, false)));
        tok.with_decoder(Some(MetaspaceDecoder::new('\u{2581}', PrependScheme::Always, false)));
        let special = "eos";
        tok.add_special_tokens(&[AddedToken::from(special, true)]);
        let trainer = BpeTrainerBuilder::default().show_progress(true).vocab_size(vocab_size).min_frequency(2)
            .special_tokens(vec![AddedToken::from(special, true)]).build();
        let mut tw = TrainerWrapper::from(trainer);
        let tmp = "temp_train_bit2.txt";
        fs::write(tmp, text)?;
        tok.train_from_files(&mut tw, vec![tmp.to_string()]).map_err(|e| format!("Tokenizer: {}", e))?;
        fs::remove_file(tmp)?;
        Ok(Self { tokenizer: tok })
    }
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> { self.tokenizer.save(path, true).map_err(|e| -> Box<dyn Error> { format!("{}", e).into() }) }
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tok = HFTokenizer::from_file(path).map_err(|e| -> Box<dyn Error> { format!("{}", e).into() })?;
        tok.with_decoder(Some(MetaspaceDecoder::new('\u{2581}', PrependScheme::Always, false)));
        Ok(Self { tokenizer: tok })
    }
    pub fn encode(&self, text: &str) -> Vec<usize> { self.tokenizer.encode(text, false).unwrap().get_ids().iter().map(|&id| id as usize).collect() }
    pub fn decode(&self, indices: &[usize]) -> String { self.tokenizer.decode(&indices.iter().map(|&i| i as u32).collect::<Vec<_>>(), true).unwrap() }
    pub fn vocab_size(&self) -> usize { self.tokenizer.get_vocab_size(true) }
    pub fn id_to_token(&self, id: usize) -> Option<String> { self.tokenizer.id_to_token(id as u32) }
}

// ─── File Fragment Iterator (Streaming) ─────────────────────────────────────

pub struct FileFragmentIterator {
    reader: BufReader<fs::File>,
    buffer_size: usize,
    finished: bool,
}

impl FileFragmentIterator {
    pub fn new(path: &Path, buffer_size_mb: usize) -> io::Result<Self> {
        Ok(Self { reader: BufReader::new(fs::File::open(path)?), buffer_size: buffer_size_mb * 1024 * 1024, finished: false })
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
        while !buffer.is_empty() && String::from_utf8(buffer.clone()).is_err() { buffer.pop(); }
        if buffer.is_empty() { return None; }
        String::from_utf8(buffer).ok()
    }
}

// ─── BitLinear QKV Projection ──────────────────────────────────────────────

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
        let q = self.q_proj.forward(x.clone()).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward(x).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        (q, k, v)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();
        let q = self.q_proj.forward_inference(x.clone(), device).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward_inference(x.clone(), device).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward_inference(x, device).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
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
        self.o_proj.forward(x.reshape([batch, seq_len, self.num_heads * self.head_dim]))
    }

    pub fn forward_inference(&self, x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        self.o_proj.forward_inference(x.reshape([batch, seq_len, self.num_heads * self.head_dim]), device)
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
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward_inference(x, device);
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        self.down_proj.forward_inference(h, device)
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
        let rms = x.clone().powf_scalar(2.0).mean_dim(2).sqrt().clamp_min(self.eps as f32);
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
        if keep == 0 { return self.clone(); }
        let keep = keep.min(seq);
        if keep == seq { return self.clone(); }
        let start = seq - keep;
        let k = self.cached_k.clone().slice([0..b, start..seq, 0..g, 0..d]);
        let v = self.cached_v.clone().slice([0..b, start..seq, 0..g, 0..d]);
        KVCache { cached_k: k, cached_v: v }
    }
}

// ─── RoPE ──────────────────────────────────────────────────────────────────

pub fn apply_rope<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, offset: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_batch, seq_len, _num_heads, head_dim] = q.dims();

    let theta: Vec<f32> = (0..head_dim / 2)
        .map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let theta_tensor = Tensor::<B, 1>::from_data(TensorData::new(theta, [head_dim / 2]), &q.device());
    let positions: Vec<f32> = (offset..offset + seq_len).map(|p| p as f32).collect();
    let pos_tensor = Tensor::<B, 1>::from_data(TensorData::new(positions, [seq_len]), &q.device());

    let angles = pos_tensor.reshape([seq_len, 1]) * theta_tensor.reshape([1, head_dim / 2]);
    let cos = angles.clone().cos().reshape([1, seq_len, 1, head_dim / 2]);
    let sin = angles.sin().reshape([1, seq_len, 1, head_dim / 2]);

    let q_chunks = q.chunk(2, 3);
    let (q1, q2) = (q_chunks[0].clone(), q_chunks[1].clone());
    let k_chunks = k.chunk(2, 3);
    let (k1, k2) = (k_chunks[0].clone(), k_chunks[1].clone());

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

pub fn repeat_kv<B: Backend>(x: Tensor<B, 4>, num_heads: usize, num_kv_groups: usize) -> Tensor<B, 4> {
    if num_kv_groups == num_heads { return x; }
    let repeats = num_heads / num_kv_groups;
    let [batch, seq_len, _nkv, head_dim] = x.dims();
    let x = x.unsqueeze_dim::<5>(3).repeat_dim(3, repeats);
    x.reshape([batch, seq_len, num_heads, head_dim])
}

// ─── Causal Mask ───────────────────────────────────────────────────────────

pub fn apply_causal_mask<B: Backend>(scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = 1.0;
        }
    }
    let mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [seq_len, seq_len]), &device)
        .unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let neg_inf = mask.clone() * (-1e9);
    let keep = (mask * (-1.0)) + 1.0;
    scores * keep + neg_inf
}

pub fn apply_causal_mask_with_offset<B: Backend>(scores: Tensor<B, 4>, q_len: usize, kv_len: usize) -> Tensor<B, 4> {
    let device = scores.device();
    let offset = kv_len - q_len;
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_attend = offset + i;
        for j in (max_attend + 1)..kv_len {
            mask_data[i * kv_len + j] = 1.0;
        }
    }
    let mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [q_len, kv_len]), &device)
        .unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let neg_inf = mask.clone() * (-1e9);
    let keep = (mask * (-1.0)) + 1.0;
    scores * keep + neg_inf
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
        let residual = x.clone();
        let h = self.attention_forward(self.attn_norm.forward(x), offset);
        let x = residual + self.residual_dropout.forward(h);

        let residual = x.clone();
        let h = self.ffn.forward(self.ffn_norm.forward(x));
        residual + self.residual_dropout.forward(h)
    }

    fn attention_forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_batch, seq_len, _d] = x.dims();
        let (q, k, v) = self.qkv.forward(x);
        let (q, k) = apply_rope(q, k, offset);
        let k = repeat_kv(k, self.qkv.num_heads, self.qkv.num_kv_groups);
        let v = repeat_kv(v, self.qkv.num_heads, self.qkv.num_kv_groups);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (self.qkv.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        if seq_len > 1 { scores = apply_causal_mask(scores, seq_len); }
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        let attn_output = attn_weights.matmul(v).swap_dims(1, 2);
        self.o_proj.forward(attn_output)
    }

    pub fn forward_with_cache(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>) -> (Tensor<B, 3>, KVCache<B>) {
        let residual = x.clone();
        let (h, new_cache) = self.attention_with_cache(self.attn_norm.forward(x), offset, cache);
        let x = residual + self.residual_dropout.forward(h);

        let residual = x.clone();
        let h = self.ffn.forward(self.ffn_norm.forward(x));
        (residual + self.residual_dropout.forward(h), new_cache)
    }

    fn attention_with_cache(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>) -> (Tensor<B, 3>, KVCache<B>) {
        let (q, k_new, v_new) = self.qkv.forward(x);
        let (q, k_new) = apply_rope(q, k_new, offset);

        let (k_full, v_full) = if let Some(prev) = cache {
            (Tensor::cat(vec![prev.cached_k, k_new.clone()], 1), Tensor::cat(vec![prev.cached_v, v_new.clone()], 1))
        } else {
            (k_new.clone(), v_new.clone())
        };

        let new_cache = KVCache { cached_k: k_full.clone(), cached_v: v_full.clone() };
        let k_exp = repeat_kv(k_full, self.qkv.num_heads, self.qkv.num_kv_groups);
        let v_exp = repeat_kv(v_full, self.qkv.num_heads, self.qkv.num_kv_groups);

        let q = q.swap_dims(1, 2);
        let k = k_exp.swap_dims(1, 2);
        let v = v_exp.swap_dims(1, 2);

        let scale = (self.qkv.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        let [_, _, q_len, kv_len] = scores.dims();
        if q_len > 1 { scores = apply_causal_mask_with_offset(scores, q_len, kv_len); }

        let attn_output = burn::tensor::activation::softmax(scores, 3).matmul(v).swap_dims(1, 2);
        (self.o_proj.forward(attn_output), new_cache)
    }

    pub fn forward_with_cache_inference(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>, device: &B::Device) -> (Tensor<B, 3>, KVCache<B>) {
        let residual = x.clone();
        let (h, new_cache) = self.attention_with_cache_inference(self.attn_norm.forward(x), offset, cache, device);
        let x = residual + self.residual_dropout.forward(h);

        let residual = x.clone();
        let h = self.ffn.forward_inference(self.ffn_norm.forward(x), device);
        (residual + self.residual_dropout.forward(h), new_cache)
    }

    fn attention_with_cache_inference(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>, device: &B::Device) -> (Tensor<B, 3>, KVCache<B>) {
        let (q, k_new, v_new) = self.qkv.forward_inference(x, device);
        let (q, k_new) = apply_rope(q, k_new, offset);

        let (k_full, v_full) = if let Some(prev) = cache {
            (Tensor::cat(vec![prev.cached_k, k_new.clone()], 1), Tensor::cat(vec![prev.cached_v, v_new.clone()], 1))
        } else {
            (k_new.clone(), v_new.clone())
        };

        let new_cache = KVCache { cached_k: k_full.clone(), cached_v: v_full.clone() };
        let k_exp = repeat_kv(k_full, self.qkv.num_heads, self.qkv.num_kv_groups);
        let v_exp = repeat_kv(v_full, self.qkv.num_heads, self.qkv.num_kv_groups);

        let q = q.swap_dims(1, 2);
        let k = k_exp.swap_dims(1, 2);
        let v = v_exp.swap_dims(1, 2);

        let scale = (self.qkv.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        let [_, _, q_len, kv_len] = scores.dims();
        if q_len > 1 { scores = apply_causal_mask_with_offset(scores, q_len, kv_len); }

        let attn_output = burn::tensor::activation::softmax(scores, 3).matmul(v).swap_dims(1, 2);
        (self.o_proj.forward_inference(attn_output, device), new_cache)
    }
}

// ─── Transformer Stack ─────────────────────────────────────────────────────

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
    pub embedding: burn::nn::Embedding<B>,
    pub transformer: BitLinearTransformerStack<B>,
    pub head: BitLinear<B>,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
}

impl<B: Backend> TransformerBitLinearLM<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let x = self.transformer_forward(x, 0);
        self.head.forward(x)
    }

    pub fn forward_with_cache(&self, input: Tensor<B, 2, Int>, offset: usize, caches: Vec<Option<KVCache<B>>>) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer_forward_with_cache(x, offset, caches);
        (self.head.forward(x), new_caches)
    }

    pub fn forward_with_cache_inference(&self, input: Tensor<B, 2, Int>, offset: usize, caches: Vec<Option<KVCache<B>>>, device: &B::Device) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer_forward_with_cache_inference(x, offset, caches, device);
        (self.head.forward_inference(x, device), new_caches)
    }

    fn transformer_forward(&self, mut x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        for layer in &self.transformer.layers { x = layer.forward(x, offset); }
        self.transformer.final_norm.forward(x)
    }

    fn transformer_forward_with_cache(&self, mut x: Tensor<B, 3>, offset: usize, caches: Vec<Option<KVCache<B>>>) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (layer, cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            let (out, new_cache) = layer.forward_with_cache(x, offset, cache);
            x = out;
            new_caches.push(new_cache);
        }
        (self.transformer.final_norm.forward(x), new_caches)
    }

    fn transformer_forward_with_cache_inference(&self, mut x: Tensor<B, 3>, offset: usize, caches: Vec<Option<KVCache<B>>>, device: &B::Device) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (layer, cache) in self.transformer.layers.iter().zip(caches.into_iter()) {
            let (out, new_cache) = layer.forward_with_cache_inference(x, offset, cache, device);
            x = out;
            new_caches.push(new_cache);
        }
        (self.transformer.final_norm.forward(x), new_caches)
    }
}

// ─── Shared Utilities ──────────────────────────────────────────────────────

pub fn create_batch<B: Backend>(tokens: &[usize], start_idx: usize, batch_size: usize, seq_length: usize, stride: usize, device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);
    for i in 0..batch_size {
        let s = start_idx + i * stride;
        for j in 0..seq_length {
            x_indices.push(tokens[s + j] as i64);
            y_indices.push(tokens[s + j + 1] as i64);
        }
    }
    (Tensor::<B, 2, Int>::from_data(TensorData::new(x_indices, [batch_size, seq_length]), device),
     Tensor::<B, 2, Int>::from_data(TensorData::new(y_indices, [batch_size, seq_length]), device))
}

pub fn sample_from_logits<B: Backend>(logits: Tensor<B, 2>, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, previous_tokens: &[usize]) -> usize {
    use burn::tensor::activation::softmax;
    use rand::Rng;
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.into_data().as_slice::<f32>().unwrap().iter().enumerate().map(|(i, &x)| (i, x)).collect();

    if repetition_penalty != 1.0 {
        for (id, prob) in probs_vec.iter_mut() {
            if previous_tokens.contains(id) { *prob /= repetition_penalty; }
        }
    }

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered = Vec::with_capacity(k);
    let mut cum = 0.0;
    for (i, p) in probs_vec { filtered.push((i, p)); cum += p; if filtered.len() >= k || cum >= top_p { break; } }

    let indices: Vec<usize> = filtered.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = filtered.iter().map(|(_, p)| *p).collect();

    if temperature <= 1e-6 { return indices[0]; }
    for p in weights.iter_mut() { *p = (p.max(1e-10).ln() / temperature).exp(); }
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng();
    let sample: f32 = rng.random::<f32>() * sum;
    let mut cum = 0.0;
    for (i, &p) in weights.iter().enumerate() { cum += p; if sample <= cum { return indices[i]; } }
    indices[0]
}
