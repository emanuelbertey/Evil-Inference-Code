// ─── BitLinear Transformer Model ────────────────────────────────────────────
// Shared model structs for transformer_bit2 (CPU & CUDA).

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend, TensorData, Int};
use crate::blocks::bitlinear::kernel::KernelKind;
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

use crate::blocks::bitlinear::layer::{BitLinear, BitLinearInferenceState};

// ─── Cached Inference State ────────────────────────────────────────────────
pub struct TransformerInferenceState {
    pub qkv: Vec<(BitLinearInferenceState, BitLinearInferenceState, BitLinearInferenceState)>,
    pub o_proj: Vec<BitLinearInferenceState>,
    pub ffn_gate_up: Vec<BitLinearInferenceState>,
    pub ffn_down: Vec<BitLinearInferenceState>,
    pub head: BitLinearInferenceState,
}

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
    pub fn release_weights(&mut self, device: &B::Device) {
        self.q_proj.release_weights(device);
        self.k_proj.release_weights(device);
        self.v_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();
        let q = self.q_proj.forward(x.clone()).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward(x).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        (q, k, v)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, q_state: &BitLinearInferenceState, k_state: &BitLinearInferenceState, v_state: &BitLinearInferenceState) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();
        let q = self.q_proj.forward_inference(x.clone(), q_state).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward_inference(x.clone(), k_state).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward_inference(x, v_state).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
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
    pub fn release_weights(&mut self, device: &B::Device) {
        self.o_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        self.o_proj.forward(x.reshape([batch, seq_len, self.num_heads * self.head_dim]))
    }

    pub fn forward_inference(&self, x: Tensor<B, 4>, state: &BitLinearInferenceState) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        self.o_proj.forward_inference(x.reshape([batch, seq_len, self.num_heads * self.head_dim]), state)
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
    pub fn release_weights(&mut self, device: &B::Device) {
        self.gate_up_proj.release_weights(device);
        self.down_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward(x);
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, gate_up_state: &BitLinearInferenceState, down_state: &BitLinearInferenceState) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward_inference(x, gate_up_state);
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        self.down_proj.forward_inference(h, down_state)
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

// ─── KV Cache (Vec+cat chunks) ─────────────────────────────────────────────

pub const MAX_CACHE_LEN: usize = 4096;

#[derive(Debug)]
pub struct KVCache<B: Backend> {
    k_chunks: Vec<Tensor<B, 4>>,
    v_chunks: Vec<Tensor<B, 4>>,
    pub current_len: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new(_num_kv_groups: usize, _head_dim: usize, _device: &B::Device) -> Self {
        KVCache { k_chunks: Vec::new(), v_chunks: Vec::new(), current_len: 0 }
    }

    pub fn append(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) {
        let add_len = k_new.dims()[1];
        let new_len = self.current_len + add_len;

        if new_len > MAX_CACHE_LEN {
            let safe_keep = MAX_CACHE_LEN.saturating_sub(add_len);
            self.keep_last(safe_keep);
        }

        self.k_chunks.push(k_new);
        self.v_chunks.push(v_new);
        self.current_len = self.k_chunks.iter().map(|t| t.dims()[1]).sum();
    }

    pub fn view(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.k_chunks.len() == 1 {
            (self.k_chunks[0].clone(), self.v_chunks[0].clone())
        } else {
            (Tensor::cat(self.k_chunks.iter().map(|t| t.clone()).collect(), 1),
             Tensor::cat(self.v_chunks.iter().map(|t| t.clone()).collect(), 1))
        }
    }

    pub fn keep_last(&mut self, keep: usize) {
        if keep >= self.current_len { return; }
        let mut remaining = keep;
        let mut new_k = Vec::new();
        let mut new_v = Vec::new();
        let mut rev_k = Vec::new();
        let mut rev_v = Vec::new();
        for (k, v) in self.k_chunks.drain(..).rev().zip(self.v_chunks.drain(..).rev()) {
            let seq = k.dims()[1];
            if remaining >= seq {
                rev_k.push(k);
                rev_v.push(v);
                remaining -= seq;
            } else {
                rev_k.push(k.narrow(1, seq - remaining, remaining));
                rev_v.push(v.narrow(1, seq - remaining, remaining));
                break;
            }
        }
        for (k, v) in rev_k.into_iter().rev().zip(rev_v.into_iter().rev()) {
            new_k.push(k);
            new_v.push(v);
        }
        self.k_chunks = new_k;
        self.v_chunks = new_v;
        self.current_len = keep;
    }
}

// ─── Fused RoPE (raw slice, no cache, no Tensor ops) ───────────────────────

pub fn apply_rope_fused<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq_len, nheads, head_dim] = q.dims();
    let nkv = k.dims()[2];
    let hh = head_dim / 2;
    let device = q.device();

    let q_data = q.into_data();
    let k_data = k.into_data();
    let mut q_slice = q_data.as_slice::<f32>().unwrap().to_vec();
    let mut k_slice = k_data.as_slice::<f32>().unwrap().to_vec();

    let theta: Vec<f32> = (0..hh)
        .map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    for b in 0..batch {
        for s in 0..seq_len {
            let pos = (offset + s) as f32;
            for h in 0..nheads {
                let base_q = ((b * seq_len) + s) * nheads * head_dim + h * head_dim;
                for i in 0..hh {
                    let cos = (pos * theta[i]).cos();
                    let sin = (pos * theta[i]).sin();
                    let q1 = q_slice[base_q + i];
                    let q2 = q_slice[base_q + i + hh];
                    q_slice[base_q + i] = q1 * cos - q2 * sin;
                    q_slice[base_q + i + hh] = q1 * sin + q2 * cos;
                }
            }
            for h in 0..nkv {
                let base_k = ((b * seq_len) + s) * nkv * head_dim + h * head_dim;
                for i in 0..hh {
                    let cos = (pos * theta[i]).cos();
                    let sin = (pos * theta[i]).sin();
                    let k1 = k_slice[base_k + i];
                    let k2 = k_slice[base_k + i + hh];
                    k_slice[base_k + i] = k1 * cos - k2 * sin;
                    k_slice[base_k + i + hh] = k1 * sin + k2 * cos;
                }
            }
        }
    }

    let q_out = Tensor::<B, 4>::from_data(
        TensorData::new(q_slice, [batch, seq_len, nheads, head_dim]), &device);
    let k_out = Tensor::<B, 4>::from_data(
        TensorData::new(k_slice, [batch, seq_len, nkv, head_dim]), &device);
    (q_out, k_out)
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
    pub fn release_weights(&mut self, device: &B::Device) {
        self.qkv.release_weights(device);
        self.o_proj.release_weights(device);
        self.ffn.release_weights(device);
    }

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
        let (q, k) = apply_rope_fused(q, k, offset);
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
        let (q, k_new) = apply_rope_fused(q, k_new, offset);

        let mut kv_cache = cache.unwrap_or_else(|| KVCache::new(k_new.dims()[2], k_new.dims()[3], &k_new.device()));
        kv_cache.append(k_new, v_new);
        let (k_full, v_full) = kv_cache.view();
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
        (self.o_proj.forward(attn_output), kv_cache)
    }

    pub fn forward_with_cache_inference(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>, states: (&BitLinearInferenceState, &BitLinearInferenceState, &BitLinearInferenceState, &BitLinearInferenceState, &BitLinearInferenceState, &BitLinearInferenceState)) -> (Tensor<B, 3>, KVCache<B>) {
        let residual = x.clone();
        let (h, new_cache) = self.attention_with_cache_inference(self.attn_norm.forward(x), offset, cache, states.0, states.1, states.2, states.3);
        let x = residual + self.residual_dropout.forward(h);

        let residual = x.clone();
        let h = self.ffn.forward_inference(self.ffn_norm.forward(x), states.4, states.5);
        (residual + self.residual_dropout.forward(h), new_cache)
    }

    fn attention_with_cache_inference(&self, x: Tensor<B, 3>, offset: usize, cache: Option<KVCache<B>>, q_state: &BitLinearInferenceState, k_state: &BitLinearInferenceState, v_state: &BitLinearInferenceState, o_state: &BitLinearInferenceState) -> (Tensor<B, 3>, KVCache<B>) {
        let (q, k_new, v_new) = self.qkv.forward_inference(x, q_state, k_state, v_state);
        let (q, k_new) = apply_rope_fused(q, k_new, offset);

        let mut kv_cache = cache.unwrap_or_else(|| KVCache::new(k_new.dims()[2], k_new.dims()[3], &k_new.device()));
        kv_cache.append(k_new, v_new);
        let (k_full, v_full) = kv_cache.view();
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
        (self.o_proj.forward_inference(attn_output, o_state), kv_cache)
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
    pub fn release_all_weights(&mut self, device: &B::Device) {
        for layer in &mut self.transformer.layers {
            layer.release_weights(device);
        }
        self.head.release_weights(device);
    }

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

    pub fn build_inference_state(&self, device: &B::Device, layer_kernel: KernelKind, head_kernel: KernelKind) -> TransformerInferenceState {
        let mut qkv_states = Vec::new();
        let mut o_proj_states = Vec::new();
        let mut ffn_gate_up_states = Vec::new();
        let mut ffn_down_states = Vec::new();

        for layer in &self.transformer.layers {
            qkv_states.push((
                layer.qkv.q_proj.export_inference_layer(device, layer_kernel),
                layer.qkv.k_proj.export_inference_layer(device, layer_kernel),
                layer.qkv.v_proj.export_inference_layer(device, layer_kernel),
            ));
            o_proj_states.push(layer.o_proj.o_proj.export_inference_layer(device, layer_kernel));
            ffn_gate_up_states.push(layer.ffn.gate_up_proj.export_inference_layer(device, layer_kernel));
            ffn_down_states.push(layer.ffn.down_proj.export_inference_layer(device, layer_kernel));
        }

        TransformerInferenceState {
            qkv: qkv_states,
            o_proj: o_proj_states,
            ffn_gate_up: ffn_gate_up_states,
            ffn_down: ffn_down_states,
            head: self.head.export_inference_layer(device, head_kernel),
        }
    }

    pub fn forward_with_cache_inference(&self, input: Tensor<B, 2, Int>, offset: usize, caches: Vec<Option<KVCache<B>>>, state: &TransformerInferenceState) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let device = input.device();
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer_forward_with_cache_inference(x, offset, caches, state);
        let x_flat = x;
        let [batch, seq, d] = x_flat.dims();
        let x_2d = x_flat.reshape([batch * seq, d]);
        let x_data = x_2d.into_data();
        let x_slice = x_data.as_slice::<f32>().unwrap();
        let out_data = state.head.forward_raw(x_slice, batch * seq);
        let output = Tensor::<B, 2>::from_data(TensorData::new(out_data, [batch * seq, self.vocab_size]), &device);
        (output.reshape([batch, seq, self.vocab_size]), new_caches)
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

    fn transformer_forward_with_cache_inference(&self, mut x: Tensor<B, 3>, offset: usize, caches: Vec<Option<KVCache<B>>>, state: &TransformerInferenceState) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (idx, (layer, cache)) in self.transformer.layers.iter().zip(caches.into_iter()).enumerate() {
            let layer_states = (&state.qkv[idx].0, &state.qkv[idx].1, &state.qkv[idx].2, &state.o_proj[idx], &state.ffn_gate_up[idx], &state.ffn_down[idx]);
            let (out, new_cache) = layer.forward_with_cache_inference(x, offset, cache, layer_states);
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
