// ─── BitLinear Transformer Model ────────────────────────────────────────────
// Shared model structs for transformer_bit2 (CPU & CUDA).
// Re-exports all sub-modules for backward compatibility.

pub use super::ops::*;
pub use super::cache::*;
pub use super::projections::*;
pub use super::layer::*;

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

    pub fn build_kuant_caches(&self, bits: usize, seed: u64) -> Vec<KuantKVCache> {
        self.transformer
            .layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                KuantKVCache::new(
                    layer.qkv.num_kv_groups,
                    layer.qkv.head_dim,
                    bits,
                    seed.wrapping_add(idx as u64),
                )
            })
            .collect()
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

    pub fn forward_with_kuant_cache_inference(
        &self,
        input: Tensor<B, 2, Int>,
        offset: usize,
        caches: Vec<KuantKVCache>,
        state: &TransformerInferenceState,
    ) -> (Tensor<B, 3>, Vec<KuantKVCache>) {
        let device = input.device();
        let x = self.embedding.forward(input);
        let (x, new_caches) = self.transformer_forward_with_kuant_cache_inference(x, offset, caches, state);
        let [batch, seq, d] = x.dims();
        let x_2d = x.reshape([batch * seq, d]);
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

    fn transformer_forward_with_kuant_cache_inference(
        &self,
        mut x: Tensor<B, 3>,
        offset: usize,
        caches: Vec<KuantKVCache>,
        state: &TransformerInferenceState,
    ) -> (Tensor<B, 3>, Vec<KuantKVCache>) {
        let mut new_caches = Vec::with_capacity(self.num_layers);
        for (idx, (layer, cache)) in self.transformer.layers.iter().zip(caches.into_iter()).enumerate() {
            let layer_states = (
                &state.qkv[idx].0,
                &state.qkv[idx].1,
                &state.qkv[idx].2,
                &state.o_proj[idx],
                &state.ffn_gate_up[idx],
                &state.ffn_down[idx],
            );
            let (out, new_cache) = layer.forward_with_kuant_cache_inference(x, offset, cache, layer_states);
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
