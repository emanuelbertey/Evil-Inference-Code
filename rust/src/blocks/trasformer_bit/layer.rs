// ─── BitLinear Transformer Layer ────────────────────────────────────────────

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use crate::blocks::bitlinear::layer::BitLinearInferenceState;

use super::ops::{apply_rope, apply_rope_fused, repeat_kv, apply_causal_mask, apply_causal_mask_with_offset};
use super::projections::{BitLinearQKVProjection, BitLinearOutputProjection, BitLinearSwiGLUFeedForward, BitLinearRMSNorm};
use super::cache::KVCache;
use super::cache::KuantKVCache;

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

    pub fn forward_with_kuant_cache_inference(
        &self,
        x: Tensor<B, 3>,
        offset: usize,
        cache: KuantKVCache,
        states: (
            &BitLinearInferenceState,
            &BitLinearInferenceState,
            &BitLinearInferenceState,
            &BitLinearInferenceState,
            &BitLinearInferenceState,
            &BitLinearInferenceState,
        ),
    ) -> (Tensor<B, 3>, KuantKVCache) {
        let residual = x.clone();
        let (h, new_cache) = self.attention_with_kuant_cache_inference(
            self.attn_norm.forward(x),
            offset,
            cache,
            states.0,
            states.1,
            states.2,
            states.3,
        );
        let x = residual + self.residual_dropout.forward(h);

        let residual = x.clone();
        let h = self.ffn.forward_inference(self.ffn_norm.forward(x), states.4, states.5);
        (residual + self.residual_dropout.forward(h), new_cache)
    }

    fn attention_with_kuant_cache_inference(
        &self,
        x: Tensor<B, 3>,
        offset: usize,
        mut cache: KuantKVCache,
        q_state: &BitLinearInferenceState,
        k_state: &BitLinearInferenceState,
        v_state: &BitLinearInferenceState,
        o_state: &BitLinearInferenceState,
    ) -> (Tensor<B, 3>, KuantKVCache) {
        let (q, k_new, v_new) = self.qkv.forward_inference(x, q_state, k_state, v_state);
        let (q, k_new) = apply_rope_fused(q, k_new, offset);
        let device = q.device();
        let old_len = cache.current_len;

        cache.append(k_new, v_new);
        let attn_output = cache.attend(q, old_len, self.qkv.num_heads, &device);
        (self.o_proj.forward_inference(attn_output, o_state), cache)
    }
}
