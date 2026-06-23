// ─── KV Cache (Vec+cat chunks) ─────────────────────────────────────────────

use burn::tensor::{Tensor, backend::Backend, TensorData};
use crate::blocks::turbokuant::TurboQuant;
use super::ops::softmax_vec;

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
            if remaining == 0 { break; }
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

// ─── KuantKVCache (TurboQuant compressed) ──────────────────────────────────

#[derive(Debug)]
pub struct KuantKVCache {
    tq: TurboQuant,
    num_kv_groups: usize,
    head_dim: usize,
    key_stride: usize,
    value_stride: usize,
    keys: Vec<Vec<u8>>,
    values: Vec<Vec<u8>>,
    pub current_len: usize,
}

impl KuantKVCache {
    pub fn new(num_kv_groups: usize, head_dim: usize, bits: usize, seed: u64) -> Self {
        let tq = TurboQuant::new(head_dim, bits, seed).expect("invalid TurboQuant cache shape");
        let key_stride = tq.key_bytes();
        let value_stride = tq.value_bytes();
        Self {
            tq,
            num_kv_groups,
            head_dim,
            key_stride,
            value_stride,
            keys: vec![Vec::new(); num_kv_groups],
            values: vec![Vec::new(); num_kv_groups],
            current_len: 0,
        }
    }

    pub fn append<B: Backend>(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) {
        let [batch, seq_len, num_kv_groups, head_dim] = k_new.dims();
        assert_eq!(batch, 1, "KuantKVCache currently supports batch=1 inference");
        assert_eq!(num_kv_groups, self.num_kv_groups);
        assert_eq!(head_dim, self.head_dim);

        let k_data = k_new.into_data();
        let v_data = v_new.into_data();
        let k = k_data.as_slice::<f32>().unwrap();
        let v = v_data.as_slice::<f32>().unwrap();

        for s in 0..seq_len {
            for g in 0..self.num_kv_groups {
                let base = (s * self.num_kv_groups + g) * self.head_dim;
                self.keys[g].extend_from_slice(&self.tq.quantize_key(&k[base..base + self.head_dim]).unwrap());
                self.values[g].extend_from_slice(&self.tq.quantize_value(&v[base..base + self.head_dim]).unwrap());
            }
            self.current_len += 1;
        }
    }

    pub fn attend<B: Backend>(&self, q: Tensor<B, 4>, old_len: usize, num_heads: usize, device: &B::Device) -> Tensor<B, 4> {
        let [batch, q_len, q_heads, head_dim] = q.dims();
        assert_eq!(batch, 1, "KuantKVCache currently supports batch=1 inference");
        assert_eq!(q_heads, num_heads);
        assert_eq!(head_dim, self.head_dim);

        let repeats = num_heads / self.num_kv_groups;
        let q_data = q.into_data();
        let q_slice = q_data.as_slice::<f32>().unwrap();
        let mut out = vec![0.0f32; batch * q_len * num_heads * self.head_dim];
        let scale = (self.head_dim as f32).sqrt();

        for qi in 0..q_len {
            let max_attend = old_len + qi;
            for h in 0..num_heads {
                let group = h / repeats;
                let q_base = (qi * num_heads + h) * self.head_dim;
                let rotated_q = self.tq.rotate_query(&q_slice[q_base..q_base + self.head_dim]).unwrap();
                let mut scores = self.tq.attention_scores(
                    &rotated_q,
                    &self.keys[group],
                    self.current_len,
                    self.key_stride,
                ).unwrap();

                for (idx, score) in scores.iter_mut().enumerate() {
                    if idx > max_attend {
                        *score = -1.0e9;
                    } else {
                        *score /= scale;
                    }
                }

                let weights = softmax_vec(&scores);
                let combined = self.tq.attention_combine(
                    &self.values[group],
                    self.current_len,
                    self.value_stride,
                    &weights,
                ).unwrap();
                let out_base = (qi * num_heads + h) * self.head_dim;
                out[out_base..out_base + self.head_dim].copy_from_slice(&combined);
            }
        }

        Tensor::<B, 4>::from_data(TensorData::new(out, [batch, q_len, num_heads, self.head_dim]), device)
    }

    pub fn compressed_bytes(&self) -> usize {
        self.keys.iter().map(Vec::len).sum::<usize>() + self.values.iter().map(Vec::len).sum::<usize>()
    }
}
