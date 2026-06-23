// ─── Shared Operations ─────────────────────────────────────────────────────

use burn::tensor::{Tensor, backend::Backend, TensorData};

// ─── Softmax (raw f32 vec) ─────────────────────────────────────────────────

pub(crate) fn softmax_vec(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &score in scores {
        let v = (score - max).exp();
        exp.push(v);
        sum += v;
    }
    if sum <= 0.0 {
        vec![0.0; scores.len()]
    } else {
        exp.into_iter().map(|v| v / sum).collect()
    }
}

// ─── Fused RoPE (tensor ops, preserves autodiff) ──────────────────────────

pub fn apply_rope_fused<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq_len, nheads, head_dim] = q.dims();
    let nkv = k.dims()[2];
    let hh = head_dim / 2;
    let device = q.device();

    let inv_freq: Vec<f32> = (0..hh)
        .map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq_t = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), &device);

    let positions: Vec<f32> = (0..seq_len).map(|i| (offset + i) as f32).collect();
    let pos_t = Tensor::<B, 1>::from_floats(positions.as_slice(), &device);

    let freqs = pos_t.unsqueeze_dim::<2>(1) * inv_freq_t.unsqueeze_dim::<2>(0);
    let cos = freqs.clone().cos().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2);
    let sin = freqs.sin().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2);

    let q_first = q.clone().slice([0..batch, 0..seq_len, 0..nheads, 0..hh]);
    let q_second = q.slice([0..batch, 0..seq_len, 0..nheads, hh..head_dim]);
    let q_out_first = q_first.clone() * cos.clone() - q_second.clone() * sin.clone();
    let q_out_second = q_first * sin.clone() + q_second * cos.clone();
    let q_out = Tensor::cat(vec![q_out_first, q_out_second], 3);

    let k_first = k.clone().slice([0..batch, 0..seq_len, 0..nkv, 0..hh]);
    let k_second = k.slice([0..batch, 0..seq_len, 0..nkv, hh..head_dim]);
    let k_out_first = k_first.clone() * cos.clone() - k_second.clone() * sin.clone();
    let k_out_second = k_first * sin + k_second * cos;
    let k_out = Tensor::cat(vec![k_out_first, k_out_second], 3);

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
