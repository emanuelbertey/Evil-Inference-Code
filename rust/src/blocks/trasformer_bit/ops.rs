// ─── Shared Operations ─────────────────────────────────────────────────────

use burn::tensor::{Tensor, backend::Backend, TensorData, Bool};

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

// ─── RoPE (tensor ops, preserves autodiff) ────────────────────────────────

pub fn apply_rope<B: Backend>(
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

// ─── RoPE Fused (raw slice ops, fast inference, no autodiff) ───────────────

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

// ─── RoPE Partial (tensor ops, preserves autodiff) ─────────────────────────
// Rotates only `rotary_pct` of the head dimensions (e.g., 0.5 = 50%).
// First rotary_dim dimensions are rotated; rest pass through unchanged.
// Used by Kimi K2, Phi-2, etc. for training stability.

pub fn apply_rope_partial<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: usize,
    rotary_pct: f64,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq_len, nheads, head_dim] = q.dims();
    let nkv = k.dims()[2];
    let rotary_dim = ((head_dim as f64) * rotary_pct) as usize;
    let rotary_dim = rotary_dim - (rotary_dim % 2);
    if rotary_dim >= head_dim || rotary_dim == 0 {
        return if rotary_dim >= head_dim { apply_rope(q, k, offset) } else { (q, k) };
    }
    let hh = rotary_dim / 2;
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

    let q_rot = q.clone().slice([0..batch, 0..seq_len, 0..nheads, 0..rotary_dim]);
    let q_pass = q.slice([0..batch, 0..seq_len, 0..nheads, rotary_dim..head_dim]);
    let qr_first = q_rot.clone().slice([0..batch, 0..seq_len, 0..nheads, 0..hh]);
    let qr_second = q_rot.slice([0..batch, 0..seq_len, 0..nheads, hh..rotary_dim]);
    let q_out_first_part = qr_first.clone() * cos.clone() - qr_second.clone() * sin.clone();
    let q_out_second_part = qr_first * sin.clone() + qr_second * cos.clone();
    let q_out_rot = Tensor::cat(vec![q_out_first_part, q_out_second_part], 3);
    let q_out = Tensor::cat(vec![q_out_rot, q_pass], 3);

    let k_rot = k.clone().slice([0..batch, 0..seq_len, 0..nkv, 0..rotary_dim]);
    let k_pass = k.slice([0..batch, 0..seq_len, 0..nkv, rotary_dim..head_dim]);
    let kr_first = k_rot.clone().slice([0..batch, 0..seq_len, 0..nkv, 0..hh]);
    let kr_second = k_rot.slice([0..batch, 0..seq_len, 0..nkv, hh..rotary_dim]);
    let k_out_first_part = kr_first.clone() * cos.clone() - kr_second.clone() * sin.clone();
    let k_out_second_part = kr_first * sin + kr_second * cos;
    let k_out_rot = Tensor::cat(vec![k_out_first_part, k_out_second_part], 3);
    let k_out = Tensor::cat(vec![k_out_rot, k_pass], 3);

    (q_out, k_out)
}

// ─── RoPE Partial Fused (raw slice ops, fast inference, no autodiff) ──────

pub fn apply_rope_fused_partial<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: usize,
    rotary_pct: f64,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq_len, nheads, head_dim] = q.dims();
    let nkv = k.dims()[2];
    let rotary_dim = ((head_dim as f64) * rotary_pct) as usize;
    let rotary_dim = rotary_dim - (rotary_dim % 2);
    if rotary_dim >= head_dim || rotary_dim == 0 {
        return if rotary_dim >= head_dim { apply_rope_fused(q, k, offset) } else { (q, k) };
    }
    let hh = rotary_dim / 2;
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
    let mask = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &device)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    scores.mask_fill(mask, f32::NEG_INFINITY)
}

pub fn apply_causal_mask_with_offset<B: Backend>(scores: Tensor<B, 4>, q_len: usize, kv_len: usize) -> Tensor<B, 4> {
    let device = scores.device();
    let offset = (kv_len - q_len) as i64;
    let mask = Tensor::<B, 2, Bool>::tril_mask([q_len, kv_len], offset, &device)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    scores.mask_fill(mask, f32::NEG_INFINITY)
}
