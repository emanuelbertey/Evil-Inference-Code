// Optimized Ternary Kernels for CPU
// Based on BitNet b1.58 (arXiv:2410.16144) and bitnet.cpp implementations

pub const GROUP_SIZE: usize = 128;

/// I2_S Kernel: 2-bit Integer Signed Unpacking + MAD
/// Packs 16 ternary weights into a 32-bit integer for memory efficiency.
pub struct I2SKernel;

impl I2SKernel {
    pub fn pack_weights(weights: &[f32]) -> Vec<u32> {
        let mut packed = Vec::with_capacity((weights.len() + 15) / 16);
        for chunk in weights.chunks(16) {
            let mut p: u32 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                let bits = if w < -0.5 { 0b00 } else if w > 0.5 { 0b10 } else { 0b01 };
                p |= bits << (i * 2);
            }
            packed.push(p);
        }
        packed
    }

    #[inline(always)]
    fn compute_row(x_data: &[f32], packed_w: &[u32], scales: &[f32], _b: usize, o: usize, in_features: usize, x_offset: usize) -> f32 {
        let mut sum_pos = 0.0f32;
        let mut sum_neg = 0.0f32;
        let row_base = o * in_features;
        let w_idx_base = row_base / 16;
        let bit_offset = row_base % 16;

        if bit_offset == 0 {
            for i in 0..in_features {
                let w_idx = w_idx_base + (i / 16);
                let local = i % 16;
                let bits = (packed_w[w_idx] >> (local * 2)) & 0b11;
                if bits == 0b01 { continue; }
                let group_idx = ((row_base + i) / GROUP_SIZE).min(scales.len() - 1);
                let s = scales[group_idx];
                let x_val = x_data[x_offset + i];
                if bits == 0b10 { sum_pos += x_val * s; } else { sum_neg += x_val * s; }
            }
        } else {
            let first_bits_left = 16 - bit_offset;
            let packed_first = packed_w[w_idx_base];
            for i in 0..first_bits_left {
                let bits = (packed_first >> ((bit_offset + i) * 2)) & 0b11;
                if bits == 0b01 { continue; }
                let group_idx = ((row_base + i) / GROUP_SIZE).min(scales.len() - 1);
                let s = scales[group_idx];
                let x_val = x_data[x_offset + i];
                if bits == 0b10 { sum_pos += x_val * s; } else { sum_neg += x_val * s; }
            }
            let remaining = in_features - first_bits_left;
            let full_chunks = remaining / 16;
            for c in 0..full_chunks {
                let packed = packed_w[w_idx_base + 1 + c];
                let base_i = first_bits_left + c * 16;
                for j in 0..16 {
                    let bits = (packed >> (j * 2)) & 0b11;
                    if bits == 0b01 { continue; }
                    let group_idx = ((row_base + base_i + j) / GROUP_SIZE).min(scales.len() - 1);
                    let s = scales[group_idx];
                    let x_val = x_data[x_offset + base_i + j];
                    if bits == 0b10 { sum_pos += x_val * s; } else { sum_neg += x_val * s; }
                }
            }
            let tail_start = first_bits_left + full_chunks * 16;
            if tail_start < in_features {
                let packed = packed_w[w_idx_base + 1 + full_chunks];
                let tail_bits = in_features - tail_start;
                for j in 0..tail_bits {
                    let bits = (packed >> (j * 2)) & 0b11;
                    if bits == 0b01 { continue; }
                    let group_idx = ((row_base + tail_start + j) / GROUP_SIZE).min(scales.len() - 1);
                    let s = scales[group_idx];
                    let x_val = x_data[x_offset + tail_start + j];
                    if bits == 0b10 { sum_pos += x_val * s; } else { sum_neg += x_val * s; }
                }
            }
        }

        sum_pos - sum_neg
    }

    pub fn forward_raw(
        x_data: &[f32],
        batch: usize,
        packed_w: &[u32],
        scales: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        let total = batch * out_features;
        let mut out_data = vec![0.0f32; total];

        if total < 4096 {
            for idx in 0..total {
                let b = idx / out_features;
                let o = idx % out_features;
                out_data[idx] = Self::compute_row(x_data, packed_w, scales, b, o, in_features, b * in_features);
            }
            return out_data;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = std::cmp::max(1, (total + num_threads - 1) / num_threads);

        std::thread::scope(|s| {
            for (thread_idx, chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                if chunk.is_empty() { continue; }
                s.spawn(move || {
                    let start = thread_idx * chunk_size;
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let idx = start + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        *out_val = Self::compute_row(x_data, packed_w, scales, b, o, in_features, b * in_features);
                    }
                });
            }
        });

        out_data
    }
}

/// TL1 Kernel: Ternary Lookup Table (2 weights per 4-bit index)
pub struct TL1Kernel;

impl TL1Kernel {
    pub fn pack_weights(weights: &[f32]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((weights.len() + 1) / 2);
        for chunk in weights.chunks(2) {
            let mut p: u8 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                let val = if w < -0.5 { 0 } else if w > 0.5 { 2 } else { 1 };
                p += val * 3u8.pow(i as u32);
            }
            packed.push(p);
        }
        packed
    }

    pub fn forward_raw(
        x_data: &[f32],
        batch: usize,
        packed_w: &[u8],
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out_data = vec![0.0f32; batch * out_features];

        // FAST PATH: Avoid OS thread spawning overhead for small matrices
        if out_data.len() < 4096 {
            for b in 0..batch {
                for o in 0..out_features {
                    let mut sum = 0.0f32;
                    for i in (0..in_features).step_by(2) {
                        let w_idx = (o * in_features + i) / 2;
                        if w_idx >= packed_w.len() { break; }
                        let p = packed_w[w_idx];
                        
                        let x0 = x_data[b * in_features + i];
                        let x1 = if i + 1 < in_features { x_data[b * in_features + i + 1] } else { 0.0 };
                        
                        let lut_sum = match p {
                            0 => -x0 - x1, 1 => 0.0 - x1, 2 => x0 - x1,
                            3 => -x0 + 0.0, 4 => 0.0 + 0.0, 5 => x0 + 0.0,
                            6 => -x0 + x1, 7 => 0.0 + x1, 8 => x0 + x1,
                            _ => 0.0,
                        };
                        sum += lut_sum;
                    }
                    out_data[b * out_features + o] = sum * scale;
                }
            }
            return out_data;
        }

        // HILOS DE RUST (Nativos)
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = std::cmp::max(1, (out_data.len() + num_threads - 1) / num_threads);

        std::thread::scope(|s| {
            for (thread_idx, chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                if chunk.is_empty() { continue; }
                s.spawn(move || {
                    let start_idx = thread_idx * chunk_size;
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let idx = start_idx + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        
                        let mut sum = 0.0f32;
                        for i in (0..in_features).step_by(2) {
                            let w_idx = (o * in_features + i) / 2;
                            if w_idx >= packed_w.len() { break; }
                            let p = packed_w[w_idx];
                            
                            let x0 = x_data[b * in_features + i];
                            let x1 = if i + 1 < in_features { x_data[b * in_features + i + 1] } else { 0.0 };
                            
                            let lut_sum = match p {
                                0 => -x0 - x1, 1 => 0.0 - x1, 2 => x0 - x1,
                                3 => -x0 + 0.0, 4 => 0.0 + 0.0, 5 => x0 + 0.0,
                                6 => -x0 + x1, 7 => 0.0 + x1, 8 => x0 + x1,
                                _ => 0.0,
                            };
                            sum += lut_sum;
                        }
                        *out_val = sum * scale;
                    }
                });
            }
        });
        out_data
    }
}

/// TL2 Kernel: Ternary Lookup Table (3 weights per 5-bit index)
pub struct TL2Kernel;

impl TL2Kernel {
    pub fn pack_weights(weights: &[f32]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((weights.len() + 2) / 3);
        for chunk in weights.chunks(3) {
            let mut p: u8 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                let val = if w < -0.5 { 0 } else if w > 0.5 { 2 } else { 1 };
                p += val * 3u8.pow(i as u32);
            }
            packed.push(p);
        }
        packed
    }

    pub fn forward_raw(
        x_data: &[f32],
        batch: usize,
        packed_w: &[u8],
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out_data = vec![0.0f32; batch * out_features];

        // FAST PATH: Avoid OS thread spawning overhead for small matrices
        if out_data.len() < 4096 {
            for b in 0..batch {
                for o in 0..out_features {
                    let mut sum = 0.0f32;
                    for i in (0..in_features).step_by(3) {
                        let w_idx = (o * in_features + i) / 3;
                        if w_idx >= packed_w.len() { break; }
                        let p = packed_w[w_idx];
                        
                        let x0 = x_data[b * in_features + i];
                        let x1 = if i + 1 < in_features { x_data[b * in_features + i + 1] } else { 0.0 };
                        let x2 = if i + 2 < in_features { x_data[b * in_features + i + 2] } else { 0.0 };
                        
                        let w0 = match p % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                        let w1 = match (p / 3) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                        let w2 = match (p / 9) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };

                        sum += w0 * x0 + w1 * x1 + w2 * x2;
                    }
                    out_data[b * out_features + o] = sum * scale;
                }
            }
            return out_data;
        }

        // HILOS DE RUST (Nativos)
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = std::cmp::max(1, (out_data.len() + num_threads - 1) / num_threads);

        std::thread::scope(|s| {
            for (thread_idx, chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                if chunk.is_empty() { continue; }
                s.spawn(move || {
                    let start_idx = thread_idx * chunk_size;
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let idx = start_idx + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        
                        let mut sum = 0.0f32;
                        for i in (0..in_features).step_by(3) {
                            let w_idx = (o * in_features + i) / 3;
                            if w_idx >= packed_w.len() { break; }
                            let p = packed_w[w_idx];
                            
                            let x0 = x_data[b * in_features + i];
                            let x1 = if i + 1 < in_features { x_data[b * in_features + i + 1] } else { 0.0 };
                            let x2 = if i + 2 < in_features { x_data[b * in_features + i + 2] } else { 0.0 };
                            
                            let w0 = match p % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                            let w1 = match (p / 3) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                            let w2 = match (p / 9) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };

                            sum += w0 * x0 + w1 * x1 + w2 * x2;
                        }
                        *out_val = sum * scale;
                    }
                });
            }
        });
        out_data
    }
}
