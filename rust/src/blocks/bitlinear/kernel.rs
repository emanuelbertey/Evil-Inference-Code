// Optimized Ternary Kernels for CPU
// Based on BitNet b1.58 (arXiv:2410.16144) and bitnet.cpp implementations

/// I2_S Kernel: 2-bit Integer Signed Unpacking + MAD
/// Packs 16 ternary weights into a 32-bit integer for memory efficiency.
pub struct I2SKernel;

impl I2SKernel {
    /// Simulates the packing of ternary weights (-1, 0, 1) into 2-bit values (16 weights per u32)
    pub fn pack_weights(weights: &[f32]) -> Vec<u32> {
        let mut packed = Vec::with_capacity((weights.len() + 15) / 16);
        for chunk in weights.chunks(16) {
            let mut p: u32 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                // Map: -1.0 -> 0b00, 0.0 -> 0b01, 1.0 -> 0b10
                let bits = if w < -0.5 {
                    0b00
                } else if w > 0.5 {
                    0b10
                } else {
                    0b01
                };
                p |= bits << (i * 2);
            }
            packed.push(p);
        }
        packed
    }

    /// Forward pass simulating the I2_S CPU kernel behavior on raw slices.
    /// Uses per-group scales: each GROUP_SIZE weights share one scale.
    pub fn forward_raw(
        x_data: &[f32],
        batch: usize,
        packed_w: &[u32],
        out_features: usize,
        in_features: usize,
        scales: &[f32],
    ) -> Vec<f32> {
        const GROUP_SIZE: usize = 128;
        let mut out_data = vec![0.0f32; batch * out_features];

        // FAST PATH: Avoid OS thread spawning overhead for small matrices
        if out_data.len() < 4096 {
            for b in 0..batch {
                for o in 0..out_features {
                    let mut sum = 0.0f32;
                    for i in (0..in_features).step_by(16) {
                        let w_idx = (o * in_features + i) / 16;
                        if w_idx >= packed_w.len() { break; }
                        let packed = packed_w[w_idx];
                        
                        for j in 0..16 {
                            if i + j >= in_features { break; }
                            let bits = (packed >> (j * 2)) & 0b11;
                            if bits == 0b01 { continue; }
                            
                            let x_val = x_data[b * in_features + i + j];
                            // Per-group scale: group index based on weight position
                            let weight_pos = o * in_features + i + j;
                            let group_idx = (weight_pos / GROUP_SIZE).min(scales.len() - 1);
                            let s = scales[group_idx];
                            if bits == 0b10 { sum += x_val * s; } else { sum -= x_val * s; }
                        }
                    }
                    out_data[b * out_features + o] = sum;
                }
            }
            return out_data;
        }

        // HILOS DE RUST (Nativos) para matrices grandes
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
                        for i in (0..in_features).step_by(16) {
                            let w_idx = (o * in_features + i) / 16;
                            if w_idx >= packed_w.len() { break; }
                            let packed = packed_w[w_idx];
                            
                            for j in 0..16 {
                                if i + j >= in_features { break; }
                                let bits = (packed >> (j * 2)) & 0b11;
                                if bits == 0b01 { continue; }
                                
                                let x_val = x_data[b * in_features + i + j];
                                let weight_pos = o * in_features + i + j;
                                let group_idx = (weight_pos / GROUP_SIZE).min(scales.len() - 1);
                                let s = scales[group_idx];
                                if bits == 0b10 { sum += x_val * s; } else { sum -= x_val * s; }
                            }
                        }
                        *out_val = sum;
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
