pub const GROUP_SIZE: usize = 128;
const TILE_ROWS: usize = 32;

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
    fn compute_row_aligned(
        x_data: &[f32], x_off: usize,
        packed_w: &[u32], w_row_base: usize,
        scales: &[f32], in_features: usize,
    ) -> f32 {
        let mut sum_pos = 0.0f32;
        let mut sum_neg = 0.0f32;
        let w_idx_base = w_row_base >> 4;
        let mut i = 0usize;
        while i + 15 < in_features {
            let packed = packed_w[w_idx_base + (i >> 4)];
            let mut local = 0u32;
            while local < 16 {
                let bits = (packed >> (local * 2)) & 0b11;
                let x_val = x_data[x_off + i + local as usize];
                if bits == 0b10 { sum_pos += x_val; }
                else if bits == 0b00 { sum_neg += x_val; }
                local += 1;
            }
            i += 16;
        }
        if i < in_features {
            let packed = packed_w[w_idx_base + (i >> 4)];
            while i < in_features {
                let local = (i & 15) as u32;
                let bits = (packed >> (local * 2)) & 0b11;
                let x_val = x_data[x_off + i];
                if bits == 0b10 { sum_pos += x_val; }
                else if bits == 0b00 { sum_neg += x_val; }
                i += 1;
            }
        }
        sum_pos - sum_neg
    }

    fn forward_inner(
        x_data: &[f32], batch: usize,
        packed_w: &[u32], scales: &[f32],
        out_features: usize, in_features: usize,
        out_data: &mut [f32],
    ) {
        let num_tiles = (out_features + TILE_ROWS - 1) / TILE_ROWS;
        let total_tiles = batch * num_tiles;

        if total_tiles <= 2 {
            for b in 0..batch {
                let x_off = b * in_features;
                for o in 0..out_features {
                    let raw = Self::compute_row_aligned(x_data, x_off, packed_w, o * in_features, scales, in_features);
                    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
                    out_data[b * out_features + o] = raw * scales[g];
                }
            }
            return;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        std::thread::scope(|s| {
            let tiles_per_thread = (total_tiles + num_threads - 1) / num_threads;
            for t in 0..num_threads {
                let start = t * tiles_per_thread;
                let end = (start + tiles_per_thread).min(total_tiles);
                if start >= end { continue; }
                s.spawn(move || {
                    for tile_idx in start..end {
                        let b = tile_idx / num_tiles;
                        let tile = tile_idx % num_tiles;
                        let o_start = tile * TILE_ROWS;
                        let o_end = (o_start + TILE_ROWS).min(out_features);
                        let x_off = b * in_features;

                        for o in o_start..o_end {
                            let w_row = o * in_features;
                            let w_idx_base = w_row >> 4;
                            let mut sum_pos = 0.0f32;
                            let mut sum_neg = 0.0f32;
                            let mut i = 0usize;

                            while i + 15 < in_features {
                                let packed = packed_w[w_idx_base + (i >> 4)];
                                let mut j = 0u32;
                                while j < 16 {
                                    let bits = (packed >> (j * 2)) & 0b11;
                                    let x_val = x_data[x_off + i + j as usize];
                                    if bits == 0b10 { sum_pos += x_val; }
                                    else if bits == 0b00 { sum_neg += x_val; }
                                    j += 1;
                                }
                                i += 16;
                            }
                            if i < in_features {
                                let packed = packed_w[w_idx_base + (i >> 4)];
                                while i < in_features {
                                    let local = (i & 15) as u32;
                                    let bits = (packed >> (local * 2)) & 0b11;
                                    let x_val = x_data[x_off + i];
                                    if bits == 0b10 { sum_pos += x_val; }
                                    else if bits == 0b00 { sum_neg += x_val; }
                                    i += 1;
                                }
                            }

                            let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
                            out_data[b * out_features + o] = (sum_pos - sum_neg) * scales[g];
                        }
                    }
                });
            }
        });
    }

    pub fn forward_raw(
        x_data: &[f32], batch: usize,
        packed_w: &[u32], scales: &[f32],
        out_features: usize, in_features: usize,
    ) -> Vec<f32> {
        let mut out_data = vec![0.0f32; batch * out_features];
        Self::forward_inner(x_data, batch, packed_w, scales, out_features, in_features, &mut out_data);
        out_data
    }
}

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

    #[inline(always)]
    fn compute_row(
        x_data: &[f32], x_off: usize,
        packed_w: &[u8], w_row: usize,
        scales: &[f32], in_features: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i + 1 < in_features {
            let w_idx = (w_row + i) >> 1;
            let p = packed_w[w_idx];
            let x0 = x_data[x_off + i];
            let x1 = x_data[x_off + i + 1];
            let v = match p {
                0 => -x0 - x1,
                1 => -x1,
                2 => x0 - x1,
                3 => -x0,
                4 => 0.0,
                5 => x0,
                6 => -x0 + x1,
                7 => x1,
                8 => x0 + x1,
                _ => 0.0,
            };
            sum += v;
            i += 2;
        }
        if i < in_features {
            let w_idx = (w_row + i) >> 1;
            let p = packed_w[w_idx];
            let x0 = x_data[x_off + i];
            sum += match p & 3 {
                0 => -x0,
                2 => x0,
                _ => 0.0,
            };
        }
        let g = (w_row / GROUP_SIZE).min(scales.len() - 1);
        sum * scales[g]
    }

    fn forward_inner(
        x_data: &[f32], batch: usize,
        packed_w: &[u8], scales: &[f32],
        out_features: usize, in_features: usize,
        out_data: &mut [f32],
    ) {
        let total_out = batch * out_features;
        if total_out < 4 {
            for b in 0..batch {
                let x_off = b * in_features;
                for o in 0..out_features {
                    out_data[b * out_features + o] = Self::compute_row(x_data, x_off, packed_w, o * in_features, scales, in_features);
                }
            }
            return;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = std::cmp::max(1, (total_out + num_threads - 1) / num_threads);

        std::thread::scope(|s| {
            for (ti, chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                let start = ti * chunk_size;
                s.spawn(move || {
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let idx = start + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        *out_val = Self::compute_row(x_data, b * in_features, packed_w, o * in_features, scales, in_features);
                    }
                });
            }
        });
    }

    pub fn forward_raw(
        x_data: &[f32], batch: usize,
        packed_w: &[u8], scales: &[f32],
        out_features: usize, in_features: usize,
    ) -> Vec<f32> {
        let mut out_data = vec![0.0f32; batch * out_features];
        Self::forward_inner(x_data, batch, packed_w, scales, out_features, in_features, &mut out_data);
        out_data
    }
}

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

    #[inline(always)]
    fn compute_row(
        x_data: &[f32], x_off: usize,
        packed_w: &[u8], w_row: usize,
        scales: &[f32], in_features: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i + 2 < in_features {
            let w_idx = (w_row + i) / 3;
            let p = packed_w[w_idx];
            let x0 = x_data[x_off + i];
            let x1 = x_data[x_off + i + 1];
            let x2 = x_data[x_off + i + 2];
            let w0 = match p % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
            let w1 = match (p / 3) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
            let w2 = match (p / 9) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
            sum += w0 * x0 + w1 * x1 + w2 * x2;
            i += 3;
        }
        while i < in_features {
            let w_idx = (w_row + i) / 3;
            let local = (w_row + i) % 3;
            let p = packed_w[w_idx];
            let w = match (p / 3u8.pow(local as u32)) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
            sum += w * x_data[x_off + i];
            i += 1;
        }
        let g = (w_row / GROUP_SIZE).min(scales.len() - 1);
        sum * scales[g]
    }

    fn forward_inner(
        x_data: &[f32], batch: usize,
        packed_w: &[u8], scales: &[f32],
        out_features: usize, in_features: usize,
        out_data: &mut [f32],
    ) {
        let total_out = batch * out_features;
        if total_out < 4 {
            for b in 0..batch {
                let x_off = b * in_features;
                for o in 0..out_features {
                    out_data[b * out_features + o] = Self::compute_row(x_data, x_off, packed_w, o * in_features, scales, in_features);
                }
            }
            return;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = std::cmp::max(1, (total_out + num_threads - 1) / num_threads);

        std::thread::scope(|s| {
            for (ti, chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                let start = ti * chunk_size;
                s.spawn(move || {
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let idx = start + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        *out_val = Self::compute_row(x_data, b * in_features, packed_w, o * in_features, scales, in_features);
                    }
                });
            }
        });
    }

    pub fn forward_raw(
        x_data: &[f32], batch: usize,
        packed_w: &[u8], scales: &[f32],
        out_features: usize, in_features: usize,
    ) -> Vec<f32> {
        let mut out_data = vec![0.0f32; batch * out_features];
        Self::forward_inner(x_data, batch, packed_w, scales, out_features, in_features, &mut out_data);
        out_data
    }
}
