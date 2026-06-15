pub const GROUP_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    I2S,
    Tile16,
}

pub struct I2SKernel;

impl I2SKernel {
    pub fn pack_weights(weights: &[f32]) -> Vec<u32> {
        let mut packed = Vec::with_capacity((weights.len() + 15) >> 4);
        for chunk in weights.chunks(16) {
            let mut p: u32 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                let bits = if w < -0.5 { 0b00 } else if w > 0.5 { 0b10 } else { 0b01 };
                p |= bits << (i << 1);
            }
            packed.push(p);
        }
        packed
    }

    #[inline(always)]
    fn compute_row_aligned(
        x_data: &[f32], x_off: usize,
        packed_w: &[u32], w_row_base: usize,
        in_features: usize,
    ) -> f32 {
        unsafe {
            let mut sum_pos = 0.0f32;
            let mut sum_neg = 0.0f32;
            let w_idx_base = w_row_base >> 4;
            let mut i = 0usize;

            while i + 15 < in_features {
                let packed = *packed_w.get_unchecked(w_idx_base + (i >> 4));
                let x_base = x_off + i;

                let bits0 = (packed) & 0b11;
                let bits1 = (packed >> 2) & 0b11;
                let bits2 = (packed >> 4) & 0b11;
                let bits3 = (packed >> 6) & 0b11;
                let bits4 = (packed >> 8) & 0b11;
                let bits5 = (packed >> 10) & 0b11;
                let bits6 = (packed >> 12) & 0b11;
                let bits7 = (packed >> 14) & 0b11;
                let bits8 = (packed >> 16) & 0b11;
                let bits9 = (packed >> 18) & 0b11;
                let bits10 = (packed >> 20) & 0b11;
                let bits11 = (packed >> 22) & 0b11;
                let bits12 = (packed >> 24) & 0b11;
                let bits13 = (packed >> 26) & 0b11;
                let bits14 = (packed >> 28) & 0b11;
                let bits15 = (packed >> 30) & 0b11;

                let x0 = *x_data.get_unchecked(x_base);
                let x1 = *x_data.get_unchecked(x_base + 1);
                let x2 = *x_data.get_unchecked(x_base + 2);
                let x3 = *x_data.get_unchecked(x_base + 3);
                let x4 = *x_data.get_unchecked(x_base + 4);
                let x5 = *x_data.get_unchecked(x_base + 5);
                let x6 = *x_data.get_unchecked(x_base + 6);
                let x7 = *x_data.get_unchecked(x_base + 7);
                let x8 = *x_data.get_unchecked(x_base + 8);
                let x9 = *x_data.get_unchecked(x_base + 9);
                let x10 = *x_data.get_unchecked(x_base + 10);
                let x11 = *x_data.get_unchecked(x_base + 11);
                let x12 = *x_data.get_unchecked(x_base + 12);
                let x13 = *x_data.get_unchecked(x_base + 13);
                let x14 = *x_data.get_unchecked(x_base + 14);
                let x15 = *x_data.get_unchecked(x_base + 15);

                if bits0 == 0b10 { sum_pos += x0; } else if bits0 == 0b00 { sum_neg += x0; }
                if bits1 == 0b10 { sum_pos += x1; } else if bits1 == 0b00 { sum_neg += x1; }
                if bits2 == 0b10 { sum_pos += x2; } else if bits2 == 0b00 { sum_neg += x2; }
                if bits3 == 0b10 { sum_pos += x3; } else if bits3 == 0b00 { sum_neg += x3; }
                if bits4 == 0b10 { sum_pos += x4; } else if bits4 == 0b00 { sum_neg += x4; }
                if bits5 == 0b10 { sum_pos += x5; } else if bits5 == 0b00 { sum_neg += x5; }
                if bits6 == 0b10 { sum_pos += x6; } else if bits6 == 0b00 { sum_neg += x6; }
                if bits7 == 0b10 { sum_pos += x7; } else if bits7 == 0b00 { sum_neg += x7; }
                if bits8 == 0b10 { sum_pos += x8; } else if bits8 == 0b00 { sum_neg += x8; }
                if bits9 == 0b10 { sum_pos += x9; } else if bits9 == 0b00 { sum_neg += x9; }
                if bits10 == 0b10 { sum_pos += x10; } else if bits10 == 0b00 { sum_neg += x10; }
                if bits11 == 0b10 { sum_pos += x11; } else if bits11 == 0b00 { sum_neg += x11; }
                if bits12 == 0b10 { sum_pos += x12; } else if bits12 == 0b00 { sum_neg += x12; }
                if bits13 == 0b10 { sum_pos += x13; } else if bits13 == 0b00 { sum_neg += x13; }
                if bits14 == 0b10 { sum_pos += x14; } else if bits14 == 0b00 { sum_neg += x14; }
                if bits15 == 0b10 { sum_pos += x15; } else if bits15 == 0b00 { sum_neg += x15; }

                i += 16;
            }

            while i < in_features {
                let local = i & 15;
                let bits = (*packed_w.get_unchecked(w_idx_base + (i >> 4)) >> (local << 1)) & 0b11;
                let x_val = *x_data.get_unchecked(x_off + i);
                if bits == 0b10 { sum_pos += x_val; } else if bits == 0b00 { sum_neg += x_val; }
                i += 1;
            }

            sum_pos - sum_neg
        }
    }

    fn forward_inner(
        x_data: &[f32], batch: usize,
        packed_w: &[u32], scales: &[f32],
        out_features: usize, in_features: usize,
        out_data: &mut [f32],
    ) {
        if batch * out_features <= 2 {
            for b in 0..batch {
                let x_off = b * in_features;
                for o in 0..out_features {
                    let raw = Self::compute_row_aligned(x_data, x_off, packed_w, o * in_features, in_features);
                    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
                    unsafe { *out_data.get_unchecked_mut(b * out_features + o) = raw * *scales.get_unchecked(g); }
                }
            }
            return;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        std::thread::scope(|s| {
            let rows_per_thread = (batch * out_features + num_threads - 1) / num_threads;
            let mut remaining = out_data;
            for t in 0..num_threads {
                let n = rows_per_thread.min(remaining.len());
                if n == 0 { break; }
                let (chunk, rest) = remaining.split_at_mut(n);
                remaining = rest;
                let start_row = t * rows_per_thread;
                s.spawn(move || {
                    for local_idx in 0..chunk.len() {
                        let idx = start_row + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        let w_row = o * in_features;
                        let x_off = b * in_features;

                        let raw = Self::compute_row_aligned(x_data, x_off, packed_w, w_row, in_features);
                        let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
                        unsafe { *chunk.get_unchecked_mut(local_idx) = raw * *scales.get_unchecked(g); }
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
        let mut packed = Vec::with_capacity((weights.len() + 1) >> 1);
        for chunk in weights.chunks(2) {
            let mut p: u8 = 0;
            for (i, &w) in chunk.iter().enumerate() {
                let val = if w < -0.5 { 0u8 } else if w > 0.5 { 2 } else { 1 };
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
        unsafe {
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i + 1 < in_features {
                let p = *packed_w.get_unchecked((w_row + i) >> 1);
                let x0 = *x_data.get_unchecked(x_off + i);
                let x1 = *x_data.get_unchecked(x_off + i + 1);
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
                let p = *packed_w.get_unchecked((w_row + i) >> 1);
                let x0 = *x_data.get_unchecked(x_off + i);
                sum += match p & 3 {
                    0 => -x0,
                    2 => x0,
                    _ => 0.0,
                };
            }
            let g = (w_row / GROUP_SIZE).min(scales.len() - 1);
            sum * *scales.get_unchecked(g)
        }
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
                    unsafe { *out_data.get_unchecked_mut(b * out_features + o) = Self::compute_row(x_data, x_off, packed_w, o * in_features, scales, in_features); }
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
                let val = if w < -0.5 { 0u8 } else if w > 0.5 { 2 } else { 1 };
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
        unsafe {
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i + 2 < in_features {
                let p = *packed_w.get_unchecked((w_row + i) / 3);
                let x0 = *x_data.get_unchecked(x_off + i);
                let x1 = *x_data.get_unchecked(x_off + i + 1);
                let x2 = *x_data.get_unchecked(x_off + i + 2);
                let w0 = match p % 3 { 0 => -1.0f32, 2 => 1.0, _ => 0.0 };
                let w1 = match (p / 3) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                let w2 = match (p / 9) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                sum += w0 * x0 + w1 * x1 + w2 * x2;
                i += 3;
            }
            while i < in_features {
                let p = *packed_w.get_unchecked((w_row + i) / 3);
                let local = (w_row + i) % 3;
                let w = match (p / 3u8.pow(local as u32)) % 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 };
                sum += w * *x_data.get_unchecked(x_off + i);
                i += 1;
            }
            let g = (w_row / GROUP_SIZE).min(scales.len() - 1);
            sum * *scales.get_unchecked(g)
        }
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
                    unsafe { *out_data.get_unchecked_mut(b * out_features + o) = Self::compute_row(x_data, x_off, packed_w, o * in_features, scales, in_features); }
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

pub struct I2STile16Kernel;

impl I2STile16Kernel {
    pub fn quantize_to_i8(weights: &[f32]) -> Vec<i8> {
        weights.iter().map(|&w| {
            if w < -0.5 { -1i8 } else if w > 0.5 { 1i8 } else { 0i8 }
        }).collect()
    }

    #[inline(always)]
    fn compute_row(
        x_data: &[f32], x_off: usize,
        w: &[i8], w_row: usize,
        scales: &[f32], in_features: usize,
    ) -> f32 {
        unsafe {
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i + 7 < in_features {
                let w0 = *w.get_unchecked(w_row + i) as f32;
                let w1 = *w.get_unchecked(w_row + i + 1) as f32;
                let w2 = *w.get_unchecked(w_row + i + 2) as f32;
                let w3 = *w.get_unchecked(w_row + i + 3) as f32;
                let w4 = *w.get_unchecked(w_row + i + 4) as f32;
                let w5 = *w.get_unchecked(w_row + i + 5) as f32;
                let w6 = *w.get_unchecked(w_row + i + 6) as f32;
                let w7 = *w.get_unchecked(w_row + i + 7) as f32;
                let x0 = *x_data.get_unchecked(x_off + i);
                let x1 = *x_data.get_unchecked(x_off + i + 1);
                let x2 = *x_data.get_unchecked(x_off + i + 2);
                let x3 = *x_data.get_unchecked(x_off + i + 3);
                let x4 = *x_data.get_unchecked(x_off + i + 4);
                let x5 = *x_data.get_unchecked(x_off + i + 5);
                let x6 = *x_data.get_unchecked(x_off + i + 6);
                let x7 = *x_data.get_unchecked(x_off + i + 7);
                sum += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7;
                i += 8;
            }
            while i < in_features {
                sum += *w.get_unchecked(w_row + i) as f32 * *x_data.get_unchecked(x_off + i);
                i += 1;
            }
            let g = (w_row / GROUP_SIZE).min(scales.len() - 1);
            sum * *scales.get_unchecked(g)
        }
    }

    pub fn forward_raw(
        x_data: &[f32], batch: usize,
        w_i8: &[i8], scales: &[f32],
        out_features: usize, in_features: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_features];
        let total = batch * out_features;
        if total < 16 {
            for b in 0..batch {
                let x_off = b * in_features;
                for o in 0..out_features {
                    unsafe {
                        *out.get_unchecked_mut(b * out_features + o) =
                            Self::compute_row(x_data, x_off, w_i8, o * in_features, scales, in_features);
                    }
                }
            }
            return out;
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let rows_per_thread = (total + num_threads - 1) / num_threads;

        std::thread::scope(|s| {
            let mut remaining = &mut out[..];
            let mut start_row = 0usize;
            for _ in 0..num_threads {
                let n = rows_per_thread.min(remaining.len());
                if n == 0 { break; }
                let (chunk, rest) = remaining.split_at_mut(n);
                remaining = rest;
                let sr = start_row;
                start_row += n;
                s.spawn(move || {
                    for local_idx in 0..chunk.len() {
                        let idx = sr + local_idx;
                        let b = idx / out_features;
                        let o = idx % out_features;
                        unsafe {
                            *chunk.get_unchecked_mut(local_idx) =
                                Self::compute_row(x_data, b * in_features, w_i8, o * in_features, scales, in_features);
                        }
                    }
                });
            }
        });

        out
    }
}
