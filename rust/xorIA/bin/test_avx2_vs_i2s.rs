use std::arch::x86_64::*;
use xlstm::blocks::bitlinear::kernel::{I2SKernel, GROUP_SIZE};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_unpack_to_reg(packed_w: &[u32], w_row: usize, in_features: usize, out: *mut i8) {
    let w_idx_base = w_row >> 4;
    let mut k = 0usize;
    while k + 31 < in_features {
        let mut vals = [0i8; 32];
        for j in 0..32 {
            let local = (k + j) & 15;
            let bits = (*packed_w.get_unchecked(w_idx_base + ((k + j) >> 4)) >> (local << 1)) & 0b11;
            *vals.get_unchecked_mut(j) = (bits as i8) - 1;
        }
        std::ptr::copy_nonoverlapping(vals.as_ptr(), out.add(k), 32);
        k += 32;
    }
    while k < in_features {
        let local = k & 15;
        let bits = (*packed_w.get_unchecked(w_idx_base + (k >> 4)) >> (local << 1)) & 0b11;
        *out.add(k) = (bits as i8) - 1;
        k += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_i8_32(w_ptr: *const i8, x_ptr: *const i8) -> i32 {
    let w = _mm256_loadu_si256(w_ptr as *const __m256i);
    let x = _mm256_loadu_si256(x_ptr as *const __m256i);

    let w_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 0));
    let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
    let x_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 0));
    let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));

    let prod_lo = _mm256_madd_epi16(w_lo, x_lo);
    let prod_hi = _mm256_madd_epi16(w_hi, x_hi);
    let sum = _mm256_add_epi32(prod_lo, prod_hi);

    let hi128 = _mm256_extracti128_si256(sum, 1);
    let lo128 = _mm256_castsi256_si128(sum);
    let pair = _mm_add_epi32(lo128, hi128);
    let shuffle = _mm_shuffle_epi32(pair, 0x4E);
    let pair2 = _mm_add_epi32(pair, shuffle);
    let shuffle2 = _mm_shuffle_epi32(pair2, 0xB1);
    let final_v = _mm_add_epi32(pair2, shuffle2);
    _mm_cvtsi128_si32(final_v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_i8_dot(w_i8: &[i8], x_i8: &[i8], in_features: usize) -> i32 {
    let mut sum = 0i32;
    let mut i = 0usize;

    while i + 31 < in_features {
        sum += avx2_dot_i8_32(w_i8.as_ptr().add(i), x_i8.as_ptr().add(i));
        i += 32;
    }

    while i + 7 < in_features {
        let w_ptr = w_i8.as_ptr().add(i) as *const i8;
        let x_ptr = x_i8.as_ptr().add(i) as *const i8;
        let w_vec = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w_ptr as *const __m128i));
        let x_vec = _mm256_cvtepi8_epi32(_mm_loadl_epi64(x_ptr as *const __m128i));
        let prod = _mm256_mullo_epi32(w_vec, x_vec);
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, prod);
        for &v in &tmp { sum += v; }
        i += 8;
    }

    while i < in_features {
        sum += *w_i8.get_unchecked(i) as i32 * *x_i8.get_unchecked(i) as i32;
        i += 1;
    }

    sum
}

unsafe fn avx2_unpack_weights(packed_w: &[u32], w_row: usize, in_features: usize) -> Vec<i8> {
    let mut buf = vec![0i8; in_features];
    avx2_unpack_to_reg(packed_w, w_row, in_features, buf.as_mut_ptr());
    buf
}

fn avx2_forward_i8(
    x_i8: &[i8], batch: usize,
    packed_w: &[u32], scales: &[f32],
    out_features: usize, in_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_features];

    let mut w_buf: Vec<i8> = vec![0i8; out_features * in_features];
    for o in 0..out_features {
        unsafe { avx2_unpack_to_reg(packed_w, o * in_features, in_features, w_buf.as_mut_ptr().add(o * in_features)); }
    }

    for b in 0..batch {
        let x_off = b * in_features;
        for o in 0..out_features {
            let raw = unsafe { avx2_i8_dot(&w_buf[o * in_features..(o + 1) * in_features], x_i8.get_unchecked(x_off..), in_features) };
            let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
            out[b * out_features + o] = raw as f32 * scales[g];
        }
    }
    out
}

fn avx2_forward_chunked(
    x_i8: &[i8], batch: usize,
    packed_w: &[u32], scales: &[f32],
    out_features: usize, in_features: usize,
    _chunk_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_features];
    let groups_per_row = in_features / GROUP_SIZE;

    for b in 0..batch {
        let x_off = b * in_features;
        for o in 0..out_features {
            let base_g = o * groups_per_row;
            let mut row_sum = 0.0f32;
            for gi in 0..groups_per_row {
                unsafe {
                    let w_row = (o * in_features + gi * GROUP_SIZE) / 16;
                    let x_ptr = x_i8.as_ptr().add(x_off + gi * GROUP_SIZE);
                    let raw = avx2_dot_group_inline(packed_w.as_ptr().add(w_row), x_ptr);
                    let si = (base_g + gi).min(scales.len() - 1);
                    row_sum += raw as f32 * *scales.as_ptr().add(si);
                }
            }
            out[b * out_features + o] = row_sum;
        }
    }
    out
}

fn make_test_data(batch: usize, in_features: usize, out_features: usize) -> (Vec<u32>, Vec<f32>, Vec<i8>, Vec<f32>) {
    let mut rng_state: u32 = 12345;
    let mut rand_i32 = || -> i32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        rng_state as i32
    };

    let packed_w = I2SKernel::pack_weights(
        &(0..(out_features * in_features))
            .map(|_| {
                let r = rand_i32().abs() % 3;
                match r { 0 => -1.0, 1 => 1.0, _ => 0.0 }
            })
            .collect::<Vec<_>>(),
    );

    let n_groups = (out_features * in_features + GROUP_SIZE - 1) / GROUP_SIZE;
    let scales: Vec<f32> = (0..n_groups).map(|i| 0.1 + (i as f32) * 0.001).collect();

    let x_i8: Vec<i8> = (0..(batch * in_features))
        .map(|_| ((rand_i32().abs() % 255) as i8).wrapping_sub(127))
        .collect();

    let x_f32: Vec<f32> = x_i8.iter().map(|&v| v as f32).collect();

    (packed_w, scales, x_i8, x_f32)
}

fn cpu_scalar_bench() {
    println!("\n=== CPU Scalar Benchmark (i8 dot de 512 elems) ===\n");

    let n = 512usize;
    let w: Vec<i8> = (0..n).map(|i| match i % 3 { 0 => -1, 1 => 1, _ => 0 }).collect();
    let x: Vec<i8> = (0..n).map(|i| ((i * 7 + 13) % 255) as i8 - 127).collect();

    let mut total = 0u64;
    let start = Instant::now();
    let one_sec = std::time::Duration::from_secs(1);

    while start.elapsed() < one_sec {
        let mut sum = 0i32;
        for i in 0..n {
            sum += w[i] as i32 * x[i] as i32;
        }
        std::hint::black_box(sum);
        total += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ops_sec = total as f64 / elapsed;
    let muls_sec = ops_sec * n as f64;
    println!("  Loop i8 * i8 ({} elems): {:.0} ops/s ({:.2} M mul/s)", n, ops_sec, muls_sec / 1e6);

    let mut total2 = 0u64;
    let start2 = Instant::now();
    let mut acc = 0i64;

    while start2.elapsed() < one_sec {
        for i in 0..n {
            acc += w[i] as i64 + x[i] as i64;
        }
        std::hint::black_box(acc);
        total2 += 1;
    }

    let elapsed2 = start2.elapsed().as_secs_f64();
    let ops2 = total2 as f64 / elapsed2;
    let adds2 = ops2 * n as f64;
    println!("  Loop i8 + i8 ({} elems): {:.0} ops/s ({:.2} M add/s)", n, ops2, adds2 / 1e6);
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 no soportado en esta CPU");
        return;
    }

    println!("=== AVX2 i8 vs Scalar I2S i8 ===\n");

    let configs = [
        ("1x128x128", 1usize, 128usize, 128usize),
        ("4x256x256", 4, 256, 256),
        ("1x512x512", 1, 512, 512),
        ("4x512x512", 4, 512, 512),
        ("1x16000x512", 1, 16000, 512),
        ("4x16000x512", 4, 16000, 512),
    ];

    for (name, batch, in_f, out_f) in configs {
        let (pw, sc, xi8, _xf32) = make_test_data(batch, in_f, out_f);

        let iters = if batch * in_f * out_f > 1_000_000 { 3 } else { 50 };

        let start_avx = Instant::now();
        let mut avx_out = Vec::new();
        for _ in 0..iters {
            avx_out = avx2_forward_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        }
        let avx_time = start_avx.elapsed().as_secs_f64() / iters as f64;

        let start_scalar = Instant::now();
        let mut scalar_out = Vec::new();
        for _ in 0..iters {
            scalar_out = I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        }
        let scalar_time = start_scalar.elapsed().as_secs_f64() / iters as f64;

        let mut avx_mismatches = 0usize;
        let mut avx_max_diff = 0.0f32;
        for (a, b) in avx_out.iter().zip(scalar_out.iter()) {
            let diff = (a - b).abs();
            if diff > avx_max_diff { avx_max_diff = diff; }
            if a.to_bits() != b.to_bits() { avx_mismatches += 1; }
        }

        let speedup = scalar_time / avx_time;
        println!(
            "{:>14}: scalar={:.3}ms avx2={:.3}ms speedup={:.2}x mismatches={} max_diff={:.6}",
            name, scalar_time * 1000.0, avx_time * 1000.0, speedup, avx_mismatches, avx_max_diff
        );
    }

    println!("\n=== AVX2 Chunked I2S vs I2S Completo ===\n");

    let configs2 = [
        ("1x512x2048",  1usize, 512usize, 2048usize),
        ("4x512x2048",  4, 512, 2048),
        ("1x2048x2048", 1, 2048, 2048),
        ("4x2048x2048", 4, 2048, 2048),
        ("1x4096x2048", 1, 4096, 2048),
    ];

    for (name, batch, in_f, out_f) in configs2 {
        let (pw, sc, xi8, _xf32) = make_test_data(batch, in_f, out_f);

        let iters = if batch * in_f * out_f > 2_000_000 { 2 } else { 5 };

        let start_i2s = Instant::now();
        let mut i2s_out = Vec::new();
        for _ in 0..iters {
            i2s_out = I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        }
        let i2s_time = start_i2s.elapsed().as_secs_f64() / iters as f64;

        println!("{}: I2S completo ({}) = {:.3}ms", name, batch * out_f, i2s_time * 1000.0);

        for chunk_size in [128, 256, 512, 1024] {
            let start_avx = Instant::now();
            let mut avx_out = Vec::new();
            for _ in 0..iters {
                avx_out = avx2_forward_chunked(&xi8, batch, &pw, &sc, out_f, in_f, chunk_size);
            }
            let avx_time = start_avx.elapsed().as_secs_f64() / iters as f64;

            let mut mismatches = 0usize;
            let mut max_diff = 0.0f32;
            for (a, b) in avx_out.iter().zip(i2s_out.iter()) {
                let diff = (a - b).abs();
                if diff > max_diff { max_diff = diff; }
                if a.to_bits() != b.to_bits() { mismatches += 1; }
            }

            let speedup = i2s_time / avx_time;
            println!(
                "  AVX2 I2S (chunk={:>4}): {:.3}ms speedup={:.2}x mismatches={} max_diff={:.6}",
                chunk_size, avx_time * 1000.0, speedup, mismatches, max_diff
            );
        }
        println!();
    }

    cpu_scalar_bench();

    // ─── Test 3: AVX2 Threaded (unpack inline, per-group scales) ───────────
    println!("\n=== AVX2 Threaded vs I2S Threaded ===\n");

    for (name, batch, in_f, out_f) in configs {
        let (pw, sc, xi8, _) = make_test_data(batch, in_f, out_f);

        let iters = if batch * in_f * out_f > 1_000_000 { 3 } else { 30 };

        let start_avx = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(avx2_forward_threaded(&xi8, batch, &pw, &sc, out_f, in_f));
        }
        let avx_time = start_avx.elapsed().as_secs_f64() / iters as f64;

        let start_scalar = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f));
        }
        let scalar_time = start_scalar.elapsed().as_secs_f64() / iters as f64;

        let avx_out = avx2_forward_threaded(&xi8, batch, &pw, &sc, out_f, in_f);
        let scalar_out = I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        let mut mismatches = 0usize;
        let mut max_diff = 0.0f32;
        for (a, b) in avx_out.iter().zip(scalar_out.iter()) {
            let diff = (a - b).abs();
            if diff > max_diff { max_diff = diff; }
            if diff > 1e-4 { mismatches += 1; }
        }

        let speedup = scalar_time / avx_time;
        println!(
            "{:>14}: scalar={:.3}ms  avx2_thr={:.3}ms  speedup={:.2}x  mismatches={}  max_diff={:.6}",
            name, scalar_time * 1000.0, avx_time * 1000.0, speedup, mismatches, max_diff
        );
    }

    println!("\n=== DONE ===");
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_group_inline(packed: *const u32, x: *const i8) -> i32 {
    let mut sum = _mm256_setzero_si256();
    let mut c = 0usize;
    while c + 31 < GROUP_SIZE {
        let word_idx = c >> 4;
        let p0 = *packed.add(word_idx);
        let p1 = *packed.add(word_idx + 1);
        let mut vals = [0i8; 32];
        let mut i = 0usize;
        while i < 16 {
            *vals.get_unchecked_mut(i) = ((p0 >> (i << 1)) & 3) as i8 - 1;
            *vals.get_unchecked_mut(i + 16) = ((p1 >> (i << 1)) & 3) as i8 - 1;
            i += 1;
        }
        let w = _mm256_loadu_si256(vals.as_ptr() as *const __m256i);
        let x_vec = _mm256_loadu_si256(x.add(c) as *const __m256i);
        let w_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 0));
        let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
        let x_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 0));
        let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(w_lo, x_lo));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(w_hi, x_hi));
        c += 32;
    }
    let hi128 = _mm256_extracti128_si256(sum, 1);
    let lo128 = _mm256_castsi256_si128(sum);
    let pair = _mm_add_epi32(lo128, hi128);
    let shuf = _mm_shuffle_epi32(pair, 0x4E);
    let pair2 = _mm_add_epi32(pair, shuf);
    let shuf2 = _mm_shuffle_epi32(pair2, 0xB1);
    let result = _mm_add_epi32(pair2, shuf2);
    _mm_cvtsi128_si32(result)
}

fn avx2_forward_threaded(
    x_i8: &[i8], batch: usize,
    packed_w: &[u32], scales: &[f32],
    out_features: usize, in_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_features];
    let groups_per_row = in_features / GROUP_SIZE;
    let total_rows = batch * out_features;
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    std::thread::scope(|s| {
        let mut remaining = &mut out[..];
        let mut start_row = 0usize;

        for _ in 0..num_threads {
            let n = rows_per_thread.min(remaining.len());
            if n == 0 { break; }
            let (chunk, rest) = remaining.split_at_mut(n);
            remaining = rest;

            let pw_usize = packed_w.as_ptr() as usize;
            let sc_usize = scales.as_ptr() as usize;
            let xi8_usize = x_i8.as_ptr() as usize;
            let n_el = in_features;
            let n_gr = groups_per_row;
            let n_out = out_features;
            let sr = start_row;
            let sc_len = scales.len();

            s.spawn(move || {
                unsafe {
                    let pw = pw_usize as *const u32;
                    let sc = sc_usize as *const f32;
                    let xi8 = xi8_usize as *const i8;
                    for local_idx in 0..chunk.len() {
                        let idx = sr + local_idx;
                        let b = idx / n_out;
                        let o = idx % n_out;
                        let x_off = b * n_el;
                        let w_row_base = o * n_el;
                        let base_g = o * n_gr;
                        let mut row_sum = 0.0f32;
                        for gi in 0..n_gr {
                            let packed_ptr = pw.add((w_row_base + gi * GROUP_SIZE) / 16);
                            let x_ptr = xi8.add(x_off + gi * GROUP_SIZE);
                            let raw = avx2_dot_group_inline(packed_ptr, x_ptr);
                            let si = (base_g + gi).min(sc_len - 1);
                            row_sum += raw as f32 * *sc.add(si);
                        }
                        *chunk.get_unchecked_mut(local_idx) = row_sum;
                    }
                }
            });

            start_row += n;
        }
    });

    out
}
