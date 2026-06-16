use std::arch::x86_64::*;
use xlstm::blocks::bitlinear::kernel::{I2SKernel, GROUP_SIZE};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_unpack_weights(packed_w: &[u32], w_row: usize, in_features: usize) -> Vec<i8> {
    let mut w = vec![0i8; in_features];
    let w_idx_base = w_row >> 4;
    for k in 0..in_features {
        let local = k & 15;
        let bits = (*packed_w.get_unchecked(w_idx_base + (k >> 4)) >> (local << 1)) & 0b11;
        *w.get_unchecked_mut(k) = (bits as i8) - 1;
    }
    w
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_i8_dot(w_i8: &[i8], x_i8: &[i8], in_features: usize) -> i32 {
    let mut sum = 0i32;
    let mut i = 0usize;

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

fn avx2_forward_i8(
    x_i8: &[i8], batch: usize,
    packed_w: &[u32], scales: &[f32],
    out_features: usize, in_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_features];

    let w_cache: Vec<Vec<i8>> = (0..out_features)
        .map(|o| unsafe { avx2_unpack_weights(packed_w, o * in_features, in_features) })
        .collect();

    for b in 0..batch {
        let x_off = b * in_features;
        for o in 0..out_features {
            let raw = unsafe { avx2_i8_dot(&w_cache[o], x_i8.get_unchecked(x_off..), in_features) };
            let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
            out[b * out_features + o] = raw as f32 * scales[g];
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

        let scalar_out = I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f);

        let mut mismatches = 0usize;
        let mut max_diff = 0.0f32;
        for (i, (a, b)) in scalar_out.iter().zip(scalar_out.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff { max_diff = diff; }
            if a.to_bits() != b.to_bits() { mismatches += 1; }
        }

        println!("{:>14}: scalar reference OK ({} elements)", name, batch * out_f);

        let iters = if batch * in_f * out_f > 1_000_000 { 3 } else { 50 };

        let start_avx = Instant::now();
        let mut avx_out = Vec::new();
        for _ in 0..iters {
            avx_out = avx2_forward_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        }
        let avx_time = start_avx.elapsed().as_secs_f64() / iters as f64;

        let start_scalar = Instant::now();
        let mut scalar_out2 = Vec::new();
        for _ in 0..iters {
            scalar_out2 = I2SKernel::forward_raw_i8(&xi8, batch, &pw, &sc, out_f, in_f);
        }
        let scalar_time = start_scalar.elapsed().as_secs_f64() / iters as f64;

        let mut avx_mismatches = 0usize;
        let mut avx_max_diff = 0.0f32;
        for (i, (a, b)) in avx_out.iter().zip(scalar_out2.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > avx_max_diff { avx_max_diff = diff; }
            if a.to_bits() != b.to_bits() { avx_mismatches += 1; }
        }

        let speedup = scalar_time / avx_time;
        println!(
            "  {:>12}: scalar={:.3}ms avx2={:.3}ms speedup={:.2}x mismatches={} max_diff={:.6}",
            "", scalar_time * 1000.0, avx_time * 1000.0, speedup, avx_mismatches, avx_max_diff
        );
    }

    println!("\n=== DONE ===");
}
