use std::time::Instant;
use std::hint::black_box;
use xlstm::blocks::bitlinear::kernel::{I2SKernel, GROUP_SIZE};

fn naive_i8_dot(x: &[f32], w: &[i8], scales: &[f32], o: usize, in_features: usize) -> f32 {
    let mut sum = 0.0f32;
    let w_row = o * in_features;
    unsafe {
        let mut i = 0usize;
        while i < in_features {
            let wt = *w.get_unchecked(w_row + i);
            let xv = *x.get_unchecked(i);
            if wt == 1 { sum += xv; } else if wt == -1 { sum -= xv; }
            i += 1;
        }
    }
    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
    sum * unsafe { *scales.get_unchecked(g) }
}

fn naive_i8_unrolled(x: &[f32], w: &[i8], scales: &[f32], o: usize, in_features: usize) -> f32 {
    let mut sum = 0.0f32;
    let w_row = o * in_features;
    unsafe {
        let mut i = 0usize;
        while i + 7 < in_features {
            let w0 = *w.get_unchecked(w_row + i) as i32;
            let w1 = *w.get_unchecked(w_row + i + 1) as i32;
            let w2 = *w.get_unchecked(w_row + i + 2) as i32;
            let w3 = *w.get_unchecked(w_row + i + 3) as i32;
            let w4 = *w.get_unchecked(w_row + i + 4) as i32;
            let w5 = *w.get_unchecked(w_row + i + 5) as i32;
            let w6 = *w.get_unchecked(w_row + i + 6) as i32;
            let w7 = *w.get_unchecked(w_row + i + 7) as i32;
            let x0 = *x.get_unchecked(i);
            let x1 = *x.get_unchecked(i + 1);
            let x2 = *x.get_unchecked(i + 2);
            let x3 = *x.get_unchecked(i + 3);
            let x4 = *x.get_unchecked(i + 4);
            let x5 = *x.get_unchecked(i + 5);
            let x6 = *x.get_unchecked(i + 6);
            let x7 = *x.get_unchecked(i + 7);
            sum += (w0 as f32) * x0 + (w1 as f32) * x1 + (w2 as f32) * x2 + (w3 as f32) * x3
                 + (w4 as f32) * x4 + (w5 as f32) * x5 + (w6 as f32) * x6 + (w7 as f32) * x7;
            i += 8;
        }
        while i < in_features {
            let wt = *w.get_unchecked(w_row + i);
            if wt == 1 { sum += *x.get_unchecked(i); } else if wt == -1 { sum -= *x.get_unchecked(i); }
            i += 1;
        }
    }
    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
    sum * unsafe { *scales.get_unchecked(g) }
}

fn branchless_dot(x: &[f32], w: &[i8], scales: &[f32], o: usize, in_features: usize) -> f32 {
    let mut sum = 0.0f32;
    let w_row = o * in_features;
    unsafe {
        let mut i = 0usize;
        while i < in_features {
            sum += *w.get_unchecked(w_row + i) as f32 * *x.get_unchecked(i);
            i += 1;
        }
    }
    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
    sum * unsafe { *scales.get_unchecked(g) }
}

fn branchless_unrolled(x: &[f32], w: &[i8], scales: &[f32], o: usize, in_features: usize) -> f32 {
    let mut sum = 0.0f32;
    let w_row = o * in_features;
    unsafe {
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
            let x0 = *x.get_unchecked(i);
            let x1 = *x.get_unchecked(i + 1);
            let x2 = *x.get_unchecked(i + 2);
            let x3 = *x.get_unchecked(i + 3);
            let x4 = *x.get_unchecked(i + 4);
            let x5 = *x.get_unchecked(i + 5);
            let x6 = *x.get_unchecked(i + 6);
            let x7 = *x.get_unchecked(i + 7);
            sum += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7;
            i += 8;
        }
        while i < in_features {
            sum += *w.get_unchecked(w_row + i) as f32 * *x.get_unchecked(i);
            i += 1;
        }
    }
    let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
    sum * unsafe { *scales.get_unchecked(g) }
}

const TILE: usize = 8;

fn branchless_tiled(x: &[f32], w: &[i8], scales: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    unsafe {
        let mut o = 0usize;
        while o + TILE - 1 < out_features {
            let mut acc = [0.0f32; TILE];
            let mut i = 0usize;
            while i < in_features {
                let xv = *x.get_unchecked(i);
                if xv != 0.0 {
                    acc[0] += *w.get_unchecked(0 * in_features + i) as f32 * xv;
                    acc[1] += *w.get_unchecked(1 * in_features + i) as f32 * xv;
                    acc[2] += *w.get_unchecked(2 * in_features + i) as f32 * xv;
                    acc[3] += *w.get_unchecked(3 * in_features + i) as f32 * xv;
                    acc[4] += *w.get_unchecked(4 * in_features + i) as f32 * xv;
                    acc[5] += *w.get_unchecked(5 * in_features + i) as f32 * xv;
                    acc[6] += *w.get_unchecked(6 * in_features + i) as f32 * xv;
                    acc[7] += *w.get_unchecked(7 * in_features + i) as f32 * xv;
                }
                i += 1;
            }
            for k in 0..TILE {
                let oi = o + k;
                let g = (oi * in_features / GROUP_SIZE).min(scales.len() - 1);
                *out.get_unchecked_mut(oi) = acc[k] * *scales.get_unchecked(g);
            }
            o += TILE;
        }
        while o < out_features {
            let mut sum = 0.0f32;
            let w_row = o * in_features;
            let mut i = 0usize;
            while i < in_features {
                sum += *w.get_unchecked(w_row + i) as f32 * *x.get_unchecked(i);
                i += 1;
            }
            let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
            *out.get_unchecked_mut(o) = sum * *scales.get_unchecked(g);
            o += 1;
        }
    }
    out
}

const TILE16: usize = 16;

fn branchless_tiled16(x: &[f32], w: &[i8], scales: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    unsafe {
        let mut o = 0usize;
        while o + TILE16 - 1 < out_features {
            let mut acc = [0.0f32; TILE16];
            let mut i = 0usize;
            while i < in_features {
                let xv = *x.get_unchecked(i);
                if xv != 0.0 {
                    acc[0]  += *w.get_unchecked(0  * in_features + i) as f32 * xv;
                    acc[1]  += *w.get_unchecked(1  * in_features + i) as f32 * xv;
                    acc[2]  += *w.get_unchecked(2  * in_features + i) as f32 * xv;
                    acc[3]  += *w.get_unchecked(3  * in_features + i) as f32 * xv;
                    acc[4]  += *w.get_unchecked(4  * in_features + i) as f32 * xv;
                    acc[5]  += *w.get_unchecked(5  * in_features + i) as f32 * xv;
                    acc[6]  += *w.get_unchecked(6  * in_features + i) as f32 * xv;
                    acc[7]  += *w.get_unchecked(7  * in_features + i) as f32 * xv;
                    acc[8]  += *w.get_unchecked(8  * in_features + i) as f32 * xv;
                    acc[9]  += *w.get_unchecked(9  * in_features + i) as f32 * xv;
                    acc[10] += *w.get_unchecked(10 * in_features + i) as f32 * xv;
                    acc[11] += *w.get_unchecked(11 * in_features + i) as f32 * xv;
                    acc[12] += *w.get_unchecked(12 * in_features + i) as f32 * xv;
                    acc[13] += *w.get_unchecked(13 * in_features + i) as f32 * xv;
                    acc[14] += *w.get_unchecked(14 * in_features + i) as f32 * xv;
                    acc[15] += *w.get_unchecked(15 * in_features + i) as f32 * xv;
                }
                i += 1;
            }
            for k in 0..TILE16 {
                let oi = o + k;
                let g = (oi * in_features / GROUP_SIZE).min(scales.len() - 1);
                *out.get_unchecked_mut(oi) = acc[k] * *scales.get_unchecked(g);
            }
            o += TILE16;
        }
        while o < out_features {
            let mut sum = 0.0f32;
            let w_row = o * in_features;
            let mut i = 0usize;
            while i < in_features {
                sum += *w.get_unchecked(w_row + i) as f32 * *x.get_unchecked(i);
                i += 1;
            }
            let g = (o * in_features / GROUP_SIZE).min(scales.len() - 1);
            *out.get_unchecked_mut(o) = sum * *scales.get_unchecked(g);
            o += 1;
        }
    }
    out
}

fn f32_unrolled(x: &[f32], w: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    unsafe {
        for o in 0..out_features {
            let mut sum = 0.0f32;
            let w_row = o * in_features;
            let mut i = 0usize;
            while i + 7 < in_features {
                sum += *w.get_unchecked(w_row + i)     * *x.get_unchecked(i)
                     + *w.get_unchecked(w_row + i + 1) * *x.get_unchecked(i + 1)
                     + *w.get_unchecked(w_row + i + 2) * *x.get_unchecked(i + 2)
                     + *w.get_unchecked(w_row + i + 3) * *x.get_unchecked(i + 3)
                     + *w.get_unchecked(w_row + i + 4) * *x.get_unchecked(i + 4)
                     + *w.get_unchecked(w_row + i + 5) * *x.get_unchecked(i + 5)
                     + *w.get_unchecked(w_row + i + 6) * *x.get_unchecked(i + 6)
                     + *w.get_unchecked(w_row + i + 7) * *x.get_unchecked(i + 7);
                i += 8;
            }
            while i < in_features {
                sum += *w.get_unchecked(w_row + i) * *x.get_unchecked(i);
                i += 1;
            }
            *out.get_unchecked_mut(o) = sum;
        }
    }
    out
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Kernel Sum Benchmark — Solo Sumas vs I2S Packed          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let iters = 2000;
    let warmup = 100;
    let dims: Vec<(usize, usize)> = vec![
        (512, 512),
        (128, 512),
        (512, 640),
        (640, 512),
        (16000, 512),
    ];

    for (out, inp) in &dims {
        println!("─── {}x{} ───", out, inp);

        let x: Vec<f32> = (0..*inp).map(|_| rand::random::<f32>() * 0.1).collect();
        let scales = vec![0.1f32; (*out * *inp / GROUP_SIZE).max(1)];

        let mut w_f32: Vec<f32> = Vec::with_capacity(out * inp);
        let mut w_i8: Vec<i8> = Vec::with_capacity(out * inp);
        for _ in 0..*out * *inp {
            let r: f32 = rand::random();
            let v: i8 = if r < 0.33 { -1 } else if r < 0.66 { 0 } else { 1 };
            w_f32.push(v as f32 * 0.1);
            w_i8.push(v);
        }

        let packed = I2SKernel::pack_weights(&w_f32);

        // f32
        for _ in 0..warmup { black_box(f32_unrolled(&x, &w_f32, *out, *inp)); }
        let t0 = Instant::now();
        for _ in 0..iters { black_box(f32_unrolled(&x, &w_f32, *out, *inp)); }
        let t_f32 = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // I2S
        for _ in 0..warmup { black_box(I2SKernel::forward_raw(&x, 1, &packed, &scales, *out, *inp)); }
        let t0 = Instant::now();
        for _ in 0..iters { black_box(I2SKernel::forward_raw(&x, 1, &packed, &scales, *out, *inp)); }
        let t_i2s = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // naive_i8
        for _ in 0..warmup {
            for o in 0..*out { black_box(naive_i8_dot(&x, &w_i8, &scales, o, *inp)); }
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for o in 0..*out { black_box(naive_i8_dot(&x, &w_i8, &scales, o, *inp)); }
        }
        let t_naive = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // naive_8x
        for _ in 0..warmup {
            for o in 0..*out { black_box(naive_i8_unrolled(&x, &w_i8, &scales, o, *inp)); }
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for o in 0..*out { black_box(naive_i8_unrolled(&x, &w_i8, &scales, o, *inp)); }
        }
        let t_naive8 = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // branchless
        for _ in 0..warmup {
            for o in 0..*out { black_box(branchless_dot(&x, &w_i8, &scales, o, *inp)); }
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for o in 0..*out { black_box(branchless_dot(&x, &w_i8, &scales, o, *inp)); }
        }
        let t_br = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // br8
        for _ in 0..warmup {
            for o in 0..*out { black_box(branchless_unrolled(&x, &w_i8, &scales, o, *inp)); }
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for o in 0..*out { black_box(branchless_unrolled(&x, &w_i8, &scales, o, *inp)); }
        }
        let t_br8 = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // tile_8
        for _ in 0..warmup { black_box(branchless_tiled(&x, &w_i8, &scales, *out, *inp)); }
        let t0 = Instant::now();
        for _ in 0..iters { black_box(branchless_tiled(&x, &w_i8, &scales, *out, *inp)); }
        let t_tile8 = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        // tile_16
        for _ in 0..warmup { black_box(branchless_tiled16(&x, &w_i8, &scales, *out, *inp)); }
        let t0 = Instant::now();
        for _ in 0..iters { black_box(branchless_tiled16(&x, &w_i8, &scales, *out, *inp)); }
        let t_tile16 = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

        println!("  f32_8x:     {:.3} ms", t_f32);
        println!("  I2S_pack:   {:.3} ms  ({:.2}x f32)", t_i2s, t_f32 / t_i2s);
        println!("  naive_i8:   {:.3} ms  ({:.2}x f32)  branch peso", t_naive, t_f32 / t_naive);
        println!("  naive_8x:   {:.3} ms  ({:.2}x f32)  unroll 8 branch", t_naive8, t_f32 / t_naive8);
        println!("  branchless: {:.3} ms  ({:.2}x f32)  w*x sin branch", t_br, t_f32 / t_br);
        println!("  br8:        {:.3} ms  ({:.2}x f32)  unroll 8 branchless", t_br8, t_f32 / t_br8);
        println!("  tile_8:     {:.3} ms  ({:.2}x f32)  reutiliza x 8 filas", t_tile8, t_f32 / t_tile8);
        println!("  tile_16:    {:.3} ms  ({:.2}x f32)  reutiliza x 16 filas", t_tile16, t_f32 / t_tile16);
        println!();
    }
}
