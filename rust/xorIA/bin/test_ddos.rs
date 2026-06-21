// ─── DDoS: GPU Throughput Stress Test ─────────────────────────────
// Lanza N kernels I2S concurrentes en streams paralelos para medir
// throughput máximo y saturar la GPU.
//
// Usage:
//   cargo run --release --bin test_ddos

use std::error::Error;
use std::time::Instant;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const I2S_KERNEL_SRC: &str = r#"
extern "C" __global__ void i2s_matmul(
    const float* __restrict__ input,
    const unsigned int* __restrict__ weights,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int K_packed = (K + 15) / 16;
    float sum = 0.0f;
    for (int kp = 0; kp < K_packed; kp++) {
        unsigned int packed_w = weights[col * K_packed + kp];
        int base = row * K + (kp << 4);
        for (int j = 0; j < 16; j++) {
            int k_idx = (kp << 4) + j;
            if (k_idx >= K) break;
            int ternary = (packed_w >> (j << 1)) & 3;
            float w_val = (ternary == 0) ? -1.0f : ((ternary == 2) ? 1.0f : 0.0f);
            sum += input[base + j] * w_val;
        }
    }
    output[row * N + col] = sum * scales[col];
}
"#;

fn gen_test_data(M: usize, N: usize, K: usize) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut rng = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut rand = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let input: Vec<f32> = (0..M * K).map(|_| ((rand() as f32) * (1.0 / 18446744073709551615.0) - 0.5) * 2.0).collect();
    let K_packed = (K + 15) / 16;
    let mut weights = vec![0u32; N * K_packed];
    for w in &mut weights {
        let mut packed = 0u32;
        for j in 0..16 { packed |= ((rand() % 3) as u32) << (j * 2); }
        *w = packed;
    }
    let scales: Vec<f32> = (0..N).map(|_| ((rand() as f32) * (1.0 / 18446744073709551615.0) - 0.5) * 2.0).collect();
    (input, weights, scales)
}

pub fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   DDoS: GPU Throughput Stress Test                      ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    const M: usize = 256;
    const N: usize = 768;
    const K: usize = 768;
    const STREAMS: usize = 4;
    const ITER: usize = 100;

    println!("\n  Matrix: M={}, N={}, K={}", M, N, K);
    println!("  Streams: {}, Iterations: {}", STREAMS, ITER);

    let (input, weights, scales) = gen_test_data(M, N, K);
    let total_flops = 2.0 * M as f64 * N as f64 * K as f64;

    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA ctx: {}", e))?;
    let ptx = compile_ptx(I2S_KERNEL_SRC).map_err(|e| format!("NVRTC: {}", e))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("Module: {}", e))?;
    let kernel = module.load_function("i2s_matmul").map_err(|e| format!("Kernel: {}", e))?;

    let m_i32 = M as i32; let n_i32 = N as i32; let k_i32 = K as i32;
    let block_x = 16u32; let block_y = 16u32;
    let grid_x = (N as u32 + block_x - 1) / block_x;
    let grid_y = (M as u32 + block_y - 1) / block_y;
    let launch_cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (block_x, block_y, 1), shared_mem_bytes: 0 };

    let streams: Vec<_> = (0..STREAMS).map(|_| ctx.new_stream().unwrap()).collect();
    let mut dev_bufs: Vec<_> = streams.iter().map(|s| {
        let di = s.clone_htod(&input).unwrap();
        let dw = s.clone_htod(&weights).unwrap();
        let ds = s.clone_htod(&scales).unwrap();
        let mut dout = s.alloc_zeros::<f32>(M * N).unwrap();
        (di, dw, ds, dout)
    }).collect();

    // Warmup
    for s in 0..STREAMS {
        let mut b = streams[s].launch_builder(&kernel);
        let (ref di, ref dw, ref ds, ref mut dout) = dev_bufs[s];
        b.arg(di); b.arg(dw); b.arg(ds); b.arg(dout);
        b.arg(&m_i32); b.arg(&n_i32); b.arg(&k_i32);
        unsafe { b.launch(launch_cfg).unwrap(); }
    }
    for s in 0..STREAMS { streams[s].synchronize().unwrap(); }

    // Benchmark
    let t0 = Instant::now();
    for _ in 0..ITER {
        for s in 0..STREAMS {
            let mut b = streams[s].launch_builder(&kernel);
            let (ref di, ref dw, ref ds, ref mut dout) = dev_bufs[s];
            b.arg(di); b.arg(dw); b.arg(ds); b.arg(dout);
            b.arg(&m_i32); b.arg(&n_i32); b.arg(&k_i32);
            unsafe { b.launch(launch_cfg).unwrap(); }
        }
    }
    for s in 0..STREAMS { streams[s].synchronize().unwrap(); }
    let elapsed = t0.elapsed();

    let total_ops = total_flops * (ITER * STREAMS) as f64;
    let tflops = total_ops / elapsed.as_secs_f64() / 1e12;

    println!("\n  ── Results ──");
    println!("  Elapsed:   {:.3}s", elapsed.as_secs_f32());
    println!("  Throughput: {:.2} TFLOP/s", tflops);
    println!("  Ops total:  {:.2e}", total_ops);

    Ok(())
}
