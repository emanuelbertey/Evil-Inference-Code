// ─── CUDA I2S Kernel Test ─────────────────────────────────────────
// Custom I2S matmul kernel compiled at runtime via NVRTC.
// Compares GPU result with CPU reference.
//
// Usage (standalone):
//   cargo run --release --bin test_cuda_kernel

use std::error::Error;
use cudarc::driver::PushKernelArg;

// ─── CUDA C++ kernel source (compiled at runtime via NVRTC) ────────
//
// I2S ternary matmul: output[M,N] = (input[M,K] @ weights[N,K]) * scales[N]
//   weights packed as: 16 ternary values per u32, 2 bits each
//   encoding: 0→-1, 1→0, 2→+1, 3→0
const I2S_KERNEL_SRC: &str = r#"
extern "C" __global__ void i2s_matmul(
    const float* input,
    const unsigned int* weights,
    const float* scales,
    float* output,
    int M, int N, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;

    int row = tid / N;
    int col = tid % N;
    int K_packed = (K + 15) / 16;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        int packed_idx = col * K_packed + (i >> 4);
        int bit_pos = (i & 15) << 1;
        int ternary = (weights[packed_idx] >> bit_pos) & 3;
        float w_val;
        if (ternary == 0) w_val = -1.0f;
        else if (ternary == 1) w_val = 0.0f;
        else if (ternary == 2) w_val = 1.0f;
        else w_val = 0.0f;
        sum += input[row * K + i] * w_val;
    }
    output[tid] = sum * scales[col];
}
"#;

// ─── CPU reference (same logic, pure Rust) ─────────────────────────
fn cpu_i2s_matmul(
    input: &[f32],
    weights: &[u32],
    scales: &[f32],
    M: usize, N: usize, K: usize,
) -> Vec<f32> {
    let K_packed = (K + 15) / 16;
    let mut out = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0f32;
            for i in 0..K {
                let packed_idx = col * K_packed + (i >> 4);
                let bit_pos = ((i & 15) << 1) as u32;
                let ternary = (weights[packed_idx] >> bit_pos) & 3;
                let w_val = match ternary {
                    0 => -1.0,
                    2 => 1.0,
                    _ => 0.0,
                };
                sum += input[row * K + i] * w_val;
            }
            out[row * N + col] = sum * scales[col];
        }
    }
    out
}

// ─── Helper: generate random test data ─────────────────────────────
fn gen_test_data(
    M: usize, N: usize, K: usize,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut rng = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let input: Vec<f32> = (0..M * K).map(|_| {
        let v = rand();
        ((v as f32) * (1.0 / 18446744073709551615.0) - 0.5) * 2.0
    }).collect();
    let K_packed = (K + 15) / 16;
    let mut weights = vec![0u32; N * K_packed];
    for w in &mut weights {
        let mut packed = 0u32;
        for j in 0..16 {
            let val = rand() % 3; // 0→-1, 1→0, 2→+1
            packed |= (val as u32) << (j * 2);
        }
        *w = packed;
    }
    let scales: Vec<f32> = (0..N).map(|_| {
        let v = rand();
        ((v as f32) * (1.0 / 18446744073709551615.0) - 0.5) * 2.0
    }).collect();

    (input, weights, scales)
}

// ─── Main entry point (works as both binary and module) ────────────
pub fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   CUDA I2S Kernel — NVRTC Runtime Compilation           ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    const M: usize = 32;
    const N: usize = 64;
    const K: usize = 128;
    const K_PACKED: usize = (K + 15) / 16;

    println!("\n  Matrix: M={}, N={}, K={} (packed={})", M, N, K, K_PACKED);

    // ─── Generate test data ─────────────────────────────────────────
    let (input, weights, scales) = gen_test_data(M, N, K);
    println!("  Input:  {} floats ({} KB)", input.len(), input.len() * 4 / 1024);
    println!("  Weights: {} u32s ({} KB)", weights.len(), weights.len() * 4 / 1024);
    println!("  Scales: {} floats", scales.len());

    // ─── CPU reference ──────────────────────────────────────────────
    use std::time::Instant;
    let t0 = Instant::now();
    let cpu_out = cpu_i2s_matmul(&input, &weights, &scales, M, N, K);
    let t_cpu = t0.elapsed();
    println!("\n  CPU reference: {:.3}s", t_cpu.as_secs_f32());

    // ─── CUDA I2S kernel ────────────────────────────────────────────
    println!("\n  Compiling I2S kernel via NVRTC...");
    let t0 = Instant::now();

    use cudarc::driver::{CudaContext, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;

    let ctx = CudaContext::new(0)
        .map_err(|e| format!("Failed to create CUDA context: {}", e))?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(I2S_KERNEL_SRC)
        .map_err(|e| format!("NVRTC compilation failed: {}", e))?;
    let module = ctx.load_module(ptx)
        .map_err(|e| format!("Failed to load module: {}", e))?;
    let kernel = module.load_function("i2s_matmul")
        .map_err(|e| format!("Failed to get kernel: {}", e))?;

    let d_input = stream.clone_htod(&input)
        .map_err(|e| format!("Failed to copy input: {}", e))?;
    let d_weights = stream.clone_htod(&weights)
        .map_err(|e| format!("Failed to copy weights: {}", e))?;
    let d_scales = stream.clone_htod(&scales)
        .map_err(|e| format!("Failed to copy scales: {}", e))?;
    let mut d_output = stream.alloc_zeros::<f32>(M * N)
        .map_err(|e| format!("Failed to alloc output: {}", e))?;

    let t_compile = t0.elapsed();
    println!("  Compilation + transfer: {:.3}s", t_compile.as_secs_f32());

    // Launch kernel
    let num_threads = (M * N) as u32;
    let block_size = 256u32;
    let grid_size = (num_threads + block_size - 1) / block_size;
    let m_i32 = M as i32;
    let n_i32 = N as i32;
    let k_i32 = K as i32;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&d_input);
    builder.arg(&d_weights);
    builder.arg(&d_scales);
    builder.arg(&mut d_output);
    builder.arg(&m_i32);
    builder.arg(&n_i32);
    builder.arg(&k_i32);

    let t0 = Instant::now();
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(num_threads))
            .map_err(|e| format!("Kernel launch failed: {}", e))?;
    }

    // Read back
    let gpu_out = stream
        .clone_dtoh(&d_output)
        .map_err(|e| format!("Failed to read back: {}", e))?;
    let t_gpu = t0.elapsed();
    println!("  CUDA kernel (no sync): {:.3}s", t_gpu.as_secs_f32());

    // ─── Compare ────────────────────────────────────────────────────
    let max_diff = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / cpu_out.len() as f64;

    println!("\n  ── Comparison ──");
    println!("  Max diff:  {:.2e}", max_diff);
    println!("  Mean diff: {:.2e}", mean_diff);

    if max_diff > 1e-5 {
        println!("  ⚠  Mismatch detected!");
    } else {
        println!("  ✓  Results match (within tolerance)");
    }

    println!("\n  ── Kernel stats ──");
    println!("  Grid:  ({}, 1, 1)", grid_size);
    println!("  Block: ({}, 1, 1)", block_size);

    Ok(())
}
