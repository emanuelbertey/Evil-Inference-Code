use std::error::Error;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use xlstm::blocks::kuantgrad::adamw::{AdamWConfig, AdamWState};
use xlstm::blocks::kuantgrad::compress::{compress, decompress};
use std::time::Instant;
use rand::Rng;
use xlstm::blocks::kuantgrad::cuda_kuantgrad::KUANTGRAD_ADAMW_SRC;

pub fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   CUDA KuantGrad AdamW Kernel Test                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    const N: usize = 100000;
    println!("\n  Parameters: {}", N);

    let mut rng = rand::rng();
    
    let mut params = vec![0.0f32; N];
    let mut grads = vec![0.0f32; N];
    for i in 0..N {
        params[i] = rng.random::<f32>() * 2.0 - 1.0;
        grads[i] = rng.random::<f32>() * 0.1 - 0.05;
    }

    // Compress
    let (compressed, n_groups) = compress(&grads);
    println!("  Compressed {} floats to {} bytes ({} groups)", N, compressed.len(), n_groups);

    // CPU baseline
    let cfg = AdamWConfig { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.01 };
    let mut cpu_params = params.clone();
    let mut cpu_state = AdamWState::new(N);
    let decompressed_grads = decompress(&compressed, n_groups, N);
    
    let t0 = Instant::now();
    cpu_state.step(&mut cpu_params, &decompressed_grads, &cfg);
    let t_cpu = t0.elapsed();
    println!("  CPU AdamW: {:.3}ms", t_cpu.as_secs_f64() * 1000.0);

    // GPU 
    println!("  Compiling kernel via NVRTC...");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(KUANTGRAD_ADAMW_SRC)?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("kuantgrad_adamw")?;

    let gpu_params = params.clone();
    let gpu_m = vec![0.0f32; N];
    let gpu_v = vec![0.0f32; N];

    let mut d_params = stream.clone_htod(&gpu_params)?;
    let mut d_m = stream.clone_htod(&gpu_m)?;
    let mut d_v = stream.clone_htod(&gpu_v)?;
    let d_compressed = stream.clone_htod(&compressed)?;

    let t0 = Instant::now();
    let num_threads = n_groups as u32;
    let block_size = 256u32;
    // let grid_size = (num_threads + block_size - 1) / block_size;

    let n_i32 = N as i32;
    let t = 1.0f64;
    let inv_beta1_t = (1.0 - (cfg.beta1 as f64).powf(t)) as f32;
    let inv_beta2_t = (1.0 - (cfg.beta2 as f64).powf(t)) as f32;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_params);
    builder.arg(&mut d_m);
    builder.arg(&mut d_v);
    builder.arg(&d_compressed);
    builder.arg(&n_i32);
    builder.arg(&cfg.lr);
    builder.arg(&cfg.beta1);
    builder.arg(&cfg.beta2);
    builder.arg(&cfg.eps);
    builder.arg(&cfg.wd);
    builder.arg(&inv_beta1_t);
    builder.arg(&inv_beta2_t);

    unsafe {
        builder.launch(LaunchConfig::for_num_elems(num_threads))?;
    }

    let gpu_params_res = stream.clone_dtoh(&d_params)?;
    let gpu_m_res = stream.clone_dtoh(&d_m)?;
    let gpu_v_res = stream.clone_dtoh(&d_v)?;
    let t_gpu = t0.elapsed();
    println!("  CUDA kernel: {:.3}ms", t_gpu.as_secs_f64() * 1000.0);

    // Compare
    let mut max_diff_p = 0.0f32;
    let mut max_diff_m = 0.0f32;
    let mut max_diff_v = 0.0f32;
    for i in 0..N {
        max_diff_p = max_diff_p.max((cpu_params[i] - gpu_params_res[i]).abs());
        max_diff_m = max_diff_m.max((cpu_state.m[i] - gpu_m_res[i]).abs());
        max_diff_v = max_diff_v.max((cpu_state.v[i] - gpu_v_res[i]).abs());
    }

    println!("\n  ── Comparison ──");
    println!("  Max diff params: {:.2e}", max_diff_p);
    println!("  Max diff m:      {:.2e}", max_diff_m);
    println!("  Max diff v:      {:.2e}", max_diff_v);

    if max_diff_p > 1e-4 {
        println!("  ⚠  Mismatch detected!");
    } else {
        println!("  ✓  Results match (within tolerance)");
    }

    Ok(())
}
