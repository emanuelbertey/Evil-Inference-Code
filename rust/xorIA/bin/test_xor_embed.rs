// ─── XOR-Encrypted Kernel Embed ───────────────────────────────────
// Demostración: kernel CUDA en texto plano → encriptado con XOR
// (key 0xAB) → emebebido en binario como .enc → descifrado en
// runtime → compilado con NVRTC → ejecutado vs CPU.
//
// Usage: cargo run --release --bin test_xor_embed

use std::error::Error;
use std::time::Instant;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// Embeber el .enc encriptado (193 bytes)
const KERNEL_ENC: &[u8] = include_bytes!("kernels/demo.enc");
const XOR_KEY: u8 = 0xAB;

fn xor_decrypt(data: &[u8]) -> Vec<u8> {
    data.iter().map(|b| b ^ XOR_KEY).collect()
}

fn cpu_vec_add(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() { a[i] += b[i]; }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   XOR-Encrypted Kernel Embed Demo                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    println!("\n  Encrypted kernel: {} bytes emebebidos", KERNEL_ENC.len());

    let t0 = Instant::now();
    let src_bytes = xor_decrypt(KERNEL_ENC);
    let src = String::from_utf8(src_bytes)?;
    let t_decrypt = t0.elapsed();
    println!("  XOR decrypt:       {:.3}s", t_decrypt.as_secs_f32());
    println!("  Source (decrypted):\n{}", src.trim());

    // ── Test: vector add ────────────────────────────────────────────
    const N: usize = 1024;
    let mut a_host = vec![1.0f32; N];
    let b_host = vec![2.0f32; N];

    // CPU
    let t0 = Instant::now();
    cpu_vec_add(&mut a_host, &b_host);
    let t_cpu = t0.elapsed();

    // CUDA
    let t0 = Instant::now();
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA: {}", e))?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(&src).map_err(|e| format!("NVRTC: {}", e))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("Module: {}", e))?;
    let kernel = module.load_function("vec_add").map_err(|e| format!("Kernel: {}", e))?;

    let mut d_a = stream.clone_htod(&a_host).map_err(|e| format!("H2D a: {}", e))?;
    let d_b = stream.clone_htod(&b_host).map_err(|e| format!("H2D b: {}", e))?;
    let n_i32 = N as i32;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_a);
    builder.arg(&d_b);
    builder.arg(&n_i32);
    unsafe { builder.launch(LaunchConfig::for_num_elems(N as u32)).map_err(|e| format!("Launch: {}", e))?; }

    let gpu_out: Vec<f32> = stream.clone_dtoh(&d_a).map_err(|e| format!("D2H: {}", e))?;
    let t_gpu = t0.elapsed();

    // Compare
    let max_diff = a_host.iter().zip(gpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("\n  ── Vector Add (N={}) ──", N);
    println!("  CPU: {:.6}s", t_cpu.as_secs_f32());
    println!("  GPU: {:.6}s", t_gpu.as_secs_f32());
    println!("  Max diff: {:.2e}", max_diff);
    if max_diff < 1e-5 {
        println!("  ✓  Results match");
    } else {
        println!("  ⚠  Mismatch!");
    }

    Ok(())
}
