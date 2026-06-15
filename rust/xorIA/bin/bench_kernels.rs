use std::time::Instant;
use xlstm::blocks::bitlinear::kernel::{I2SKernel, I2STile16Kernel, KernelKind, TL1Kernel, TL2Kernel};
use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearConfig, BitLinearInferenceState};
use burn::module::Module;
use burn_flex::Flex;

type MyBackend = Flex<f32>;

fn bench_f32_matmul(x: &[f32], w: &[f32], batch: usize, out: usize, inp: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; batch * out];
    for b in 0..batch {
        for o in 0..out {
            let mut sum = 0.0f32;
            for i in 0..inp {
                sum += x[b * inp + i] * w[o * inp + i];
            }
            result[b * out + o] = sum;
        }
    }
    result
}

fn bench_bitlinear(state: &BitLinearInferenceState, x: &[f32], iters: usize) -> f32 {
    for _ in 0..50 { let _ = state.forward_raw(x, 1); }
    let t0 = Instant::now();
    for _ in 0..iters { let _ = state.forward_raw(x, 1); }
    t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
}

fn bench_raw_kernel(kernel_name: &str, x: &[f32], batch: usize, out: usize, inp: usize, iters: usize) -> f32 {
    let mut w_t: Vec<f32> = Vec::with_capacity(out * inp);
    for _ in 0..out * inp {
        let r: f32 = rand::random();
        if r < 0.33 { w_t.push(-1.0); } else if r < 0.66 { w_t.push(0.0); } else { w_t.push(1.0); }
    }
    let w_f32: Vec<f32> = w_t.iter().map(|v| v * 0.1).collect();
    let scales = vec![0.1f32; (out * inp / 128).max(1)];

    match kernel_name {
        "I2S" => {
            let packed = I2SKernel::pack_weights(&w_t);
            for _ in 0..50 { let _ = I2SKernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            let t0 = Instant::now();
            for _ in 0..iters { let _ = I2SKernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
        }
        "TL1" => {
            let packed = TL1Kernel::pack_weights(&w_t);
            for _ in 0..50 { let _ = TL1Kernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            let t0 = Instant::now();
            for _ in 0..iters { let _ = TL1Kernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
        }
        "TL2" => {
            let packed = TL2Kernel::pack_weights(&w_t);
            for _ in 0..50 { let _ = TL2Kernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            let t0 = Instant::now();
            for _ in 0..iters { let _ = TL2Kernel::forward_raw(x, batch, &packed, &scales, out, inp); }
            t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
        }
        "Tile16" => {
            let w_i8 = I2STile16Kernel::quantize_to_i8(&w_t);
            for _ in 0..50 { let _ = I2STile16Kernel::forward_raw(x, batch, &w_i8, &scales, out, inp); }
            let t0 = Instant::now();
            for _ in 0..iters { let _ = I2STile16Kernel::forward_raw(x, batch, &w_i8, &scales, out, inp); }
            t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
        }
        "f32" => {
            for _ in 0..50 { let _ = bench_f32_matmul(x, &w_f32, batch, out, inp); }
            let t0 = Instant::now();
            for _ in 0..iters { let _ = bench_f32_matmul(x, &w_f32, batch, out, inp); }
            t0.elapsed().as_secs_f32() * 1000.0 / iters as f32
        }
        _ => unreachable!()
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     BitLinear Kernel Benchmark                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let batch = 1;
    let iters = 200;

    let dims: Vec<(usize, usize)> = vec![
        (512, 512),
        (128, 512),
        (512, 640),
        (640, 512),
        (16000, 512),
    ];

    println!("─── Parte 1: Pesos Random ───\n");

    for (out, inp) in &dims {
        let x: Vec<f32> = (0..batch * inp).map(|_| rand::random::<f32>() * 0.1).collect();
        let t_i2s = bench_raw_kernel("I2S", &x, batch, *out, *inp, iters);
        let t_tile16 = bench_raw_kernel("Tile16", &x, batch, *out, *inp, iters);
        let t_tl1 = bench_raw_kernel("TL1", &x, batch, *out, *inp, iters);
        let t_tl2 = bench_raw_kernel("TL2", &x, batch, *out, *inp, iters);
        let t_f32 = bench_raw_kernel("f32", &x, batch, *out, *inp, iters);
        println!("{}x{}:", out, inp);
        println!("  f32:     {:.3} ms", t_f32);
        println!("  I2S:     {:.3} ms  ({:.1}x vs f32)", t_i2s, t_f32 / t_i2s.max(0.001));
        println!("  Tile16:  {:.3} ms  ({:.1}x vs f32)", t_tile16, t_f32 / t_tile16.max(0.001));
        println!("  TL1:     {:.3} ms  ({:.1}x vs f32)", t_tl1, t_f32 / t_tl1.max(0.001));
        println!("  TL2:     {:.3} ms  ({:.1}x vs f32)", t_tl2, t_f32 / t_tl2.max(0.001));
        println!();
    }

    let model_file = "transformer_bit2.mpk";
    if !std::path::Path::new(model_file).exists() {
        println!("No se encontro {}. Fin.", model_file);
        return;
    }

    println!("─── Parte 2: Modelo Real ───\n");
    println!("Exportando kernels del modelo...");

    let device = Default::default();
    let d_model: usize = 512;
    let num_heads: usize = 8;
    let head_dim = d_model / num_heads;
    let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;
    let kv_groups = 4;

    let layer_specs: Vec<(&str, usize, usize)> = vec![
        ("Q_proj", num_heads * head_dim, d_model),
        ("K_proj", kv_groups * head_dim, d_model),
        ("V_proj", kv_groups * head_dim, d_model),
        ("o_proj", d_model, num_heads * head_dim),
        ("gate_up", 2 * ffn_dim, d_model),
        ("down", d_model, ffn_dim),
    ];

    let layer_count = 6;
    for li in 0..layer_count {
        println!("  Layer {}:", li);
        for (name, out, inp) in &layer_specs {
            let config = BitLinearConfig { in_features: *inp, out_features: *out, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 };
            let mut layer: BitLinear<MyBackend> = config.init(&device);
            let state = layer.export_inference_layer(&device, KernelKind::Tile16);
            layer.release_weights(&device);
            let x: Vec<f32> = (0..*inp).map(|_| rand::random::<f32>() * 0.1).collect();
            let t = bench_bitlinear(&state, &x, iters);
            println!("    {:>7} {}x{}: {:.3} ms", name, out, inp, t);
        }
        println!();
    }

    println!("  Head (16000x512):");
    let config = BitLinearConfig { in_features: 512, out_features: 16000, bias: false, activation_bits: 8, rms_norm_eps: 1e-5 };
    let mut layer: BitLinear<MyBackend> = config.init(&device);
    let state = layer.export_inference_layer(&device, KernelKind::I2S);
    layer.release_weights(&device);
    let x: Vec<f32> = (0..512).map(|_| rand::random::<f32>() * 0.1).collect();
    let t = bench_bitlinear(&state, &x, iters);
    println!("    {:>7} {:.3} ms\n", "I2S", t);
}
