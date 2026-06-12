use burn::prelude::*;
use burn::tensor::{Tensor, Distribution, TensorData};
use burn::backend::ndarray::NdArray;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_autodiff::Autodiff;
use xlstm::blocks::mlstm::layer::{MLSTMLayer, MLSTMLayerConfig};
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::nn::loss::MseLoss;
use burn::record::{BinBytesRecorder, Recorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

// Definimos los dos backends a comparar
type CpuBackend = Autodiff<NdArray<f32>>;
type GpuBackend = Autodiff<Wgpu<f32, i32>>;

fn run_copy_task<B: AutodiffBackend>(
    device: &B::Device,
    config: &MLSTMLayerConfig,
    fixed_input: TensorData,
    steps: usize,
    lr: f64,
    backend_name: &str,
    initial_record: Option<<MLSTMLayer<B> as Module<B>>::Record>,
) -> (f32, Vec<f32>) {
    let mut layer: MLSTMLayer<B> = if let Some(record) = initial_record {
        config.init(device).load_record(record)
    } else {
        config.init(device)
    };

    let mut optim = AdamConfig::new().init();
    let input = Tensor::<B, 3>::from_data(fixed_input, device);
    let mut final_loss = 0.0;

    for i in 1..=steps {
        let output = layer.forward(input.clone());
        let loss = MseLoss::new().forward(output, input.clone(), burn::nn::loss::Reduction::Mean);
        
        // Extraer valor de loss
        final_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &layer);
        layer = optim.step(lr, layer, grads_params);

        if i % 20 == 0 || i == 1 {
            println!("  [{}] Step {:3}: Loss {:.8}", backend_name, i, final_loss);
        }
    }

    let final_output = layer.forward(input).into_data().as_slice::<f32>().unwrap().to_vec();
    (final_loss, final_output)
}

fn main() {
    println!("=== BURN DEVICE COMPARISON: CPU (NdArray) vs GPU (WGPU) ===");
    
    let cpu_device = Default::default();
    let gpu_device = WgpuDevice::default();
    
    let steps = 100;
    let lr = 1.2e-3;
    let seq_len = 8;
    let embedding_dim = 16;

    let config = MLSTMLayerConfig::new(embedding_dim)
        .with_num_heads(4)
        .with_conv1d_kernel_size(4)
        .with_qkv_proj_blocksize(4)
        .with_proj_factor(2.0)
        .with_bias(true)
        .with_context_length(seq_len);

    // 1. Inicializar en CPU para extraer los pesos iniciales exactos
    let cpu_initial_model: MLSTMLayer<CpuBackend> = config.init(&cpu_device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    
    // Guardar record de CPU a bytes
    let record_bytes = recorder.record(cpu_initial_model.clone().into_record(), ()).expect("Failed to export record");
    
    // Cargar bytes en Record de GPU
    let gpu_record = recorder.load(record_bytes.clone(), &gpu_device).expect("Failed to load record into GPU");

    // 2. Generar Input fijo
    let input_data = Tensor::<CpuBackend, 3>::random(
        [1, seq_len, embedding_dim],
        Distribution::Normal(0.0, 1.0),
        &cpu_device
    ).into_data();

    // 3. Ejecutar en CPU (Sincronizado)
    println!("Iniciando Test en CPU...");
    let cpu_initial_record = recorder.load(record_bytes, &cpu_device).expect("Failed to reload record");
    let (cpu_loss, cpu_out) = run_copy_task::<CpuBackend>(
        &cpu_device,
        &config,
        input_data.clone(),
        steps,
        lr,
        "CPU-Sync",
        Some(cpu_initial_record),
    );

    println!("\n-------------------------------------------");

    // 4. Ejecutar en GPU
    println!("Iniciando Test en GPU (WebGPU)...");
    let (gpu_loss, gpu_out) = run_copy_task::<GpuBackend>(
        &gpu_device,
        &config,
        input_data,
        steps,
        lr,
        "GPU",
        Some(gpu_record),
    );

    // 5. Comparar resultados
    println!("\n=== COMPARATIVA FINAL ===");
    println!("Loss Final CPU: {:.8}", cpu_loss);
    println!("Loss Final GPU: {:.8}", gpu_loss);
    
    let diff_loss = (cpu_loss - gpu_loss).abs();
    println!("Diferencia de Loss: {:.10}", diff_loss);

    let mut max_diff_out = 0.0f32;
    for i in 0..cpu_out.len() {
        let d = (cpu_out[i] - gpu_out[i]).abs();
        if d > max_diff_out { max_diff_out = d; }
    }
    println!("Diferencia Máxima en Output: {:.10}", max_diff_out);

    if diff_loss < 1e-4 {
        println!("\n✅ ÉXITO: Paridad aceptable entre CPU y WebGPU.");
    } else {
        println!("\n⚠️ AVISO: Discrepancia detectada (esto puede ser normal por optimizaciones de GPU).");
    }
}
