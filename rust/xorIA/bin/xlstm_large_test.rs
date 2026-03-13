use burn::prelude::*;
use burn::tensor::{Tensor, Distribution};
use burn::backend::ndarray::NdArray;
use burn_autodiff::Autodiff;
use xlstm::blocks::xlstm_large::{XLSTMLarge, XLSTMLargeConfig};
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::nn::loss::CrossEntropyLossConfig;

type MyBackend = Autodiff<NdArray<f32>>;

fn main() {
    println!("=== xLSTM Large COPY TASK & EQUIVALENCE TEST (RUST) ===");
    let device = Default::default();
    
    let batch_size = 1;
    let seq_len = 16;
    let embedding_dim = 32;
    let lr = 1e-3;
    let vocab_size = 64;
    
    let config = XLSTMLargeConfig {
        embedding_dim,
        num_heads: 4,
        num_blocks: 2, 
        vocab_size,
        use_bias: true,
        norm_eps: 1e-6,
        norm_reduction_force_float32: true,
        add_out_norm: true,
        qk_dim_factor: 0.5,
        v_dim_factor: 1.0,
        mlstm_backend: xlstm::blocks::xlstm_large::config::MLSTMBackendConfig::new(),
        ffn_proj_factor: 2.6667,
        ffn_round_up_to_multiple_of: 64,
        gate_soft_cap: Some(15.0),
        output_logit_soft_cap: Some(30.0),
        weight_mode: "single".to_string(),
    };
        
    let mut model: XLSTMLarge<MyBackend> = XLSTMLarge::init(&config, &device);
    
    // Adam optimizer
    let mut optim = AdamConfig::new().init();
    
    // Static pattern to copy
    let fixed_x_indices = Tensor::<MyBackend, 2, Int>::random(
        [batch_size, seq_len], 
        Distribution::Default, 
        &device
    ).clamp(0, vocab_size as i64 - 1);
    
    println!("\n--- Phase 1: Training on Copy Task (100 steps) ---");
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut final_loss = 0.0;
    
    for i in 1..=100 {
        let (logits, _) = model.forward(fixed_x_indices.clone(), None);
        
        let [b, s, v] = logits.dims();
        let targets = fixed_x_indices.clone();
        
        let loss = loss_fn.forward(logits.reshape([b * s, v]), targets.reshape([b * s]));
        final_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(lr as f64, model, grads_params);
        
        if i % 20 == 0 || i == 1 {
            println!("Step {:3}: Loss: {:.8}", i, final_loss);
        }
    }
    
    println!("\n--- Phase 2: Equivalence Test (Parallel vs Recurrent) ---");
    let test_input = Tensor::<MyBackend, 2, Int>::random(
        [batch_size, seq_len], 
        Distribution::Default, 
        &device
    ).clamp(0, vocab_size as i64 - 1);
    
    // Parallel forward (passing None for state)
    let (logits_p, _) = model.forward(test_input.clone(), None);
    
    // Recurrent forward (step-by-step)
    let mut state = model.empty_state(batch_size, &device);
    let mut recurrent_logits = Vec::new();
    for t in 0..seq_len {
        let x_t = test_input.clone().narrow(1, t, 1);
        let (y_t, next_state) = model.forward(x_t, Some(state));
        recurrent_logits.push(y_t);
        state = next_state.expect("State should be returned");
    }
    let logits_r = Tensor::cat(recurrent_logits, 1);
    
    let diff = (logits_p.clone() - logits_r.clone()).abs().mean().into_scalar();
    println!("Parallel vs Recurrent Logits Diff: {:.10}", diff);
    
    if diff < 1e-4 {
        println!("✅ EQUIVALENCE PASSED");
    } else {
        println!("❌ EQUIVALENCE FAILED");
        // Print some values to debug
        let p_data = logits_p.clone().into_data();
        let r_data = logits_r.clone().into_data();
        println!("Sample P: {:?}", &p_data.as_slice::<f32>().unwrap()[0..5]);
        println!("Sample R: {:?}", &r_data.as_slice::<f32>().unwrap()[0..5]);
    }

    println!("\n--- Phase 3: Gradient Mode / Stability at 256 steps ---");
    let seq_len_long = 256;
    let long_input = Tensor::<MyBackend, 2, Int>::random(
        [batch_size, seq_len_long], 
        Distribution::Default, 
        &device
    ).clamp(0, vocab_size as i64 - 1);
    
    let (logits_long, _) = model.forward(long_input, None);
    // Simple loss to check gradients: mean of squares
    let loss_long = logits_long.powf_scalar(2.0).mean();
    let loss_val = loss_long.clone().into_data().as_slice::<f32>().unwrap()[0];
    
    if loss_val.is_nan() || loss_val.is_infinite() {
        println!("❌ Gradient Mode (256 steps): LOSS IS NAN/INF! Stability issues.");
    } else {
        println!("Loss at 256 steps: {:.8}", loss_val);
        let grads = loss_long.backward();
        
        // Calcular norma de algunos gradientes clave
        if let Some(grad) = model.lm_head.weight.grad(&grads) {
            let norm = grad.powf_scalar(2.0).sum().sqrt().into_scalar();
            println!("Gradient Norm (lm_head.weight): {:.10}", norm);
        }
        
        if let Some(grad) = model.embedding.weight.grad(&grads) {
            let norm = grad.powf_scalar(2.0).sum().sqrt().into_scalar();
            println!("Gradient Norm (embedding.weight): {:.10}", norm);
        }
        
        println!("✅ Gradients computed successfully for 256 sequence length.");
    }
    
    println!("\n=== TEST COMPLETED ===");
}
