use burn::prelude::*;
use burn::tensor::{Tensor, Distribution};
use burn::backend::ndarray::NdArray;
use burn_autodiff::Autodiff;
use xlstm::blocks::mlstm::layer::{MLSTMLayer, MLSTMLayerConfig};
use burn::optim::SgdConfig;
use burn::optim::Optimizer;

type MyBackend = Autodiff<NdArray<f32>>;

fn main() {
    println!("=== mLSTM GRADIENT STABILITY TEST (RUST) ===");
    let device = Default::default();
    
    let batch_size = 1;
    let seq_len = 16;
    let embedding_dim = 16;
    let lr = 5.0; // High LR
    
    let config = MLSTMLayerConfig::new(embedding_dim)
        .with_num_heads(4)
        .with_conv1d_kernel_size(4)
        .with_qkv_proj_blocksize(4)
        .with_proj_factor(2.0)
        .with_bias(true); 
        
    let mut layer: MLSTMLayer<MyBackend> = config.init(&device);
    
    // Simple SGD optimizer
    let mut optim = SgdConfig::new().init();
    
    for i in 1..=100 {
        let x = Tensor::<MyBackend, 3>::random(
            [batch_size, seq_len, embedding_dim], 
            Distribution::Normal(0.0, 1.0), 
            &device
        );
        
        // Forward
        let output = layer.forward(x);
        
        // Loss: pow(2).mean()
        let loss = output.powf_scalar(2.0).mean();
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        if loss_val.is_nan() || loss_val.is_infinite() {
            println!("Step {:3}: ❌ LOSS IS NAN! Explosion detected. Val: {}", i, loss_val);
            break;
        }
        
        // Backward
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &layer);
        
        // Update
        layer = optim.step(lr, layer, grads_params);
        
        if i % 10 == 0 {
            println!("Step {:3}: Loss: {:8.6} | ✅ STABLE (Update applied)", i, loss_val);
        }
    }
    
    println!("\nTest completed.");
}
