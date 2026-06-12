use burn::prelude::*;
use burn::tensor::{Tensor, Distribution};
use burn::backend::ndarray::NdArray;
use burn_autodiff::Autodiff;
use xlstm::blocks::mlstm::layer::{MLSTMLayer, MLSTMLayerConfig};
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::nn::loss::MseLoss;

type MyBackend = Autodiff<NdArray<f32>>;

fn main() {
    println!("=== mLSTM COPY TASK TEST (RUST) ===");
    let device = Default::default();
    
    let batch_size = 1;
    let seq_len = 8;
    let embedding_dim = 16;
    let lr = 8e-4;
    
    let config = MLSTMLayerConfig::new(embedding_dim)
        .with_num_heads(4)
        .with_conv1d_kernel_size(4)
        .with_qkv_proj_blocksize(4)
        .with_proj_factor(2.0)
        .with_bias(true)
        .with_context_length(seq_len);
        
    let mut layer: MLSTMLayer<MyBackend> = config.init(&device);
    
    // Adam optimizer
    let mut optim = AdamConfig::new().init();
    
    // Static pattern to copy
    let fixed_x = Tensor::<MyBackend, 3>::random(
        [batch_size, seq_len, embedding_dim], 
        Distribution::Normal(0.0, 1.0), 
        &device
    );
    
    let mut final_loss = 0.0;
    
    for i in 1..=100 {
        // Forward
        let output = layer.forward(fixed_x.clone());
        
        // Loss: MSE
        let loss = MseLoss::new().forward(output, fixed_x.clone(), burn::nn::loss::Reduction::Mean);
        final_loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        // Backward
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &layer);
        
        // Update
        layer = optim.step(lr as f64, layer, grads_params);
        
        if i % 20 == 0 {
            println!("Step {:3}: Loss: {:.8}", i, final_loss);
        }
    }
    
    println!("\nFINAL LOSS (Rust): {:.8}", final_loss);
}
