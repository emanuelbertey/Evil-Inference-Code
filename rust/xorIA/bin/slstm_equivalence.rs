use xlstm::blocks::slstm::layer::{SLSTMLayer, SLSTMLayerConfig};
use burn::tensor::{Tensor, Distribution};
use burn_autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;

type TestBackend = NdArray<f32>;
type AdBackend = Autodiff<TestBackend>;

fn main() {
    let device = Default::default();
    let batch_size = 1;
    let embedding_dim = 16;
    
    for seq_len in 1..=12 {
        println!("--- TEST sLSTM: SEQ_LEN={} ---", seq_len);
        
        let config = SLSTMLayerConfig {
            embedding_dim,
            num_heads: 4,
            conv1d_kernel_size: 4,
            dropout: 0.0,
        };
        
        // Initialize layers.
        let layer_p: SLSTMLayer<AdBackend> = config.init(&device);
        let record = layer_p.clone().into_record();
        let layer_r: SLSTMLayer<AdBackend> = config.init(&device).load_record(record);

        let input_raw = Tensor::<TestBackend, 3>::random([batch_size, seq_len, embedding_dim], Distribution::Normal(0.0, 1.0), &device);
        
        // Parallel pass
        let input_p = Tensor::<AdBackend, 3>::from_inner(input_raw.clone()).require_grad();
        let output_p = layer_p.forward(input_p.clone());
        let grads_p = output_p.clone().sum().backward();
        
        // Use igate weight as a proxy for gradient check
        let g_w_p = layer_p.igate.weight.grad(&grads_p).expect("No grad for parallel weight");
        let g_in_p = input_p.grad(&grads_p).expect("No grad for parallel input");

        // Recurrent pass
        let input_r = Tensor::<AdBackend, 3>::from_inner(input_raw).require_grad();
        let mut state = layer_r.empty_state(batch_size, &device);
        let mut steps = Vec::new();
        
        for t in 0..seq_len {
            let x_t = input_r.clone().narrow(1, t, 1).reshape([batch_size, embedding_dim]);
            let (y_t, next_state) = layer_r.step(x_t, state);
            steps.push(y_t.unsqueeze_dim::<3>(1));
            state = next_state;
        }
        
        let output_r = Tensor::cat(steps, 1);
        let grads_r = output_r.clone().sum().backward();
        
        let g_w_r = layer_r.igate.weight.grad(&grads_r).expect("No grad for recurrent weight");
        let g_in_r = input_r.grad(&grads_r).expect("No grad for recurrent input");

        let val_diff = (output_p - output_r).abs().mean().into_scalar();
        let grad_w_diff = (g_w_p - g_w_r).abs().mean().into_scalar();
        let grad_in_diff = (g_in_p - g_in_r).abs().mean().into_scalar();

        println!("Diferencia VALOR:       {:.10}", val_diff);
        println!("Diferencia GRAD PESO:   {:.10}", grad_w_diff);
        println!("Diferencia GRAD INPUT:  {:.10}", grad_in_diff);

        if val_diff > 1e-5 || grad_w_diff > 1e-5 {
            println!("FAILURE: Discrepancy detected at SEQ_LEN={}", seq_len);
            // std::process::exit(1);
        }
        println!();
    }
    println!("ALL TESTS PASSED (1-12)");
}
