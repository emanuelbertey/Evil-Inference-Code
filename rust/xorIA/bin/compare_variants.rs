use burn::prelude::*;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;

type AdBackend = Autodiff<NdArray<f32>>;

// LOGCUMSUMEXP PURO: Sin bucles (for/while) y sin apensors. 
// Definición matemática pura: log(cumsum(exp(x)))
fn parallel_logcumsumexp<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.exp().cumsum(0).log()
}

fn main() {
    let device = Default::default();
    let seq_len = 250;
    let seq_data: Vec<f32> = (0..seq_len).map(|i| -50.0 + (i as f32 * 100.0 / (seq_len - 1) as f32)).collect();

    let seq = Tensor::<AdBackend, 1>::from_data(seq_data.as_slice(), &device).require_grad();
    
    let res = parallel_logcumsumexp(seq.clone());
    let grads = res.clone().sum().backward();
    let grad_input = seq.grad(&grads).expect("No grad");
    
    println!("--- RUST: Pure Vectorized (No Loops, No Apensors) ---");
    println!("Val Max:  {}", res.max());
    println!("Grad Max: {}", grad_input.clone().max());
    println!("Grad Min: {}", grad_input.clone().min());
}
