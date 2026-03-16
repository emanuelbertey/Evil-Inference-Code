use burn::prelude::*;
use burn::tensor::activation;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;
type AdBackend = Autodiff<TestBackend>;

fn logaddexp<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let max = a.clone().max_pair(b.clone());
    let diff = a - b;
    let exp_term = diff.abs().neg().exp();
    max + (exp_term + 1.0).log()
}

fn log_g<B: burn::prelude::Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()) + 0.5).log();
    let neg = activation::softplus(x.neg(), 1.0).neg();
    neg.mask_where(mask, pos)
}

fn main() {
    let device = Default::default();

    // --- TEST 1: Primitivas ---
    let x_vals = Tensor::<AdBackend, 1>::from_data([-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0], &device);
    println!("--- RUST: Primitivas ---");
    println!("Softplus: {}", activation::softplus(x_vals.clone(), 1.0));
    println!("Log-g:    {}", log_g(x_vals));

    // --- TEST 2: Gradiente (S=250) ---
    let seq_len = 250;
    let seq_data: Vec<f32> = (0..seq_len).map(|i| -50.0 + (i as f32 * 100.0 / (seq_len - 1) as f32)).collect();
    let seq_p = Tensor::<AdBackend, 1>::from_data(seq_data.as_slice(), &device).require_grad();
    let seq_s = Tensor::<AdBackend, 1>::from_data(seq_data.as_slice(), &device).require_grad();
    
    // 1. Version Paralela (Da 1.0)
    let x_max = seq_p.clone().max().detach();
    let res_parallel = (seq_p.clone() - x_max.clone()).exp().cumsum(0).log() + x_max;
    
    // 2. Version Estable (Da 1.95)
    let mut current = seq_s.clone().slice([0..1]);
    let mut results = vec![current.clone()];
    for i in 1..seq_len {
        let next_val = seq_s.clone().slice([i..i+1]);
        current = logaddexp(current, next_val);
        results.push(current.clone());
    }
    let res_stable = Tensor::cat(results, 0);

    let g_p = res_parallel.sum().backward();
    let grad_p = seq_p.grad(&g_p).expect("No grad");
    
    let g_s = res_stable.sum().backward();
    let grad_s = seq_s.grad(&g_s).expect("No grad");

    println!("\n--- RUST: logcumsumexp (S=250) ---");
    println!("PARALLEL - Grad Max: {}", grad_p.max());
    println!("STABLE   - Grad Max: {}", grad_s.clone().max());
    println!("STABLE   - Grad Min: {}", grad_s.clone().min());
}
