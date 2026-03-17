use burn::prelude::*;
use burn::tensor::activation;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;

type AdBackend = Autodiff<NdArray<f32>>;

fn log_g<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    let mask = x.clone().greater_equal_elem(0.0);
    // g(x) = log(relu(x) + 0.5) si x >= 0 else -softplus(-x)
    let pos = (activation::relu(x.clone()) + 0.5).log();
    let neg = activation::softplus(x.neg(), 1.0).neg();
    neg.mask_where(mask, pos)
}

fn parallel_logcumsumexp<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dim = if D == 1 { 0 } else { 1 };
    x.exp().cumsum(dim).log()
}

fn main() {
    let device = Default::default();

    println!("--- RUST: FULL PARITY TEST (Values & Gradients) ---");

    // 1. TEST GRADIENTES PRIMITIVAS
    let x_vals = Tensor::<AdBackend, 1>::from_data([-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0], &device).require_grad();
    
    // Softplus
    let res_sp = activation::softplus(x_vals.clone(), 1.0);
    let grads_sp = res_sp.clone().sum().backward();
    let grad_sp = x_vals.grad(&grads_sp).expect("No grad");
    println!("\n[1] Softplus:");
    println!("Values: {}", res_sp);
    println!("Grads:  {}", grad_sp);

    // Log-g
    let x_vals_g = Tensor::<AdBackend, 1>::from_data([-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0], &device).require_grad();
    let res_lg = log_g(x_vals_g.clone());
    let grads_lg = res_lg.clone().sum().backward();
    let grad_lg = x_vals_g.grad(&grads_lg).expect("No grad");
    println!("\n[2] Log-g:");
    println!("Values: {}", res_lg);
    println!("Grads:  {}", grad_lg);

    // 2. TEST LOGCUMSUMEXP 1D (S=250)
    let seq_len = 250;
    let seq_data: Vec<f32> = (0..seq_len).map(|i| -50.0 + (i as f32 * 100.0 / (seq_len - 1) as f32)).collect();
    let seq_1d = Tensor::<AdBackend, 1>::from_data(seq_data.as_slice(), &device).require_grad();
    
    let res_1d = parallel_logcumsumexp(seq_1d.clone());
    let grads_1d = res_1d.clone().sum().backward();
    let grad_1d = seq_1d.grad(&grads_1d).expect("No grad");

    println!("\n[3] Logcumsumexp 1D:");
    println!("Val Max:  {}", res_1d.max());
    println!("Grad Max: {}", grad_1d.max());

    // 3. TEST LOGCUMSUMEXP 3D [2, 250, 4]
    let seq_3d = Tensor::<AdBackend, 3>::from_data(
        Tensor::<AdBackend, 1>::from_data(seq_data.repeat(8).as_slice(), &device)
            .reshape([2, 250, 4])
            .to_data(),
        &device
    ).require_grad();
    
    let res_3d = parallel_logcumsumexp(seq_3d.clone());
    let grads_3d = res_3d.clone().sum().backward();
    let grad_3d = seq_3d.grad(&grads_3d).expect("No grad");

    println!("\n[4] Logcumsumexp 3D:");
    println!("Val Max:  {}", res_3d.max());
    println!("Grad Max: {}", grad_3d.max());
}
