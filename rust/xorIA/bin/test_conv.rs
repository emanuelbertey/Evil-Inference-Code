use burn::tensor::Tensor;
use burn::backend::ndarray::NdArray;
use burn_autodiff::Autodiff;
use burn::prelude::*;
use xlstm::components::conv::{CausalConv1d, CausalConv1dConfig};
use burn::module::Param;

type TestBackend = NdArray<f32>;
type AdBackend = Autodiff<TestBackend>;

fn main() {
    let device = Default::default();
    
    let config = CausalConv1dConfig {
        feature_dim: 4,
        kernel_size: 3,
        bias: true,
    };
    
    let mut conv = config.init::<AdBackend>(&device);
    
    let w_vals: Vec<f32> = vec![
        0.1, 0.2, 0.3, // f=0
        0.1, 0.2, 0.3, // f=1
        0.1, 0.2, 0.3, // f=2
        0.1, 0.2, 0.3, // f=3
    ];
    let w = Tensor::<TestBackend, 3>::from_data(TensorData::new(w_vals, [4, 1, 3]), &device);
    let b = Tensor::<TestBackend, 1>::ones([4], &device).mul_scalar(0.1);
    
    conv.conv.weight = Param::from_tensor(Tensor::<AdBackend, 3>::from_inner(w).require_grad());
    conv.conv.bias = Some(Param::from_tensor(Tensor::<AdBackend, 1>::from_inner(b).require_grad()));
    
    use burn::tensor::TensorData;
    let vals: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        2.0, 3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
        4.0, 5.0, 6.0, 7.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    let x_inner = Tensor::<TestBackend, 3>::from_data(TensorData::new(vals, [1, 5, 4]), &device);
    let x = Tensor::<AdBackend, 3>::from_inner(x_inner.clone()).require_grad();
    
    let y_p = conv.forward(x.clone());
    println!("Parallel Y:\n{}", y_p.clone().inner());
    
    let grads_p = y_p.clone().sum().backward();
    let grad_x_p = x.grad(&grads_p).unwrap();
    println!("\nParallel Grad X:\n{}", grad_x_p);
    
    // Recurrent step
    let x_r = Tensor::<AdBackend, 3>::from_inner(x_inner.clone()).require_grad();
    let mut state = conv.empty_state(1, &device);
    let mut y_steps = Vec::new();
    
    for t in 0..5 {
        let x_t = x_r.clone().narrow(1, t, 1).reshape([1, 4]);
        let (y_t, next_state) = conv.step(x_t, state);
        state = next_state;
        y_steps.push(y_t.unsqueeze_dim::<3>(1));
    }
    
    let y_r = Tensor::cat(y_steps, 1);
    println!("\nRecurrent Y:\n{}", y_r.clone().inner());
    
    let grads_r = y_r.clone().sum().backward();
    let grad_x_r = x_r.grad(&grads_r).unwrap();
    println!("\nRecurrent Grad X:\n{}", grad_x_r);
    
    let diff = (y_p - y_r).abs().max().into_scalar();
    println!("\nDiff Y: {:?}", diff);
}
