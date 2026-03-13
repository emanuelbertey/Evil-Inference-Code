extern crate alloc;
use xlstm::{MinGru, MinGruConfig, MinGruState};

use burn::tensor::Tensor;
use serde::Deserialize;
use std::fs;

type TestBackend = burn_ndarray::NdArray<f32>;

#[derive(Deserialize)]
struct TestData {
    x: Vec<f32>,
    h0: Vec<f32>,
    z_weight: Vec<f32>,
    z_bias: Vec<f32>,
    h_weight: Vec<f32>,
    y: Vec<f32>,
    shape: Vec<usize>,
}

fn main() {
    let device = Default::default();
    
    let path = "tests/data/mingru_test_data.json";
    let data_str = fs::read_to_string(path).expect("Run Python script first!");
    let data: TestData = serde_json::from_str(&data_str).unwrap();
    
    let b = data.shape[0];
    let s = data.shape[1];
    let input_dim = data.shape[2];
    let hidden_dim = data.shape[3];
    
    let x = Tensor::<TestBackend, 3>::from_floats(data.x.as_slice(), &device).reshape([b, s, input_dim]);
    let h0 = Tensor::<TestBackend, 2>::from_floats(data.h0.as_slice(), &device).reshape([b, hidden_dim]);
    let expected_y = Tensor::<TestBackend, 3>::from_floats(data.y.as_slice(), &device).reshape([b, s, hidden_dim]);
    
    // Config and model
    let mut config = MinGruConfig::new(input_dim, hidden_dim, 1);
    config.gate_bias = -1.0;
    
    let mut mingru = config.init::<TestBackend>(&device);
    
    // PyTorch nn.Linear stores weights transposed relative to Burn
    let z_w_tensor = Tensor::<TestBackend, 2>::from_floats(data.z_weight.as_slice(), &device)
        .reshape([hidden_dim, input_dim])
        .transpose();
    let z_b_tensor = Tensor::<TestBackend, 1>::from_floats(data.z_bias.as_slice(), &device);
    
    let h_w_tensor = Tensor::<TestBackend, 2>::from_floats(data.h_weight.as_slice(), &device)
        .reshape([hidden_dim, input_dim])
        .transpose();
    
    mingru.layers[0].linear_z.weight = z_w_tensor.into();
    mingru.layers[0].linear_z.bias = Some(z_b_tensor.into());
    mingru.layers[0].linear_h.weight = h_w_tensor.into();
    
    let states = vec![MinGruState::new(h0)];
    
    let (out, _) = mingru.forward(x, Some(states));
    
    let diff: Tensor<TestBackend, 3> = (out.clone() - expected_y.clone()).abs();
    let max_diff = diff.max().into_scalar();
    
    println!("Max diff between minGRU Python and Rust: {}", max_diff);
    if max_diff < 1e-4 {
         println!("✅ MinGRU Equivalence Test Passed!");
    } else {
         println!("❌ MinGRU Equivalence Test Failed!");
    }
}
