extern crate alloc;
use xlstm::{MinGruConfig, MinGruState};

use burn::tensor::Tensor;
use burn::module::Param;
use serde::Deserialize;
use std::fs;

type TestBackend = burn_ndarray::NdArray<f32>;

#[derive(Deserialize)]
struct TestData {
    x: Vec<f32>,
    h0: Vec<f32>,
    z_weight: Vec<f32>,
    // z_bias is ignored: MinGru uses bias=false per notebook
    h_weight: Vec<f32>,
    y: Vec<f32>,
    shape: Vec<usize>,
}

fn main() {
    let device = Default::default();

    let path = "tests/data/mingru_test_data.json";
    let data_str = fs::read_to_string(path).expect("Run Python script first to generate test data!");
    let data: TestData = serde_json::from_str(&data_str).unwrap();

    let b          = data.shape[0];
    let s          = data.shape[1];
    let input_dim  = data.shape[2];
    let hidden_dim = data.shape[3];

    // expansion_factor = hidden_dim / input_dim
    let expansion_factor = hidden_dim / input_dim;

    let x = Tensor::<TestBackend, 3>::from_floats(data.x.as_slice(), &device)
        .reshape([b, s, input_dim]);

    // h0 from Python: shape [B, 1, hidden_dim]
    let h0 = Tensor::<TestBackend, 3>::from_floats(data.h0.as_slice(), &device)
        .reshape([b, 1, hidden_dim]);

    // expected output: shape [B, S, input_dim]  (after output_projection)
    // Note: if Python test was before output_projection, adjust accordingly
    let expected_y = Tensor::<TestBackend, 3>::from_floats(data.y.as_slice(), &device)
        .reshape([b, s, input_dim]);

    // Build model with current API
    let config = MinGruConfig {
        input_features: input_dim,
        expansion_factor,
    };
    let mut mingru = config.init::<TestBackend>(&device);

    // Load weights — PyTorch Linear stores weight as [out, in], Burn as [in, out]
    // So we need to transpose when loading from PyTorch
    let z_w = Tensor::<TestBackend, 2>::from_floats(data.z_weight.as_slice(), &device)
        .reshape([hidden_dim, input_dim])
        .transpose();  // -> [input_dim, hidden_dim] = Burn layout
    let h_w = Tensor::<TestBackend, 2>::from_floats(data.h_weight.as_slice(), &device)
        .reshape([hidden_dim, input_dim])
        .transpose();  // -> [input_dim, hidden_dim] = Burn layout

    mingru.linear_z.weight = Param::from_tensor(z_w);
    mingru.linear_h.weight = Param::from_tensor(h_w);
    // linear_z.bias and linear_h.bias are None (bias=false, matching notebook)

    // Run forward
    let states = vec![MinGruState::new(h0)];
    let (out, _) = mingru.forward(x, Some(states));

    let diff: Tensor<TestBackend, 3> = (out.clone() - expected_y.clone()).abs();
    let max_diff = diff.max().into_scalar();

    println!("Max diff between minGRU Python and Rust: {}", max_diff);
    if max_diff < 1e-4 {
        println!("✅ MinGRU Equivalence Test Passed!");
    } else {
        println!("❌ MinGRU Equivalence Test Failed! (check weight loading or formula)");
    }
}
