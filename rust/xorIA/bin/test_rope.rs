use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, Shape, TensorData};
use burn_flex::Flex;
use xlstm::blocks::trasformer::rope::{RoPE, RoPEConfig};

type Backend = Flex;

pub fn test_rope_main() -> Result<(), String> {
    let device = Default::default();

    println!("=== Testing RoPE Mathematically ===");
    
    // Config: sequence length 4, head dimension 8
    let seq_len = 4;
    let head_dim = 8;
    let rope_config = RoPEConfig::new(head_dim, seq_len);
    let rope = rope_config.init::<Backend>(&device);

    // Create a dummy Query tensor [batch=1, seq=4, heads=1, head_dim=8]
    // Values are simple to see the rotation
    let q_data: Vec<f32> = (0..32).map(|x| x as f32).collect();
    let data = TensorData::new(q_data, [1, seq_len, 1, head_dim]);
    let q = Tensor::<Backend, 4>::from_data(data, &device);
    let k = q.clone();

    println!("Original Q tensor (first token):");
    println!("{}", q.clone().slice([0..1, 0..1, 0..1, 0..head_dim]));

    let (q_rot, k_rot) = rope.forward(q.clone(), k.clone(), 0);
    
    println!("Rotated Q tensor (first token, offset 0):");
    println!("{}", q_rot.clone().slice([0..1, 0..1, 0..1, 0..head_dim]));

    println!("Rotated Q tensor (second token, offset 0):");
    println!("{}", q_rot.clone().slice([0..1, 1..2, 0..1, 0..head_dim]));

    println!("\n=== Experimental Partial Rotation (Percentage-based) ===");
    // Partial RoPE (e.g., used in Phi-2, Pythia, etc.) rotates only a percentage of the dimensions.
    let rotary_pct = 0.50; // 50% of the dimensions
    let rotary_dim = (head_dim as f64 * rotary_pct) as usize;
    
    // Ensure it's even
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");

    println!("Using Partial RoPE: {}% of {} dimensions = {} dimensions rotated", rotary_pct * 100.0, head_dim, rotary_dim);
    
    // We instantiate a RoPE just for rotary_dim
    let rope_partial_config = RoPEConfig::new(rotary_dim, seq_len);
    let rope_partial = rope_partial_config.init::<Backend>(&device);

    // Split q into two parts
    let q_rotary = q.clone().slice([0..1, 0..seq_len, 0..1, 0..rotary_dim]);
    let q_pass = q.clone().slice([0..1, 0..seq_len, 0..1, rotary_dim..head_dim]);

    // Apply rotation only to the first part
    let q_rotary_rotated = rope_partial.apply_to_single(q_rotary, 0);

    // Concatenate back
    let q_partial_rot = Tensor::cat(vec![q_rotary_rotated, q_pass], 3);

    println!("Partial Rotated Q tensor (first token, rotary_dim=4):");
    println!("{}", q_partial_rot.clone().slice([0..1, 0..1, 0..1, 0..head_dim]));

    println!("Partial Rotated Q tensor (second token, rotary_dim=4):");
    println!("{}", q_partial_rot.clone().slice([0..1, 1..2, 0..1, 0..head_dim]));

    println!("Test completed successfully.");
    Ok(())
}

fn main() {
    if let Err(e) = test_rope_main() {
        eprintln!("Error: {}", e);
    }
}
