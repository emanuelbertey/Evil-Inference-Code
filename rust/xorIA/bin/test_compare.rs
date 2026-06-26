use std::fs;
use std::error::Error;
use burn::{
    tensor::{Tensor, TensorData, Int},
    nn::{LinearConfig, EmbeddingConfig},
};
use burn_flex::Flex;
use xlstm::blocks::trasformer::layer::{TransformerConfig, TransformerLayerConfig};
use xlstm::blocks::trasformer_bit::model::Tokenizer;
use xlstm::blocks::load_pytorch::PyTorchLoader;

type B = Flex<f32>;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();

    // Load tokenizer just for vocab_size
    let tokenizer = Tokenizer::load("python/tokenizer.json")?;
    let vocab_size = tokenizer.vocab_size();

    // Read input IDs directly from Python (same tokenization)
    let py_ids_bytes = fs::read("test_data/input_ids.npy")?;
    let hdr_ids = py_ids_bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
    let ids: Vec<usize> = py_ids_bytes[hdr_ids..]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect();
    println!("Input IDs ({}): {:?}", ids.len(), ids);
    let seq_len = ids.len();

    let layer_cfg = TransformerLayerConfig {
        d_model: 256, num_heads: 8, num_kv_groups: 4, head_dim: None,
        ffn_expansion: 4.0, use_swiglu: true, max_seq_len: 128,
        rope_base: 10000.0, rope_scaling: 1.0, causal: true,
        attn_dropout: 0.0, ffn_dropout: 0.0, residual_dropout: 0.0,
        attn_logit_cap: None, bias: false, norm_eps: 1e-5, ffn_round_to: 64,
    };
    let mut transformer = TransformerConfig { num_layers: 6, layer: layer_cfg }.init(&device);
    let mut embedding = EmbeddingConfig::new(vocab_size, 256).init(&device);
    let mut head = LinearConfig::new(256, vocab_size).with_bias(false).init(&device);

    let tensors = PyTorchLoader::load_safetensors(
        "python/model_test.safetensors", "python/model_test_mapping.json")?;
    PyTorchLoader::load_into_transformer(&mut transformer, &tensors, 6, &device)?;
    if let Some(d) = tensors.get("embedding.weight") {
        embedding.weight = burn::module::Param::from_tensor(
            Tensor::<B, 2>::from_data(d.clone(), &device));
    }
    if let Some(d) = tensors.get("head.weight") {
        head.weight = burn::module::Param::from_tensor(
            Tensor::<B, 2>::from_data(d.clone(), &device).transpose());
    }

    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&i| i as i64).collect(), [1, seq_len]), &device);
    let x = embedding.forward(input);
    let x = transformer.forward(x, 0);
    let logits = head.forward(x);
    let rust_logits: Vec<f32> = logits.to_data().as_slice::<f32>().unwrap().to_vec();

    let py_bytes = fs::read("test_data/logits.npy")?;
    let hdr_end = py_bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
    let py_logits: Vec<f32> = py_bytes[hdr_end..]
        .chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    let total = 1 * seq_len * vocab_size;
    let n = rust_logits.len().min(py_logits.len()).min(total).min(100000);

    let mut max_diff = 0.0f32;
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let d = (rust_logits[i] - py_logits[i]).abs();
        if d > max_diff { max_diff = d; }
        sum_sq += d * d;
    }
    let mse = sum_sq / n as f32;

    println!("\n--- Comparación ---");
    println!("Elementos: {}", n);
    println!("Max diff:  {:.6}", max_diff);
    println!("MSE:       {:.6}", mse);
    println!("Rust[:5]: {:?}", &rust_logits[..5]);
    println!("Py  [:5]: {:?}", &py_logits[..5]);

    if max_diff < 0.01 {
        println!("\n✓ Pesos correctos");
    } else if max_diff < 1.0 {
        println!("\n⚠ Pequeñas diferencias numéricas");
    } else {
        println!("\n✗ ERROR: pesos mal cargados");
    }
    Ok(())
}
