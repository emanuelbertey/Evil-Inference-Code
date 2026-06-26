// Test: tokenize text → forward → save logits+IDs → Python compares
use std::error::Error;
use burn::tensor::{Tensor, TensorData, Int};
use burn_flex::Flex;
use xlstm::blocks::trasformer_bit::model::Tokenizer;
use xlstm::blocks::trasformer::layer::{Transformer, TransformerConfig, TransformerLayerConfig};
use xlstm::blocks::load_pytorch::PyTorchLoader;
use burn::nn::{LinearConfig, EmbeddingConfig};

type B = Flex<f32>;

fn npy_save_f32(path: &str, data: &[f32]) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut magic = b"\x93NUMPY".to_vec();
    magic.push(3); magic.push(0);
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", data.len());
    let pad = 64 - (magic.len() + 4 + header.len()) % 64;
    let header = header + &" ".repeat(pad) + "\n";
    let mut out = magic;
    out.extend_from_slice(&(header.as_bytes().len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in data { out.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path, out)?;
    Ok(())
}

fn npy_save_i32(path: &str, data: &[i32]) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut magic = b"\x93NUMPY".to_vec();
    magic.push(3); magic.push(0);
    let header = format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}", data.len());
    let pad = 64 - (magic.len() + 4 + header.len()) % 64;
    let header = header + &" ".repeat(pad) + "\n";
    let mut out = magic;
    out.extend_from_slice(&(header.as_bytes().len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in data { out.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path, out)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();

    // 1. Load tokenizer (same as training with custom path)
    let tokenizer = Tokenizer::load_pretrained("python/tokenizer.json")?;
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {}", vocab_size);

    // 2. Tokenize first 5000 chars of input.txt (normalize CRLF for Windows compat)
    let text = std::fs::read_to_string("xorIA/input.txt")?;
    let sample: String = text.chars().take(5000).collect();
    let ids = tokenizer.encode(&sample);
    let seq_len = ids.len().min(128);
    let ids = &ids[..seq_len];
    println!("Text chars: {}, Tokens: {}", sample.len(), ids.len());
    println!("First 10 IDs: {:?}", &ids[..10.min(ids.len())]);

    // 3. Load model from safetensors
    let mut embedding = EmbeddingConfig::new(vocab_size, 256).init(&device);
    let mut transformer = TransformerConfig {
        num_layers: 6,
        layer: TransformerLayerConfig {
            d_model: 256, num_heads: 8, num_kv_groups: 4, head_dim: None,
            ffn_expansion: 4.0, use_swiglu: true, max_seq_len: 128,
            rope_base: 10000.0, rope_scaling: 1.0, causal: true,
            attn_dropout: 0.0, ffn_dropout: 0.0, residual_dropout: 0.0,
            attn_logit_cap: None, bias: false, norm_eps: 1e-5, ffn_round_to: 64,
        },
    }.init(&device);
    let mut head = LinearConfig::new(256, vocab_size).with_bias(false).init(&device);

    let tensors = PyTorchLoader::load_safetensors(
        "python/model_test.safetensors", "python/model_test_mapping.json")?;
    PyTorchLoader::load_into_transformer(&mut transformer, &tensors, 6, &device)?;
    if let Some(d) = tensors.get("embedding.weight") {
        embedding.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), &device));
    }
    if let Some(d) = tensors.get("head.weight") {
        head.weight = burn::module::Param::from_tensor(
            Tensor::<B, 2>::from_data(d.clone(), &device).transpose());
    }

    // 4. Forward
    let input_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&i| i as i64).collect(), [1, ids.len()]), &device);
    let x = embedding.forward(input_tensor);
    let x = transformer.forward(x, 0);
    let logits = head.forward(x);
    let logits_data: Vec<f32> = logits.to_data().as_slice::<f32>().unwrap().to_vec();
    println!("Logits shape: [1, {}, {}]", ids.len(), vocab_size);
    println!("Logits[0,:5]: {:?}", &logits_data[..5]);

    // 5. Save for Python
    npy_save_i32("test_data/rust_token_ids.npy", &ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?;
    npy_save_f32("test_data/rust_logits_full.npy", &logits_data)?;
    println!("\nSaved to test_data/");

    Ok(())
}
