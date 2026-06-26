use std::fs;
use std::error::Error;
use burn::{
    tensor::{Tensor, TensorData, Int, activation::softmax},
    nn::{LinearConfig, EmbeddingConfig, loss::CrossEntropyLossConfig},
    module::Param,
};
use burn_flex::Flex;
use xlstm::blocks::trasformer::layer::{TransformerConfig, TransformerLayerConfig};
use xlstm::blocks::trasformer_bit::ops::{apply_rope_partial, repeat_kv, apply_causal_mask};
use xlstm::blocks::load_pytorch::PyTorchLoader;

type B = Flex<f32>;

fn forward_train_partial_rope(
    input: Tensor<B, 2, Int>,
    embedding: &burn::nn::Embedding<B>,
    transformer: &xlstm::blocks::trasformer::layer::Transformer<B>,
    head: &burn::nn::Linear<B>,
    x0_lambdas: &Tensor<B, 2>,
    rotary_pct: f64,
) -> Tensor<B, 3> {
    let x = embedding.forward(input);
    let [batch, seq_len, _d] = x.dims();
    let x0 = x.clone();

    let mut h = x;
    for (i, layer) in transformer.layers.iter().enumerate() {
        let residual = h.clone();
        let h_norm = layer.attn_norm.forward(h);

        let (q, k, v) = layer.attention.qkv.forward(h_norm);
        let (q, k) = apply_rope_partial(q, k, 0, rotary_pct);

        let k = repeat_kv(k, layer.attention.num_heads, layer.attention.num_kv_groups);
        let v = repeat_kv(v, layer.attention.num_heads, layer.attention.num_kv_groups);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (layer.attention.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        if seq_len > 1 {
            scores = apply_causal_mask(scores, seq_len);
        }
        let attn = softmax(scores, 3);
        let h_attn = attn.matmul(v);
        let h_attn = h_attn.swap_dims(1, 2);
        let h_attn = layer.attention.o_proj.forward(h_attn);
        h = residual + h_attn;

        let residual = h.clone();
        let h_norm = layer.ffn_norm.forward(h);
        let h_ffn = layer.ffn.forward(h_norm);
        h = residual + h_ffn;

        let lam = x0_lambdas.clone().slice([0..1, i..(i+1)]).unsqueeze_dim::<3>(2);
        h = h + lam * x0.clone();
    }

    let h = transformer.final_norm.forward(h);
    head.forward(h)
}

fn npy_load_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let hdr = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
    Ok(bytes[hdr..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn npy_load_i32(path: &str) -> Result<Vec<i32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let hdr = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
    Ok(bytes[hdr..].chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn npy_save_f32(path: &str, data: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut magic = b"\x93NUMPY".to_vec();
    // v3.0: 2 version bytes, 4 byte header len
    magic.push(3);  // major
    magic.push(0);  // minor
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", data.len());
    let pad = 64 - (magic.len() + 4 + header.len()) % 64;
    let header = header + &" ".repeat(pad) + "\n";
    let mut out = magic;
    out.extend_from_slice(&(header.as_bytes().len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in data { out.extend_from_slice(&v.to_le_bytes()); }
    fs::write(path, out)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();

    // Load model
    let tokenizer = xlstm::blocks::trasformer_bit::model::Tokenizer::load("python/tokenizer.json")?;
    let vocab_size = tokenizer.vocab_size();

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
    let x0_lambdas = Tensor::<B, 2>::zeros([1, 6], &device);

    let tensors = PyTorchLoader::load_safetensors(
        "python/model_test.safetensors", "python/model_test_mapping.json")?;
    PyTorchLoader::load_into_transformer(&mut transformer, &tensors, 6, &device)?;
    if let Some(d) = tensors.get("embedding.weight") {
        embedding.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), &device));
    }
    if let Some(d) = tensors.get("head.weight") {
        head.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), &device).transpose());
    }

    // Load batch from Python
    let input_ids_data = npy_load_i32("test_data/training_input_ids.npy")?;
    let targets_data = npy_load_i32("test_data/training_targets.npy")?;
    let batch_size = 1;
    let seq_len = 128;
    println!("Loaded {} input_ids, {} targets", input_ids_data.len(), targets_data.len());

    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_ids_data.iter().map(|&i| i as i64).collect(), [batch_size, seq_len]), &device);
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets_data.iter().map(|&i| i as i64).collect(), [batch_size, seq_len]), &device);

    // Run forward_train_partial_rope (same as training)
    let logits = forward_train_partial_rope(
        input, &embedding, &transformer, &head, &x0_lambdas, 1.0);

    // Compute loss
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);
    let targets_flat = targets.reshape([batch_size * seq_len]);
    let loss = loss_fn.forward(logits_flat, targets_flat);
    let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];
    println!("\n--- Rust forward_train_partial_rope ---");
    println!("Loss: {:.4}", loss_val);

    // Save logits for Python comparison
    let logits_data: Vec<f32> = logits.to_data().as_slice::<f32>().unwrap().to_vec();
    let total = batch_size * seq_len * vocab_size;
    assert_eq!(logits_data.len(), total, "logits len mismatch");
    npy_save_f32("test_data/logits_rust_partial.npy", &logits_data)?;
    println!("Logits saved to test_data/logits_rust_partial.npy");
    println!("Rust logits [0,:5]: {:?}", &logits_data[..5]);

    Ok(())
}
