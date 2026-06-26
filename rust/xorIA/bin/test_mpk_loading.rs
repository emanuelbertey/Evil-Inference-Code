use std::fs;
use std::error::Error;
use burn::{
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData, Int, activation::softmax, module::Param},
    nn::{LinearConfig, EmbeddingConfig, loss::CrossEntropyLossConfig},
};
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use xlstm::blocks::trasformer::layer::{TransformerConfig, TransformerLayerConfig};
use xlstm::blocks::trasformer_bit::ops::{apply_rope_partial, repeat_kv, apply_causal_mask};

type B = Autodiff<Flex<f32>>;

fn npy_load_i32(path: &str) -> Result<Vec<i32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let hdr = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
    Ok(bytes[hdr..].chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    let vocab_size = 16000;

    let layer_cfg = TransformerLayerConfig {
        d_model: 256, num_heads: 8, num_kv_groups: 4, head_dim: None,
        ffn_expansion: 4.0, use_swiglu: true, max_seq_len: 128,
        rope_base: 10000.0, rope_scaling: 1.0, causal: true,
        attn_dropout: 0.0, ffn_dropout: 0.0, residual_dropout: 0.0,
        attn_logit_cap: None, bias: false, norm_eps: 1e-5, ffn_round_to: 64,
    };
    let num_layers = 6;
    let mut model = (
        EmbeddingConfig::new(vocab_size, 256).init(&device),
        TransformerConfig { num_layers, layer: layer_cfg }.init(&device),
        LinearConfig::new(256, vocab_size).with_bias(false).init(&device),
        Param::from_tensor(Tensor::<B, 2>::zeros([1, num_layers], &device)),
    );

    println!("Cargando desde transformer_chat.mpk...");
    let record = CompactRecorder::new().load("transformer_chat.mpk".into(), &device)?;
    // Can't use load_record on a tuple, need a Module
    // Let's use individual load
    let mut embedding = burn::module::Module::load_record(model.0, record.clone());
    let mut transformer = burn::module::Module::load_record(model.1, record.clone());
    let mut head = burn::module::Module::load_record(model.2, record.clone());
    let mut x0_lambdas = burn::module::Module::load_record(model.3, record);
    println!("Cargado!");

    // Load batch from Python
    let input_ids_data = npy_load_i32("test_data/training_input_ids.npy")?;
    let targets_data = npy_load_i32("test_data/training_targets.npy")?;
    let seq_len = 128;
    let batch_size = 1;

    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_ids_data.iter().map(|&i| i as i64).collect(), [batch_size, seq_len]), &device);
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets_data.iter().map(|&i| i as i64).collect(), [batch_size, seq_len]), &device);

    // Forward
    let x = embedding.forward(input);
    let [_, seq_len_fwd, _] = x.dims();
    let x0 = x.clone();
    let mut h = x;
    for (i, layer) in transformer.layers.iter().enumerate() {
        let residual = h.clone();
        let h_norm = layer.attn_norm.forward(h);
        let (q, k, v) = layer.attention.qkv.forward(h_norm);
        let (q, k) = apply_rope_partial(q, k, 0, 1.0);
        let k = repeat_kv(k, layer.attention.num_heads, layer.attention.num_kv_groups);
        let v = repeat_kv(v, layer.attention.num_heads, layer.attention.num_kv_groups);
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        let scale = (layer.attention.head_dim as f64).sqrt();
        let mut scores = q.matmul(k.transpose()) / scale;
        if seq_len_fwd > 1 { scores = apply_causal_mask(scores, seq_len_fwd); }
        let attn = softmax(scores, 3);
        let h_attn = attn.matmul(v);
        let h_attn = h_attn.swap_dims(1, 2);
        let h_attn = layer.attention.o_proj.forward(h_attn);
        h = residual + h_attn;
        let residual = h.clone();
        let h_norm = layer.ffn_norm.forward(h);
        let h_ffn = layer.ffn.forward(h_norm);
        h = residual + h_ffn;
        let lam = x0_lambdas.val().slice([0..1, i..(i+1)]).unsqueeze_dim::<3>(2);
        h = h + lam * x0.clone();
    }
    let h = transformer.final_norm.forward(h);
    let logits = head.forward(h);

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
    let targets_flat = targets.reshape([batch_size * seq_len]);
    let loss = loss_fn.forward(logits_flat, targets_flat);
    let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];
    println!("Loss: {:.4}", loss_val);

    let expected = 3.7981f32;
    if (loss_val - expected).abs() < 0.1 {
        println!("OK: MPK weights match safetensors");
    } else {
        println!("ERROR: MPK weights differ from safetensors (expected ~{:.4})", expected);
    }
    Ok(())
}
