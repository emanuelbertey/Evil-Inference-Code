use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use burn::prelude::*;
use burn_ndarray::NdArray;
use burn::module::{Module, Param};

use xlstm::blocks::xlstm_large::{XLSTMLarge, XLSTMLargeConfig};
use xlstm::blocks::xlstm_large::model::{FeedForwardWeightsRecord, XLSTMLargeState};
use xlstm::blocks::xlstm_large::layer::WeightModeRecord;
use tokenizers::tokenizer::Tokenizer as HFTokenizer;

type MyBackend = NdArray<f32>;

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== xLSTMLarge SEQUENTIAL PARITY TEST (Step-by-Step) ===");
    let filepath = "../xlstm/testbin_model/sequential_data.bin";
    let device = Default::default();

    let mut file = File::open(filepath)?;

    // 1. Número de pasos
    let steps = read_u32(&mut file)? as usize;

    // 2. Leer tokens de entrada X
    let x_len = read_u32(&mut file)? as usize;
    let mut x_bytes = vec![0u8; x_len * 4];
    file.read_exact(&mut x_bytes)?;
    let mut tokens_input = vec![0i32; x_len];
    for i in 0..x_len {
        tokens_input[i] = i32::from_le_bytes(x_bytes[i*4..(i+1)*4].try_into().unwrap());
    }

    // 3. Leer Lista de Logits Y (referencia Python)
    let mut expected_logits_list = Vec::new();
    for _ in 0..steps {
        let size = read_u32(&mut file)? as usize;
        let mut data_bytes = vec![0u8; size];
        file.read_exact(&mut data_bytes)?;
        let data: Vec<f32> = data_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        expected_logits_list.push(data);
    }

    // 4. Leer State Dict
    let num_tensors = read_u32(&mut file)?;
    let mut map = HashMap::new();
    for _ in 0..num_tensors {
        let name_len = read_u32(&mut file)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).unwrap();

        let shape_len = read_u32(&mut file)? as usize;
        let mut shape = Vec::new();
        for _ in 0..shape_len { shape.push(read_u32(&mut file)? as usize); }

        let data_len = read_u32(&mut file)? as usize;
        let mut data_bytes = vec![0u8; data_len];
        file.read_exact(&mut data_bytes)?;
        let data: Vec<f32> = data_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        map.insert(name, (shape, data));
    }

    // Configuración
    let tokenizer = HFTokenizer::from_file("tokenizer.json").expect("No se encontró el archivo tokenizer.json");
    let vocab_size = tokenizer.get_vocab_size(true);

    let config = XLSTMLargeConfig {
        embedding_dim: 128, num_heads: 2, num_blocks: 2, vocab_size,
        use_bias: false, norm_eps: 1e-6, norm_reduction_force_float32: true,
        add_out_norm: true, qk_dim_factor: 0.5, v_dim_factor: 1.0,
        mlstm_backend: xlstm::blocks::xlstm_large::config::MLSTMBackendConfig::new(),
        ffn_proj_factor: 2.6667, ffn_round_up_to_multiple_of: 64,
        gate_soft_cap: Some(15.0), output_logit_soft_cap: Some(30.0),
        weight_mode: "single".to_string(),
    };

    let model_init = XLSTMLarge::<MyBackend>::init(&config, &device);
    let mut record = model_init.into_record();

    // 1. Embedding
    if let Some((shape, data)) = map.get("embedding.weight") {
        let e = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]);
        record.embedding.weight = Param::from_tensor(e);
    }
    // 2. LM Head
    if let Some((shape, data)) = map.get("lm_head.weight") {
        let w = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose();
        record.lm_head.weight = Param::from_tensor(w);
    }

    // 3. Blocks
    for i in 0..config.num_blocks {
        let b_rec = &mut record.blocks[i];
        if let Some((_, data)) = map.get(&format!("backbone.blocks.{}.norm_mlstm.weight", i)) {
            b_rec.norm_mlstm.weight = Some(Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)));
        }
        if let Some((_, data)) = map.get(&format!("backbone.blocks.{}.norm_ffn.weight", i)) {
            b_rec.norm_ffn.weight = Some(Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)));
        }
        if let FeedForwardWeightsRecord::Single(single) = &mut b_rec.ffn.weights {
            if let Some((shape, data)) = map.get(&format!("backbone.blocks.{}.ffn.proj_up_gate.weight", i)) {
                single.proj_up_gate.weight = Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose());
            }
            if let Some((shape, data)) = map.get(&format!("backbone.blocks.{}.ffn.proj_up.weight", i)) {
                single.proj_up.weight = Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose());
            }
        }
        if let Some((shape, data)) = map.get(&format!("backbone.blocks.{}.ffn.proj_down.weight", i)) {
            b_rec.ffn.proj_down.weight = Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose());
        }

        if let WeightModeRecord::Single(w) = &mut b_rec.mlstm_layer.weights {
            let names = ["q", "k", "v", "igate_preact", "fgate_preact", "ogate_preact"];
            let targets = [&mut w.q, &mut w.k, &mut w.v, &mut w.igate, &mut w.fgate, &mut w.ogate];
            for (idx, name) in names.iter().enumerate() {
                if let Some((shape, data)) = map.get(&format!("backbone.blocks.{}.mlstm_layer.{}.weight", i, name)) {
                    targets[idx].weight = Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose());
                }
                if let Some((_, data)) = map.get(&format!("backbone.blocks.{}.mlstm_layer.{}.bias", i, name)) {
                    targets[idx].bias = Some(Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)));
                }
            }
        }
        if let Some((_, data)) = map.get(&format!("backbone.blocks.{}.mlstm_layer.multihead_norm.weight", i)) {
            b_rec.mlstm_layer.outnorm.norm.weight = Some(Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)));
        }
        if let Some((shape, data)) = map.get(&format!("backbone.blocks.{}.mlstm_layer.out_proj.weight", i)) {
            b_rec.mlstm_layer.out_proj.weight = Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]).transpose());
        }
    }
    if let Some(rn) = &mut record.out_norm {
        if let Some((_, data)) = map.get("backbone.out_norm.weight") {
            rn.weight = Some(Param::from_tensor(Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)));
        }
    }

    let model = XLSTMLarge::<MyBackend>::init(&config, &device).load_record(record);
    println!("¡Modelo inyectado con éxito!");

    // --- EJECUCIÓN SECUENCIAL PASO A PASO ---
    let mut state: Option<XLSTMLargeState<MyBackend>> = None;
    let mut max_diff_global: f32 = 0.0;

    println!("\nIniciando bucle de comparación secuencial ({} pasos)...", steps);

    for i in 0..steps {
        let token_val = tokens_input[i];
        let input_tensor = Tensor::<MyBackend, 1, Int>::from_data([token_val as i64], &device).reshape([1, 1]);

        let (logits, next_state) = model.forward(input_tensor, state);
        state = next_state;

        let rust_logits = logits.into_data().as_slice::<f32>().unwrap().to_vec();
        let py_logits = &expected_logits_list[i];

        let mut step_max_diff: f32 = 0.0;
        for (r, p) in rust_logits.iter().zip(py_logits.iter()) {
            let d = (r - p).abs();
            if d > step_max_diff { step_max_diff = d; }
        }

        if step_max_diff > max_diff_global { max_diff_global = step_max_diff; }
        println!("  Paso {:2} | Token In: {:3} | Max Diff este paso: {:.10}", i + 1, token_val, step_max_diff);
    }

    println!("\n==========================================");
    println!("Diferencia Máxima Acumulada Final: {:.10}", max_diff_global);
    if max_diff_global < 1e-2 {
        println!("✅ SECUENCIAL PASSED (La memoria es estable)");
    } else {
        println!("❌ SECUENCIAL FAILED (El error está divergiendo)");
    }

    Ok(())
}
