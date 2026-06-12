use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use burn::prelude::*;
use burn_flex::Flex;
use burn::module::{Module, Param};

use xlstm::blocks::xlstm_large::{XLSTMLarge, XLSTMLargeConfig};
use xlstm::blocks::xlstm_large::model::FeedForwardWeightsRecord;
use xlstm::blocks::xlstm_large::layer::WeightModeRecord;
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
type MyBackend = Flex<f32>;
//type MyBackend = Flex<f16>;
use burn::tensor::f16;
//type MyBackend = Flex;

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn load_test_data(filepath: &str) -> std::io::Result<(Vec<usize>, Vec<i32>, Vec<usize>, Vec<f32>, HashMap<String, (Vec<usize>, Vec<f32>)>)> {
    let mut file = File::open(filepath)?;

    let x_shape_len = read_u32(&mut file)? as usize;
    let mut x_shape = Vec::new();
    for _ in 0..x_shape_len { x_shape.push(read_u32(&mut file)? as usize); }
    let x_bytes_len = read_u32(&mut file)? as usize;
    let mut x_bytes = vec![0u8; x_bytes_len];
    file.read_exact(&mut x_bytes)?;
    let mut x_data = vec![0i32; x_bytes_len / 4];
    for i in 0..x_data.len() {
        x_data[i] = i32::from_le_bytes(x_bytes[i*4..(i+1)*4].try_into().unwrap());
    }

    let y_shape_len = read_u32(&mut file)? as usize;
    let mut y_shape = Vec::new();
    for _ in 0..y_shape_len { y_shape.push(read_u32(&mut file)? as usize); }
    let y_bytes_len = read_u32(&mut file)? as usize;
    let mut y_bytes = vec![0u8; y_bytes_len];
    file.read_exact(&mut y_bytes)?;
    let mut y_data = vec![0.0f32; y_bytes_len / 4];
    for i in 0..y_data.len() {
        y_data[i] = f32::from_le_bytes(y_bytes[i*4..(i+1)*4].try_into().unwrap());
    }

    let num_tensors = read_u32(&mut file)?;
    let mut state_dict = HashMap::new();
    for _ in 0..num_tensors {
        let name_len = read_u32(&mut file)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).unwrap();

        let shape_len = read_u32(&mut file)? as usize;
        let mut shape = Vec::new();
        for _ in 0..shape_len { shape.push(read_u32(&mut file)? as usize); }

        let data_bytes_len = read_u32(&mut file)? as usize;
        let mut data_bytes = vec![0u8; data_bytes_len];
        file.read_exact(&mut data_bytes)?;
        let mut data = vec![0.0f32; data_bytes_len / 4];
        for i in 0..data.len() {
            data[i] = f32::from_le_bytes(data_bytes[i*4..(i+1)*4].try_into().unwrap());
        }
        state_dict.insert(name, (shape, data));
    }
    
    Ok((x_shape, x_data, y_shape, y_data, state_dict))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== xLSTMLarge FLEX 16-BIT EQUIVALENCE TEST (Burn v0.21.0-pre.4) ===");
    let filepath = "../xlstm/testbin_model/test_data.bin";
    let device = Default::default();

    println!("Cargando Tokenizer de HuggingFace...");
    let tokenizer_path = "tokenizer.json";
    let tokenizer = HFTokenizer::from_file(tokenizer_path).expect("No se encontró el tokenizer.json");
    let vocab_size = tokenizer.get_vocab_size(true);

    let (x_shape, x_data, y_shape, y_data, map) = load_test_data(filepath)?;

    let config = XLSTMLargeConfig {
        embedding_dim: 128,
        num_heads: 2,
        num_blocks: 2,
        vocab_size,
        use_bias: true,
        norm_eps: 1e-6,
        norm_reduction_force_float32: true,
        add_out_norm: true,
        qk_dim_factor: 0.5,
        v_dim_factor: 1.0,
        mlstm_backend: xlstm::blocks::xlstm_large::config::MLSTMBackendConfig::new().with_chunk_size(16),
        ffn_proj_factor: 2.6667,
        ffn_round_up_to_multiple_of: 64,
        gate_soft_cap: Some(15.0),
        output_logit_soft_cap: Some(30.0),
        weight_mode: "single".to_string(),
    };

    let model_init = XLSTMLarge::<MyBackend>::init(&config, &device);
    let mut record = model_init.into_record();

    macro_rules! load_linear {
        ($rec:expr, $prefix:expr) => {
            if let Some((shape, data)) = map.get(&format!("{}.weight", $prefix)) {
                let w = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device)
                    .reshape([shape[0], shape[1]])
                    .transpose(); 
                $rec.weight = Param::from_tensor(w);
            }
            if let Some((shape, data)) = map.get(&format!("{}.bias", $prefix)) {
                let b = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0]]);
                $rec.bias = Some(Param::from_tensor(b));
            } else {
                $rec.bias = None;
            }
        }
    }

    macro_rules! load_norm_weight {
        ($rec:expr, $prefix:expr) => {
            if let Some((shape, data)) = map.get(&format!("{}.weight", $prefix)) {
                let w = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0]]);
                $rec.weight = Some(Param::from_tensor(w));
            }
        }
    }

    macro_rules! load_norm_weight_bias {
        ($rec:expr, $prefix:expr) => {
            load_norm_weight!($rec, $prefix);
            if let Some((shape, data)) = map.get(&format!("{}.bias", $prefix)) {
                let b = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0]]);
                $rec.bias = Some(Param::from_tensor(b));
            }
        }
    }

    // 1. Embedding
    if let Some((shape, data)) = map.get("embedding.weight") {
        let e = Tensor::<MyBackend, 1>::from_floats(data.as_slice(), &device).reshape([shape[0], shape[1]]);
        record.embedding.weight = Param::from_tensor(e);
    }
    // 2. LM Head
    load_linear!(record.lm_head, "lm_head");

    // 3. Blocks
    for i in 0..config.num_blocks {
        let b_rec = &mut record.blocks[i];
        
        load_norm_weight_bias!(b_rec.norm_mlstm, format!("backbone.blocks.{}.norm_mlstm", i));
        load_norm_weight_bias!(b_rec.norm_ffn, format!("backbone.blocks.{}.norm_ffn", i));
        
        // FFN
        if let FeedForwardWeightsRecord::Single(single) = &mut b_rec.ffn.weights {
            load_linear!(single.proj_up_gate, format!("backbone.blocks.{}.ffn.proj_up_gate", i));
            load_linear!(single.proj_up, format!("backbone.blocks.{}.ffn.proj_up", i));
        }
        load_linear!(b_rec.ffn.proj_down, format!("backbone.blocks.{}.ffn.proj_down", i));

        // MLSTM Layer
        if let WeightModeRecord::Single(w) = &mut b_rec.mlstm_layer.weights {
            load_linear!(w.q, format!("backbone.blocks.{}.mlstm_layer.q", i));
            load_linear!(w.k, format!("backbone.blocks.{}.mlstm_layer.k", i));
            load_linear!(w.v, format!("backbone.blocks.{}.mlstm_layer.v", i));
            load_linear!(w.igate, format!("backbone.blocks.{}.mlstm_layer.igate_preact", i));
            load_linear!(w.fgate, format!("backbone.blocks.{}.mlstm_layer.fgate_preact", i));
            load_linear!(w.ogate, format!("backbone.blocks.{}.mlstm_layer.ogate_preact", i));
        }
        
        load_norm_weight_bias!(b_rec.mlstm_layer.outnorm.norm, format!("backbone.blocks.{}.mlstm_layer.multihead_norm", i));
        load_linear!(b_rec.mlstm_layer.out_proj, format!("backbone.blocks.{}.mlstm_layer.out_proj", i));
    }
    
    // Si out_norm existe 
    if let Some(rn) = &mut record.out_norm {
        load_norm_weight_bias!(rn, "backbone.out_norm");
    }

    let model = XLSTMLarge::<MyBackend>::init(&config, &device).load_record(record);
    println!("¡Modelo inyectado con los pesos exactos de Python a Rust (FLEX 16-BIT)!");

    // PARTE 1: VERIFICAR EQUIVALENCIA LOGITS
    let b = x_shape[0];
    let s = x_shape[1];
    
    let x_tensor = Tensor::<MyBackend, 1, Int>::from_data(burn::tensor::TensorData::new(x_data.clone(), [b * s]), &device)
        .reshape([b, s]);
        
    let expected_y = Tensor::<MyBackend, 1>::from_floats(y_data.as_slice(), &device)
        .reshape([y_shape[0], y_shape[1], y_shape[2]]);
        
    let (logits, _) = model.forward(x_tensor.clone(), None);

    let diff = (logits.clone() - expected_y.clone()).abs().max().into_scalar();

    println!("Max Diff entre Logits (Python vs Rust - FLEX): {:.10}", diff);
    
    if diff < 1e-4 {
        println!("✅ EQUIVALENCE PASSED");
    } else {
        println!("❌ EQUIVALENCE FAILED (Diff es mayor a lo aceptable)");
    }

    // PARTE 2: GENERACIÓN DE TEXTO
    println!("\nGenerando texto en Rust usando 'xLSTMLarge::generate' (FLEX 16-BIT)...");
    let input_text = "The";
    let encoding = tokenizer.encode(input_text, false).unwrap();
    let tokens_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

    let input_tensor = Tensor::<MyBackend, 1, Int>::from_data(
        burn::tensor::TensorData::new(tokens_ids.clone(), [tokens_ids.len()]), &device
    ).reshape([1, tokens_ids.len()]);

    println!("Generando 300 tokens...");
    let start_time = std::time::Instant::now();
    let generated_tensor = model.generate(input_tensor, 300 , &device);
    let duration = start_time.elapsed();
    let tps = 300.0 / duration.as_secs_f64();

    let gen_ids: Vec<u32> = generated_tensor.into_data().as_slice::<i32>().unwrap().iter().map(|&v| v as u32).collect();
    let output_text = tokenizer.decode(&gen_ids, true).unwrap();
    
    println!("Input: '{}'", input_text);
    println!("Generado: '{}'", output_text);
    println!("--- Rendimiento Rust (FLEX 16-BIT) ---");
    println!("Velocidad: {:.2} tokens/segundo | Tiempo total: {:.2?}", tps, duration);

    Ok(())
}
