use std::collections::HashMap;
use std::fs;
use std::error::Error;
use std::borrow::Cow;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use safetensors::{SafeTensors, Dtype, serialize};
use safetensors::tensor::View;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct PyTorchLoader;

#[derive(serde::Deserialize, serde::Serialize)]
pub struct MappingEntry {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub burn_name: String,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct MappingFile {
    pub parameters: HashMap<String, MappingEntry>,
    #[serde(default)]
    pub config: Option<HashMap<String, serde_json::Value>>,
}

pub struct OwnedTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: Dtype,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype { self.dtype }
    fn shape(&self) -> &[usize] { &self.shape }
    fn data(&self) -> Cow<[u8]> { Cow::Borrowed(&self.bytes) }
    fn data_len(&self) -> usize { self.bytes.len() }
}

impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype { self.dtype }
    fn shape(&self) -> &[usize] { &self.shape }
    fn data(&self) -> Cow<[u8]> { Cow::Borrowed(&self.bytes) }
    fn data_len(&self) -> usize { self.bytes.len() }
}

impl PyTorchLoader {
    pub fn load_safetensors(
        path: &str,
        mapping_path: &str,
    ) -> Result<HashMap<String, TensorData>> {
        let data = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mapping_data = fs::read_to_string(mapping_path)?;
        let mapping: MappingFile = serde_json::from_str(&mapping_data)?;

        let mut result = HashMap::new();

        for (py_name, entry) in &mapping.parameters {
            let tensor_view = tensors.tensor(&entry.burn_name)?;
            let raw_bytes = tensor_view.data();
            let raw_f32: Vec<f32> = raw_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            result.insert(py_name.clone(), TensorData::new(raw_f32, entry.shape.clone()));
        }

        Ok(result)
    }

    pub fn load_into_transformer<B: Backend>(
        model: &mut crate::blocks::trasformer::layer::Transformer<B>,
        tensors: &HashMap<String, TensorData>,
        num_layers: usize,
        device: &B::Device,
    ) -> Result<()> {
        use crate::blocks::trasformer::feedforward::FeedForwardBlock;

        for i in 0..num_layers {
            let prefix = format!("transformer.layers.{}", i);

            if let Some(d) = tensors.get(&format!("{}.attn_norm.weight", prefix)) {
                model.layers[i].attn_norm.gamma = burn::module::Param::from_tensor(Tensor::<B, 1>::from_data(d.clone(), device));
            }

            if let Some(d) = tensors.get(&format!("{}.attention.qkv.q_proj.weight", prefix)) {
                model.layers[i].attention.qkv.q_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
            }
            if let Some(d) = tensors.get(&format!("{}.attention.qkv.k_proj.weight", prefix)) {
                model.layers[i].attention.qkv.k_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
            }
            if let Some(d) = tensors.get(&format!("{}.attention.qkv.v_proj.weight", prefix)) {
                model.layers[i].attention.qkv.v_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
            }
            if let Some(d) = tensors.get(&format!("{}.attention.o_proj.o_proj.weight", prefix)) {
                model.layers[i].attention.o_proj.o_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
            }

            // infer real head_dim/num_heads from loaded weight shapes
            // after transpose, weight shape is [d_model, out_dim]
            let k_out = model.layers[i].attention.qkv.k_proj.weight.val().dims()[1];
            let hd = k_out / model.layers[i].attention.qkv.num_kv_groups;
            let q_out = model.layers[i].attention.qkv.q_proj.weight.val().dims()[1];
            let nh = q_out / hd;
            model.layers[i].attention.qkv.head_dim = hd;
            model.layers[i].attention.qkv.num_heads = nh;
            model.layers[i].attention.num_heads = nh;
            model.layers[i].attention.head_dim = hd;
            model.layers[i].attention.o_proj.num_heads = nh;
            model.layers[i].attention.o_proj.head_dim = hd;
            // re-init RoPE with correct head_dim
            let max_seq = model.layers[i].attention.max_seq_len;
            let rope_base = model.layers[i].attention.rope_base;
            let rope_scaling = model.layers[i].attention.rope_scaling;
            let new_rope: crate::blocks::trasformer::rope::RoPE<B> = crate::blocks::trasformer::rope::RoPEConfig {
                head_dim: hd,
                max_seq_len: max_seq,
                base: rope_base,
                scaling_factor: rope_scaling,
            }.init(device);
            model.layers[i].attention.rope = new_rope;

            if let Some(d) = tensors.get(&format!("{}.ffn_norm.weight", prefix)) {
                model.layers[i].ffn_norm.gamma = burn::module::Param::from_tensor(Tensor::<B, 1>::from_data(d.clone(), device));
            }

            match &mut model.layers[i].ffn {
                FeedForwardBlock::SwiGLU(ref mut swiglu_ffn) => {
                    if let Some(d) = tensors.get(&format!("{}.ffn.gate_proj.weight", prefix)) {
                        swiglu_ffn.swiglu.linear_inner.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
                    }
                    if let Some(d) = tensors.get(&format!("{}.ffn.up_proj.weight", prefix)) {
                        swiglu_ffn.swiglu.linear_outer.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
                    }
                    if let Some(d) = tensors.get(&format!("{}.ffn.down_proj.weight", prefix)) {
                        swiglu_ffn.down_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
                    }
                }
                FeedForwardBlock::Standard(ref mut std_ffn) => {
                    if let Some(d) = tensors.get(&format!("{}.ffn.up_proj.weight", prefix)) {
                        std_ffn.up_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
                    }
                    if let Some(d) = tensors.get(&format!("{}.ffn.down_proj.weight", prefix)) {
                        std_ffn.down_proj.weight = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(d.clone(), device).transpose());
                    }
                }
            }
        }

        if let Some(d) = tensors.get("transformer.final_norm.weight") {
            model.final_norm.gamma = burn::module::Param::from_tensor(Tensor::<B, 1>::from_data(d.clone(), device));
        }

        Ok(())
    }

    pub fn export_from_burn<B: Backend>(
        model: &crate::blocks::trasformer::layer::Transformer<B>,
        output_path: &str,
        mapping_path: &str,
        num_layers: usize,
    ) -> Result<()> {
        let mut tensors_map: HashMap<String, Vec<f32>> = HashMap::new();
        let mut mapping_params: HashMap<String, MappingEntry> = HashMap::new();

        for i in 0..num_layers {
            let prefix = format!("transformer.layers.{}", i);

            let attn_norm_w: Tensor<B, 2> = model.layers[i].attn_norm.gamma.val().clone().unsqueeze();
            add_tensor_helper(&format!("{}.attn_norm.weight", prefix), &attn_norm_w, &mut tensors_map, &mut mapping_params);

            let q_w: Tensor<B, 2> = model.layers[i].attention.qkv.q_proj.weight.val().clone();
            add_tensor_helper(&format!("{}.attention.qkv.q_proj.weight", prefix), &q_w, &mut tensors_map, &mut mapping_params);

            let k_w: Tensor<B, 2> = model.layers[i].attention.qkv.k_proj.weight.val().clone();
            add_tensor_helper(&format!("{}.attention.qkv.k_proj.weight", prefix), &k_w, &mut tensors_map, &mut mapping_params);

            let v_w: Tensor<B, 2> = model.layers[i].attention.qkv.v_proj.weight.val().clone();
            add_tensor_helper(&format!("{}.attention.qkv.v_proj.weight", prefix), &v_w, &mut tensors_map, &mut mapping_params);

            let o_w: Tensor<B, 2> = model.layers[i].attention.o_proj.o_proj.weight.val().clone();
            add_tensor_helper(&format!("{}.attention.o_proj.o_proj.weight", prefix), &o_w, &mut tensors_map, &mut mapping_params);

            let ffn_norm_w: Tensor<B, 2> = model.layers[i].ffn_norm.gamma.val().clone().unsqueeze();
            add_tensor_helper(&format!("{}.ffn_norm.weight", prefix), &ffn_norm_w, &mut tensors_map, &mut mapping_params);
        }

        let final_norm_w: Tensor<B, 2> = model.final_norm.gamma.val().clone().unsqueeze();
        add_tensor_helper("transformer.final_norm.weight", &final_norm_w, &mut tensors_map, &mut mapping_params);

        let mut st_owned: HashMap<String, OwnedTensor> = HashMap::new();
        for (name, floats) in &tensors_map {
            let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            let shape = mapping_params[name].shape.clone();
            st_owned.insert(name.clone(), OwnedTensor { bytes, shape, dtype: Dtype::F32 });
        }

        let bytes = serialize(&st_owned, &None)?;
        fs::write(output_path, &bytes)?;

        let mapping_file = MappingFile { parameters: mapping_params, config: None };
        let mapping_json = serde_json::to_string_pretty(&mapping_file)?;
        fs::write(mapping_path, mapping_json)?;

        println!("Exported {} tensors to {} + {}", tensors_map.len(), output_path, mapping_path);
        Ok(())
    }
}

fn add_tensor_helper<B: Backend>(
    name: &str,
    tensor: &Tensor<B, 2>,
    tensors_map: &mut HashMap<String, Vec<f32>>,
    mapping: &mut HashMap<String, MappingEntry>,
) {
    let data = tensor.to_data();
    let raw: Vec<f32> = data.as_slice::<f32>().unwrap().to_vec();
    let shape: Vec<usize> = tensor.shape().dims::<2>().to_vec();
    tensors_map.insert(name.to_string(), raw);
    mapping.insert(name.to_string(), MappingEntry {
        shape,
        dtype: "f32".to_string(),
        burn_name: name.to_string(),
    });
}
