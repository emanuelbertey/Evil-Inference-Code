use std::io::{self, Read, BufReader};
use std::fs::File;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn::module::Param;

use crate::blocks::bitlinear::kernel::KernelKind;
use crate::blocks::bitlinear::layer::{BitLinearInferenceState, BitLinear, RMSNorm};
use super::model::{
    TransformerBitLinearLM, TransformerInferenceState, BitLinearTransformerStack,
    BitLinearTransformerLayer, BitLinearRMSNorm, BitLinearQKVProjection,
    BitLinearOutputProjection, BitLinearSwiGLUFeedForward,
};

const MAGIC: &[u8; 4] = b"BTNT";
const VERSION: u32 = 2;

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(r: &mut R, len: usize) -> io::Result<Vec<f32>> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len { v.push(read_f32(r)?); }
    Ok(v)
}

fn f16_to_f32(v: u16) -> f32 {
    let sign = (v >> 15) as u32;
    let exp = ((v >> 10) & 0x1F) as u32;
    let mant = (v & 0x03FF) as u32;
    if exp == 0 {
        f32::from_bits((sign << 31) | ((mant as f32 * 0.00006103515625_f32).to_bits() & 0x007FFFFF))
    } else if exp == 31 {
        if mant == 0 { f32::from_bits((sign << 31) | 0x7F800000) }
        else { f32::from_bits((sign << 31) | 0x7FC00000) }
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

fn read_f16<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(f16_to_f32(u16::from_le_bytes(buf)))
}

fn read_f16_vec<R: Read>(r: &mut R, len: usize) -> io::Result<Vec<f32>> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len { v.push(read_f16(r)?); }
    Ok(v)
}

/// Read one BitLinear bin from the file (rms_f32, packed_w, scales).
fn read_bitlinear_bin<R: Read>(r: &mut R) -> io::Result<(Vec<f32>, Vec<u32>, Vec<f32>)> {
    let _in_f = read_u32(r)? as usize;
    let _out_f = read_u32(r)? as usize;
    let rms_len = read_u32(r)? as usize;
    let rms = read_f32_vec(r, rms_len)?;
    let packed_len = read_u32(r)? as usize;
    let mut packed = Vec::with_capacity(packed_len);
    for _ in 0..packed_len { packed.push(read_u32(r)?); }
    let scales_len = read_u32(r)? as usize;
    let scales = read_f32_vec(r, scales_len)?;
    Ok((rms, packed, scales))
}

/// Load a .bitnet file for inference only — no f32 weight expansion.
/// Reads the file once and returns (model, inference_state).
/// Model's BitLinear have `weight: None`; all ternary weights go to inference state.
pub fn load<B: Backend>(
    path: &str,
    device: &B::Device,
    layer_kernel: KernelKind,
    head_kernel: KernelKind,
) -> io::Result<(TransformerBitLinearLM<B>, TransformerInferenceState)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC { return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BitNet file")); }
    if read_u32(&mut r)? != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported version"));
    }

    let vocab_size = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let num_heads = read_u32(&mut r)? as usize;
    let num_kv_groups = read_u32(&mut r)? as usize;
    let head_dim = read_u32(&mut r)? as usize;
    let ffn_dim = read_u32(&mut r)? as usize;

    // Embedding (f16 → f32 tensor)
    let embed_len = read_u32(&mut r)? as usize;
    let embed_data = read_f16_vec(&mut r, embed_len)?;
    let embedding = burn::nn::Embedding {
        weight: Param::from_tensor(
            Tensor::<B, 2>::from_data(TensorData::new(embed_data, [vocab_size, d_model]), device)
        ),
    };

    let mut layers = Vec::new();
    let mut qkv_states = Vec::new();
    let mut o_proj_states = Vec::new();
    let mut ffn_gate_up_states = Vec::new();
    let mut ffn_down_states = Vec::new();

    for _ in 0..num_layers {
        // attn norm (f16)
        let _ = read_u32(&mut r)?;
        let attn_w = read_f16_vec(&mut r, d_model)?;
        let attn_norm = BitLinearRMSNorm {
            weight: Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(attn_w, [d_model]), device)),
            eps: 1e-5,
        };

        // ffn norm (f16)
        let _ = read_u32(&mut r)?;
        let ffn_w = read_f16_vec(&mut r, d_model)?;
        let ffn_norm = BitLinearRMSNorm {
            weight: Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(ffn_w, [d_model]), device)),
            eps: 1e-5,
        };

        // BitLinear bins (rms f32, packed_w u32, scales f32)
        let (q_rms, q_packed, q_scales) = read_bitlinear_bin(&mut r)?;
        let (k_rms, k_packed, k_scales) = read_bitlinear_bin(&mut r)?;
        let (v_rms, v_packed, v_scales) = read_bitlinear_bin(&mut r)?;
        let (o_rms, o_packed, o_scales) = read_bitlinear_bin(&mut r)?;
        let (gu_rms, gu_packed, gu_scales) = read_bitlinear_bin(&mut r)?;
        let (d_rms, d_packed, d_scales) = read_bitlinear_bin(&mut r)?;

        // Inference states (ternary packed → kernel)
        let q_state = BitLinearInferenceState::from_packed(q_packed, q_scales, d_model, num_heads * head_dim, layer_kernel);
        let k_state = BitLinearInferenceState::from_packed(k_packed, k_scales, d_model, num_kv_groups * head_dim, layer_kernel);
        let v_state = BitLinearInferenceState::from_packed(v_packed, v_scales, d_model, num_kv_groups * head_dim, layer_kernel);
        let o_state = BitLinearInferenceState::from_packed(o_packed, o_scales, num_heads * head_dim, d_model, layer_kernel);
        let gu_state = BitLinearInferenceState::from_packed(gu_packed, gu_scales, d_model, 2 * ffn_dim, layer_kernel);
        let d_state = BitLinearInferenceState::from_packed(d_packed, d_scales, ffn_dim, d_model, layer_kernel);

        qkv_states.push((q_state, k_state, v_state));
        o_proj_states.push(o_state);
        ffn_gate_up_states.push(gu_state);
        ffn_down_states.push(d_state);

        // BitLinear structs with weight=None, correct RMS norms
        let make_bl = |in_f: usize, out_f: usize, rms_data: Vec<f32>| -> BitLinear<B> {
            BitLinear {
                weight: None,
                bias: None,
                rms_norm: RMSNorm {
                    weight: Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(rms_data, [in_f]), device)),
                    eps: 1e-5,
                },
                activation_bits: 8,
                in_features: in_f,
                out_features: out_f,
                quantized: true,
            }
        };

        let dropout = burn::nn::DropoutConfig::new(0.0).init();

        layers.push(BitLinearTransformerLayer {
            attn_norm,
            qkv: BitLinearQKVProjection {
                q_proj: make_bl(d_model, num_heads * head_dim, q_rms),
                k_proj: make_bl(d_model, num_kv_groups * head_dim, k_rms),
                v_proj: make_bl(d_model, num_kv_groups * head_dim, v_rms),
                num_heads, num_kv_groups, head_dim,
            },
            o_proj: BitLinearOutputProjection {
                o_proj: make_bl(num_heads * head_dim, d_model, o_rms),
                num_heads, head_dim,
            },
            ffn_norm,
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: make_bl(d_model, 2 * ffn_dim, gu_rms),
                down_proj: make_bl(ffn_dim, d_model, d_rms),
                dropout,
                intermediate_dim: ffn_dim,
            },
            residual_dropout: burn::nn::DropoutConfig::new(0.0).init(),
        });
    }

    // Final norm (f16)
    let _ = read_u32(&mut r)?;
    let final_w = read_f16_vec(&mut r, d_model)?;
    let final_norm = BitLinearRMSNorm {
        weight: Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(final_w, [d_model]), device)),
        eps: 1e-5,
    };

    // Head BitLinear
    let (head_rms, head_packed, head_scales) = read_bitlinear_bin(&mut r)?;
    let head_state = BitLinearInferenceState::from_packed(head_packed, head_scales, d_model, vocab_size, head_kernel);
    let head = BitLinear {
        weight: None,
        bias: None,
        rms_norm: RMSNorm {
            weight: Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(head_rms, [d_model]), device)),
            eps: 1e-5,
        },
        activation_bits: 8,
        in_features: d_model,
        out_features: vocab_size,
        quantized: true,
    };

    let model = TransformerBitLinearLM {
        embedding,
        transformer: BitLinearTransformerStack { layers, final_norm, num_layers, d_model },
        head,
        vocab_size, d_model, num_layers,
    };

    let state = TransformerInferenceState {
        qkv: qkv_states,
        o_proj: o_proj_states,
        ffn_gate_up: ffn_gate_up_states,
        ffn_down: ffn_down_states,
        head: head_state,
    };

    Ok((model, state))
}
