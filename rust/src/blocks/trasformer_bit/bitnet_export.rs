use std::io::{self, Read, Write, BufWriter, BufReader};
use std::fs::File;
use std::path::Path;

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::blocks::bitlinear::kernel::{I2SKernel, KernelKind};
use crate::blocks::bitlinear::layer::{BitLinear, BitLinearInferenceState};

use super::model::{
    TransformerBitLinearLM, TransformerInferenceState, BitLinearTransformerStack, BitLinearTransformerLayer,
    BitLinearRMSNorm, BitLinearQKVProjection, BitLinearOutputProjection,
    BitLinearSwiGLUFeedForward,
};

const MAGIC: &[u8; 4] = b"BTNT";
const VERSION: u32 = 2;

// ─── f32 ↔ f16 conversion ──────────────────────────────────────────────────

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 1 } else { 0 };
    }
    if exp == 0 {
        return sign | (mant >> 13) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v & 0x8000) as u32) << 16;
    let exp = ((v >> 10) & 0x1F) as i32;
    let mant = (v & 0x3FF) as u32;

    if exp == 0 {
        return f32::from_bits(sign | (mant << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    let new_exp = (exp - 15 + 127) as u32;
    f32::from_bits(sign | (new_exp << 23) | (mant << 13))
}

// ─── Low-level I/O helpers ──────────────────────────────────────────────────

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_f32_slice<W: Write>(w: &mut W, data: &[f32]) -> io::Result<()> {
    for &v in data { write_f32(w, v)?; }
    Ok(())
}
fn write_f16<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    let h = f32_to_f16(v);
    w.write_all(&h.to_le_bytes())
}
fn write_f16_slice<W: Write>(w: &mut W, data: &[f32]) -> io::Result<()> {
    for &v in data { write_f16(w, v)?; }
    Ok(())
}

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
    let mut v = vec![0.0f32; len];
    for val in v.iter_mut() { *val = read_f32(r)?; }
    Ok(v)
}
fn read_f16<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(f16_to_f32(u16::from_le_bytes(buf)))
}
fn read_f16_vec<R: Read>(r: &mut R, len: usize) -> io::Result<Vec<f32>> {
    let mut v = vec![0.0f32; len];
    for val in v.iter_mut() { *val = read_f16(r)?; }
    Ok(v)
}
fn read_u32_vec<R: Read>(r: &mut R, len: usize) -> io::Result<Vec<u32>> {
    let mut v = vec![0u32; len];
    for val in v.iter_mut() { *val = read_u32(r)?; }
    Ok(v)
}

// ─── BitLinear bin struct ───────────────────────────────────────────────────

struct BitLinearBin {
    rms_norm_weight: Vec<f32>,
    packed_w: Vec<u32>,
    scales: Vec<f32>,
    in_features: u32,
    out_features: u32,
}

fn extract_bitlinear<B: Backend>(bl: &BitLinear<B>) -> BitLinearBin {
    let device = bl.rms_norm.weight.val().device();
    let rms_norm_weight = bl.rms_norm.weight.val().into_data().as_slice::<f32>().unwrap().to_vec();

    let (w_ternary, scales_tensor) = bl.get_ternary_weights(&device);
    let dims = w_ternary.dims();
    let scales = scales_tensor.into_data().as_slice::<f32>().unwrap().to_vec();
    let w_data = w_ternary.into_data();
    let w_slice = w_data.as_slice::<f32>().unwrap();
    let packed_w = I2SKernel::pack_weights(w_slice);

    BitLinearBin { rms_norm_weight, packed_w, scales, in_features: dims[1] as u32, out_features: dims[0] as u32 }
}

fn write_bitlinear<W: Write>(w: &mut W, bl: &BitLinearBin) -> io::Result<()> {
    write_u32(w, bl.in_features)?;
    write_u32(w, bl.out_features)?;
    write_u32(w, bl.rms_norm_weight.len() as u32)?;
    write_f32_slice(w, &bl.rms_norm_weight)?;
    write_u32(w, bl.packed_w.len() as u32)?;
    for &p in &bl.packed_w { write_u32(w, p)?; }
    write_u32(w, bl.scales.len() as u32)?;
    write_f32_slice(w, &bl.scales)?;
    Ok(())
}

struct BitLinearBinLoaded {
    rms_norm_weight: Vec<f32>,
    packed_w: Vec<u32>,
    scales: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

fn read_bitlinear<R: Read>(r: &mut R) -> io::Result<BitLinearBinLoaded> {
    let in_features = read_u32(r)? as usize;
    let out_features = read_u32(r)? as usize;
    let rms_len = read_u32(r)? as usize;
    let rms_norm_weight = read_f32_vec(r, rms_len)?;
    let packed_len = read_u32(r)? as usize;
    let packed_w = read_u32_vec(r, packed_len)?;
    let scales_len = read_u32(r)? as usize;
    let scales = read_f32_vec(r, scales_len)?;
    Ok(BitLinearBinLoaded { rms_norm_weight, packed_w, scales, in_features, out_features })
}

fn write_norm_generic<W: Write, B: Backend>(w: &mut W, norm: &BitLinearRMSNorm<B>) -> io::Result<()> {
    let data = norm.weight.val().into_data();
    let slice = data.as_slice::<f32>().unwrap();
    write_u32(w, slice.len() as u32)?;
    write_f16_slice(w, slice)
}

// ─── Export ─────────────────────────────────────────────────────────────────

pub fn export_bitnet<B: Backend>(model: &TransformerBitLinearLM<B>, path: &str) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;

    write_u32(&mut w, model.vocab_size as u32)?;
    write_u32(&mut w, model.d_model as u32)?;
    write_u32(&mut w, model.num_layers as u32)?;
    write_u32(&mut w, model.transformer.layers[0].qkv.num_heads as u32)?;
    write_u32(&mut w, model.transformer.layers[0].qkv.num_kv_groups as u32)?;
    write_u32(&mut w, model.transformer.layers[0].qkv.head_dim as u32)?;
    write_u32(&mut w, model.transformer.layers[0].ffn.intermediate_dim as u32)?;

    // Embedding (f16)
    let embed_data = model.embedding.weight.val().into_data();
    let embed_slice = embed_data.as_slice::<f32>().unwrap();
    write_u32(&mut w, embed_slice.len() as u32)?;
    write_f16_slice(&mut w, embed_slice)?;

    for layer in &model.transformer.layers {
        write_norm_generic(&mut w, &layer.attn_norm)?;
        write_norm_generic(&mut w, &layer.ffn_norm)?;

        write_bitlinear(&mut w, &extract_bitlinear(&layer.qkv.q_proj))?;
        write_bitlinear(&mut w, &extract_bitlinear(&layer.qkv.k_proj))?;
        write_bitlinear(&mut w, &extract_bitlinear(&layer.qkv.v_proj))?;
        write_bitlinear(&mut w, &extract_bitlinear(&layer.o_proj.o_proj))?;
        write_bitlinear(&mut w, &extract_bitlinear(&layer.ffn.gate_up_proj))?;
        write_bitlinear(&mut w, &extract_bitlinear(&layer.ffn.down_proj))?;
    }

    write_norm_generic(&mut w, &model.transformer.final_norm)?;
    write_bitlinear(&mut w, &extract_bitlinear(&model.head))?;

    w.flush()?;

    let file_size = std::fs::metadata(path)?.len();
    let mpk_size = std::fs::metadata("transformer_bit2.mpk").map(|m| m.len()).unwrap_or(0);

    // Detailed breakdown
    let vs = model.vocab_size;
    let dm = model.d_model;
    let nl = model.num_layers;
    let nh = model.transformer.layers[0].qkv.num_heads;
    let nkv = model.transformer.layers[0].qkv.num_kv_groups;
    let hd = model.transformer.layers[0].qkv.head_dim;
    let fd = model.transformer.layers[0].ffn.intermediate_dim;

    let embed_bytes = vs * dm * 2;  // f16
    let norm_bytes = (2 * nl + 1) * dm * 2;  // f16

    // BitLinear sizes: ternary packed = ceil(in*out/16)*4 bytes, scales = ceil(in*out/128)*4 bytes
    let bl_size = |inf: usize, outf: usize| -> usize {
        let params = inf * outf;
        let packed = ((params + 15) / 16) * 4;
        let groups = (params + 127) / 128;
        let scales = groups * 4;
        let rms = inf * 4;  // f32 rms norm weight (write_f32_slice)
        packed + scales + rms
    };

    let q_size = bl_size(dm, nh * hd);
    let k_size = bl_size(dm, nkv * hd);
    let v_size = bl_size(dm, nkv * hd);
    let o_size = bl_size(nh * hd, dm);
    let gu_size = bl_size(dm, 2 * fd);
    let d_size = bl_size(fd, dm);
    let head_size = bl_size(dm, vs);
    let per_layer = q_size + k_size + v_size + o_size + gu_size + d_size;
    let all_layers = per_layer * nl;

    let bitnet_total = embed_bytes + norm_bytes + all_layers + head_size;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Exportación BitNet completada                          ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Formato:     .bitnet (ternary packed)                  ║");
    println!("║  Tamaño total:{:.2} MB ({} bytes)               ", file_size as f64 / 1e6, file_size);
    if mpk_size > 0 {
        println!("║  MPK original:{:.2} MB ({} bytes)               ", mpk_size as f64 / 1e6, mpk_size);
        println!("║  Reducción:   {:.2}x                            ", mpk_size as f64 / file_size as f64);
    }
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  DESGLOSE POR COMPONENTE (.bitnet):                     ║");
    println!("║                                                         ║");
    println!("║  [NO BitLinear - f16]                                  ║");
    println!("║    Embedding {:>5}×{:<5}: {:>8.2} MB  ({:>10} bytes)  ", vs, dm, embed_bytes as f64 / 1e6, embed_bytes);
    println!("║    Norms {}×(attn+ffn)+final: {:>6.2} MB  ({:>10} bytes)  ", nl, norm_bytes as f64 / 1e6, norm_bytes);
    println!("║                                                         ║");
    println!("║  [BitLinear - ternary packed]                            ║");
    println!("║    Per-layer:                                           ║");
    println!("║      q_proj    {:>5}×{:<5}: {:>8.2} KB              ", dm, nh*hd, q_size as f64 / 1e3);
    println!("║      k_proj    {:>5}×{:<5}: {:>8.2} KB              ", dm, nkv*hd, k_size as f64 / 1e3);
    println!("║      v_proj    {:>5}×{:<5}: {:>8.2} KB              ", dm, nkv*hd, v_size as f64 / 1e3);
    println!("║      o_proj    {:>5}×{:<5}: {:>8.2} KB              ", nh*hd, dm, o_size as f64 / 1e3);
    println!("║      gate_up   {:>5}×{:<5}: {:>8.2} KB              ", dm, 2*fd, gu_size as f64 / 1e3);
    println!("║      down      {:>5}×{:<5}: {:>8.2} KB              ", fd, dm, d_size as f64 / 1e3);
    println!("║      subtotal ×{} layers: {:>8.2} MB              ", nl, all_layers as f64 / 1e6);
    println!("║    head        {:>5}×{:<5}: {:>8.2} MB              ", dm, vs, head_size as f64 / 1e6);
    println!("║                                                         ║");
    println!("║  ─────────────────────────────────────────────────────  ║");
    println!("║  Embedding (f16):      {:>8.2} MB  ({:>5.1}% del total)  ", embed_bytes as f64 / 1e6, embed_bytes as f64 / file_size as f64 * 100.0);
    println!("║  Norms (f16):          {:>8.2} MB  ({:>5.1}% del total)  ", norm_bytes as f64 / 1e6, norm_bytes as f64 / file_size as f64 * 100.0);
    println!("║  BitLinear layers:     {:>8.2} MB  ({:>5.1}% del total)  ", all_layers as f64 / 1e6, all_layers as f64 / file_size as f64 * 100.0);
    println!("║  Head (BitLinear):     {:>8.2} MB  ({:>5.1}% del total)  ", head_size as f64 / 1e6, head_size as f64 / file_size as f64 * 100.0);
    println!("║  ─────────────────────────────────────────────────────  ║");
    println!("║  TOTAL calculado:      {:>8.2} MB                      ", bitnet_total as f64 / 1e6);
    println!("║  TOTAL archivo:        {:>8.2} MB                      ", file_size as f64 / 1e6);
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}

// ─── Load (reconstruct BitLinear from ternary) ─────────────────────────────

fn reconstruct_bitlinear<B: Backend>(bl: &BitLinearBinLoaded, device: &B::Device) -> BitLinear<B> {
    use crate::blocks::bitlinear::layer::RMSNorm;

    let rms_norm = RMSNorm {
        weight: burn::module::Param::from_tensor(
            Tensor::<B, 1>::from_data(TensorData::new(bl.rms_norm_weight.clone(), [bl.in_features]), device)
        ),
        eps: 1e-5,
    };

    let numel = bl.in_features * bl.out_features;
    let mut w_f32 = vec![0.0f32; numel];

    let u32s_per_row = (bl.in_features + 15) / 16;
    let groups_per_row = (bl.in_features + 127) / 128;

    for (g_idx, &packed) in bl.packed_w.iter().enumerate() {
        let base = g_idx * 16;
        let row = g_idx / u32s_per_row;
        let pos_in_row = g_idx % u32s_per_row;
        let gi = pos_in_row / 8;
        let scale_idx = row * groups_per_row + gi;
        let scale = bl.scales.get(scale_idx).copied().unwrap_or(1e-8);
        for bit_idx in 0..16 {
            let w_idx = base + bit_idx;
            if w_idx >= numel { break; }
            let bits = (packed >> (bit_idx * 2)) & 0b11;
            let ternary = (bits as i32) - 1;
            w_f32[w_idx] = ternary as f32 * scale;
        }
    }

    let weight_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(w_f32, [bl.out_features, bl.in_features]), device
    );

    BitLinear {
        weight: Some(burn::module::Param::from_tensor(weight_tensor)),
        bias: None,
        rms_norm,
        activation_bits: 8,
        in_features: bl.in_features,
        out_features: bl.out_features,
        quantized: true,
    }
}

pub fn load_bitnet<B: Backend>(path: &str, device: &B::Device) -> io::Result<(TransformerBitLinearLM<B>, Vec<String>)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut warnings = Vec::new();

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC { return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BitNet file")); }

    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unsupported version: {}", version)));
    }

    let vocab_size = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let num_heads = read_u32(&mut r)? as usize;
    let num_kv_groups = read_u32(&mut r)? as usize;
    let head_dim = read_u32(&mut r)? as usize;
    let ffn_dim = read_u32(&mut r)? as usize;

    // Embedding (f16)
    let embed_len = read_u32(&mut r)? as usize;
    let embed_data = read_f16_vec(&mut r, embed_len)?;
    let embedding_weight = Tensor::<B, 2>::from_data(
        TensorData::new(embed_data, [vocab_size, d_model]), device
    );

    let mut layers = Vec::new();
    for _ in 0..num_layers {
        let _ = read_u32(&mut r)?;
        let attn_norm_w = read_f16_vec(&mut r, d_model)?;
        let _ = read_u32(&mut r)?;
        let ffn_norm_w = read_f16_vec(&mut r, d_model)?;

        let attn_norm = BitLinearRMSNorm {
            weight: burn::module::Param::from_tensor(
                Tensor::<B, 1>::from_data(TensorData::new(attn_norm_w, [d_model]), device)
            ),
            eps: 1e-5,
        };
        let ffn_norm = BitLinearRMSNorm {
            weight: burn::module::Param::from_tensor(
                Tensor::<B, 1>::from_data(TensorData::new(ffn_norm_w, [d_model]), device)
            ),
            eps: 1e-5,
        };

        let q_bin = read_bitlinear(&mut r)?;
        let k_bin = read_bitlinear(&mut r)?;
        let v_bin = read_bitlinear(&mut r)?;
        let o_bin = read_bitlinear(&mut r)?;
        let gu_bin = read_bitlinear(&mut r)?;
        let d_bin = read_bitlinear(&mut r)?;

        let dropout = burn::nn::DropoutConfig::new(0.0).init();

        layers.push(BitLinearTransformerLayer {
            attn_norm,
            qkv: BitLinearQKVProjection {
                q_proj: reconstruct_bitlinear(&q_bin, device),
                k_proj: reconstruct_bitlinear(&k_bin, device),
                v_proj: reconstruct_bitlinear(&v_bin, device),
                num_heads, num_kv_groups, head_dim,
            },
            o_proj: BitLinearOutputProjection {
                o_proj: reconstruct_bitlinear(&o_bin, device),
                num_heads, head_dim,
            },
            ffn_norm,
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: reconstruct_bitlinear(&gu_bin, device),
                down_proj: reconstruct_bitlinear(&d_bin, device),
                dropout,
                intermediate_dim: ffn_dim,
            },
            residual_dropout: burn::nn::DropoutConfig::new(0.0).init(),
        });
    }

    let _ = read_u32(&mut r)?;
    let final_norm_w = read_f16_vec(&mut r, d_model)?;
    let final_norm = BitLinearRMSNorm {
        weight: burn::module::Param::from_tensor(
            Tensor::<B, 1>::from_data(TensorData::new(final_norm_w, [d_model]), device)
        ),
        eps: 1e-5,
    };

    let head_bin = read_bitlinear(&mut r)?;
    let head = reconstruct_bitlinear(&head_bin, device);

    let model = TransformerBitLinearLM {
        embedding: burn::nn::Embedding { weight: burn::module::Param::from_tensor(embedding_weight) },
        transformer: BitLinearTransformerStack { layers, final_norm, num_layers, d_model },
        head,
        vocab_size, d_model, num_layers,
    };

    warnings.push(format!("BitNet ternary load: {} layers, d_model={}, vocab={}", num_layers, d_model, vocab_size));
    warnings.push("Pesos BitLinear reconstruidos desde ternario (sin shadow weights). Solo inferencia.".into());

    Ok((model, warnings))
}

// ─── Direct Inference State Load (bypasses f32 round-trip) ─────────────────

pub fn load_bitnet_inference_state(
    path: &str,
    d_model: usize,
    num_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    layer_kernel: KernelKind,
    head_kernel: KernelKind,
) -> io::Result<TransformerInferenceState> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC { return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BitNet file")); }
    let version = read_u32(&mut r)?;
    if version != VERSION { return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unsupported version: {}", version))); }

    let _vocab = read_u32(&mut r)? as usize;
    let _dmodel = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let _nh = read_u32(&mut r)? as usize;
    let _nkv = read_u32(&mut r)? as usize;
    let _hd = read_u32(&mut r)? as usize;
    let _fd = read_u32(&mut r)? as usize;

    // Skip embedding (f16)
    let embed_len = read_u32(&mut r)? as usize;
    skip_bytes(&mut r, embed_len * 2)?;

    let mut qkv_states = Vec::new();
    let mut o_proj_states = Vec::new();
    let mut ffn_gate_up_states = Vec::new();
    let mut ffn_down_states = Vec::new();

    for _ in 0..num_layers {
        skip_norm(&mut r)?;
        skip_norm(&mut r)?;

        let q_bin = read_bitlinear(&mut r)?;
        let k_bin = read_bitlinear(&mut r)?;
        let v_bin = read_bitlinear(&mut r)?;
        let o_bin = read_bitlinear(&mut r)?;
        let gu_bin = read_bitlinear(&mut r)?;
        let d_bin = read_bitlinear(&mut r)?;

        qkv_states.push((
            BitLinearInferenceState::from_packed(q_bin.packed_w, q_bin.scales, d_model, num_heads * head_dim, layer_kernel),
            BitLinearInferenceState::from_packed(k_bin.packed_w, k_bin.scales, d_model, num_kv_groups * head_dim, layer_kernel),
            BitLinearInferenceState::from_packed(v_bin.packed_w, v_bin.scales, d_model, num_kv_groups * head_dim, layer_kernel),
        ));
        o_proj_states.push(BitLinearInferenceState::from_packed(o_bin.packed_w, o_bin.scales, num_heads * head_dim, d_model, layer_kernel));
        ffn_gate_up_states.push(BitLinearInferenceState::from_packed(gu_bin.packed_w, gu_bin.scales, d_model, 2 * ffn_dim, layer_kernel));
        ffn_down_states.push(BitLinearInferenceState::from_packed(d_bin.packed_w, d_bin.scales, ffn_dim, d_model, layer_kernel));
    }

    skip_norm(&mut r)?;
    let head_bin = read_bitlinear(&mut r)?;

    Ok(TransformerInferenceState {
        qkv: qkv_states,
        o_proj: o_proj_states,
        ffn_gate_up: ffn_gate_up_states,
        ffn_down: ffn_down_states,
        head: BitLinearInferenceState::from_packed(head_bin.packed_w, head_bin.scales, d_model, vocab_size, head_kernel),
    })
}

fn skip_norm<R: Read>(r: &mut R) -> io::Result<()> {
    let rms_len = read_u32(r)? as usize;
    skip_bytes(r, rms_len * 2)?;  // f16
    Ok(())
}

fn skip_bytes<R: Read>(r: &mut R, n: usize) -> io::Result<()> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(())
}

// ─── Memory Report ──────────────────────────────────────────────────────────
fn bitlinear_bytes(bl: &BitLinearInferenceState) -> usize {
    bl.packed_w.len() * std::mem::size_of::<u32>() + bl.scales.len() * std::mem::size_of::<f32>()
}

pub fn report_inference_state_memory(
    state: &TransformerInferenceState,
    embed_bytes: usize,
    norm_bytes: usize,
    d_model: usize,
    num_layers: usize,
) {
    let mut total_packed = 0usize;
    let mut total_scales = 0usize;

    let mut component = |name: &str, bl: &BitLinearInferenceState| {
        let pw = bl.packed_w.len() * std::mem::size_of::<u32>();
        let sc = bl.scales.len() * std::mem::size_of::<f32>();
        println!("    {:<14} packed: {:>8.2} KB  scales: {:>6.2} KB  ({}×{})",
            name, pw as f64 / 1e3, sc as f64 / 1e3, bl.in_features, bl.out_features);
        total_packed += pw;
        total_scales += sc;
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Desglose de RAM — Inference State                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for (li, (q, k, v)) in state.qkv.iter().enumerate() {
        println!("  Capa {}:", li);
        component("q_proj", q);
        component("k_proj", k);
        component("v_proj", v);
        component("o_proj", &state.o_proj[li]);
        component("gate_up", &state.ffn_gate_up[li]);
        component("down", &state.ffn_down[li]);
    }
    println!("  Head:");
    component("head", &state.head);

    let total_bitlinear = total_packed + total_scales;
    let total = embed_bytes + norm_bytes + total_bitlinear;

    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  RESUMEN                                                    ║");
    println!("║    Embedding (f16):      {:>8.2} KB  ({:>5.1}%)            ", embed_bytes as f64 / 1e3, embed_bytes as f64 / total as f64 * 100.0);
    println!("║    Norms (f16):          {:>8.2} KB  ({:>5.1}%)            ", norm_bytes as f64 / 1e3, norm_bytes as f64 / total as f64 * 100.0);
    println!("║    Packed ternary:       {:>8.2} KB  ({:>5.1}%)            ", total_packed as f64 / 1e3, total_packed as f64 / total as f64 * 100.0);
    println!("║    Scales (f32):         {:>8.2} KB  ({:>5.1}%)            ", total_scales as f64 / 1e3, total_scales as f64 / total as f64 * 100.0);
    println!("║    ─────────────────────────────────────────────────────    ║");
    println!("║    TOTAL inference:      {:>8.2} KB  ({} layers)           ", total as f64 / 1e3, num_layers);

    let act_per_token = d_model * (4 + 1);  // f32 input + i8 quantized
    println!("║    Activación/token:     {:>8.2} KB  (d_model={} i8+f32)  ", act_per_token as f64 / 1e3, d_model);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ─── Auto-detect format ────────────────────────────────────────────────────

pub fn is_bitnet_file(path: &str) -> bool {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == MAGIC
}

pub fn load_model_auto<B: Backend>(
    bitnet_path: &str,
    device: &B::Device,
) -> io::Result<(TransformerBitLinearLM<B>, Vec<String>)> {
    if Path::new(bitnet_path).exists() && is_bitnet_file(bitnet_path) {
        load_bitnet(bitnet_path, device)
    } else {
        Err(io::Error::new(io::ErrorKind::NotFound, format!("No BitNet file found: {}", bitnet_path)))
    }
}

pub fn compare_models<B: Backend>(
    mpk_path: &str,
    bitnet_path: &str,
    device: &B::Device,
) -> io::Result<()> {
    let mpk_size = std::fs::metadata(mpk_path).map(|m| m.len()).unwrap_or(0);
    let bitnet_size = std::fs::metadata(bitnet_path).map(|m| m.len()).unwrap_or(0);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Comparación de formatos                     ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  MPK:     {:.2} MB                           ", mpk_size as f64 / 1e6);
    println!("║  BitNet:  {:.2} MB                           ", bitnet_size as f64 / 1e6);
    if bitnet_size > 0 {
        println!("║  Ratio:   {:.2}x más compacto              ", mpk_size as f64 / bitnet_size as f64);
    }
    println!("╚══════════════════════════════════════════════╝");

    if Path::new(bitnet_path).exists() {
        println!("\nCargando modelo BitNet para verificación...");
        let (bn_model, bn_warn) = load_bitnet::<B>(bitnet_path, device)?;
        for w in &bn_warn { println!("  {}", w); }

        let bn_params: usize = bn_model.transformer.layers.iter().map(|l| {
            l.qkv.q_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
            + l.qkv.k_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
            + l.qkv.v_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
            + l.o_proj.o_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
            + l.ffn.gate_up_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
            + l.ffn.down_proj.weight.as_ref().map_or(0, |w| w.val().dims().iter().product::<usize>())
        }).sum::<usize>();

        println!("  BitNet BitLinear params (f32 dequant): {:.2}M", bn_params as f64 / 1e6);
        println!("  Embedding: {}x{} = {:.2}M", bn_model.vocab_size, bn_model.d_model, (bn_model.vocab_size * bn_model.d_model) as f64 / 1e6);
    }

    Ok(())
}

// ─── Compatibility Test: MPK vs .bitnet ──────────────────────────────────────

pub fn compare_compatibility<B: Backend>(
    mpk_path: &str,
    bitnet_path: &str,
    device: &B::Device,
) -> io::Result<()> {
    use burn::record::{CompactRecorder, Recorder};
    use burn::module::Module;
    use burn::tensor::Int;
    use crate::blocks::bitlinear::layer::BitLinearConfig;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Comparación de compatibilidad MPK vs .bitnet                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let d_model: usize = 512;
    let num_layers: usize = 6;
    let num_heads: usize = 8;
    let num_kv_groups: usize = 4;
    let head_dim = d_model / num_heads;
    let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;
    let vocab_size = 16000;

    // ── Load MPK model ──────────────────────────────────────────
    let mpk_layers: Vec<BitLinearTransformerLayer<B>> = (0..num_layers).map(|_| {
        BitLinearTransformerLayer {
            attn_norm: BitLinearRMSNorm::new(d_model, 1e-5, device),
            qkv: BitLinearQKVProjection {
                q_proj: BitLinearConfig { in_features: d_model, out_features: num_heads * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                k_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                v_proj: BitLinearConfig { in_features: d_model, out_features: num_kv_groups * head_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                num_heads, num_kv_groups, head_dim,
            },
            o_proj: BitLinearOutputProjection {
                o_proj: BitLinearConfig { in_features: num_heads * head_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                num_heads, head_dim,
            },
            ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, device),
            ffn: BitLinearSwiGLUFeedForward {
                gate_up_proj: BitLinearConfig { in_features: d_model, out_features: 2 * ffn_dim, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                down_proj: BitLinearConfig { in_features: ffn_dim, out_features: d_model, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
                dropout: burn::nn::DropoutConfig::new(0.0).init(),
                intermediate_dim: ffn_dim,
            },
            residual_dropout: burn::nn::DropoutConfig::new(0.0).init(),
        }
    }).collect();

    let mut mpk_model: TransformerBitLinearLM<B> = TransformerBitLinearLM {
        embedding: burn::nn::EmbeddingConfig::new(vocab_size, d_model).init(device),
        transformer: BitLinearTransformerStack { final_norm: BitLinearRMSNorm::new(d_model, 1e-5, device), num_layers, d_model, layers: mpk_layers },
        head: BitLinearConfig { in_features: d_model, out_features: vocab_size, bias: false, activation_bits: 8, rms_norm_eps: 1e-5, quantized: true }.init(device),
        vocab_size, d_model, num_layers,
    };

    println!("  Cargando MPK: {}...", mpk_path);
    let record = CompactRecorder::new().load(mpk_path.into(), device)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    mpk_model = mpk_model.load_record(record);
    println!("  MPK cargado.");

    // ── Load .bitnet model ──────────────────────────────────────
    println!("  Cargando .bitnet: {}...", bitnet_path);
    let (bn_model, warnings) = load_bitnet::<B>(bitnet_path, device)?;
    for w in &warnings { println!("    {}", w); }
    println!("  .bitnet cargado.");

    // ── Same input for both ─────────────────────────────────────
    let seq_len = 32;
    let token_ids: Vec<i64> = (0..seq_len).map(|i| (i % vocab_size) as i64).collect();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(token_ids, [1, seq_len]), device
    );

    // ── Forward pass on both ────────────────────────────────────
    println!("  Ejecutando forward pass (seq_len={})...", seq_len);

    let mpk_logits = mpk_model.forward(input.clone());
    let bn_logits = bn_model.forward(input);

    let mpk_data = mpk_logits.into_data();
    let bn_data = bn_logits.into_data();

    let mpk_slice = mpk_data.as_slice::<f32>().unwrap();
    let bn_slice = bn_data.as_slice::<f32>().unwrap();

    assert_eq!(mpk_slice.len(), bn_slice.len(), "Dimensiones de salida diferentes");

    // ── Numerical comparison ────────────────────────────────────
    let n = mpk_slice.len() as f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_mpk = 0.0f64;
    let mut sum_sq_bn = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut max_abs_diff_pos = 0usize;
    let mut cosine_num = 0.0f64;
    let mut match_count = 0usize;

    for i in 0..mpk_slice.len() {
        let a = mpk_slice[i] as f64;
        let b = bn_slice[i] as f64;
        let diff = a - b;
        sum_sq_diff += diff * diff;
        sum_sq_mpk += a * a;
        sum_sq_bn += b * b;
        sum_abs_diff += diff.abs();
        cosine_num += a * b;
        if diff.abs() > max_abs_diff {
            max_abs_diff = diff.abs();
            max_abs_diff_pos = i;
        }
        // Top-1 match: same argmax in vocab dimension
        // We'll check per-position below
    }

    let mse = sum_sq_diff / n;
    let rmse = mse.sqrt();
    let mae = sum_abs_diff / n;
    let cosine_sim = cosine_num / (sum_sq_mpk.sqrt() * sum_sq_bn.sqrt() + 1e-12);

    // Per-position top-1 accuracy
    let mut top1_matches = 0usize;
    for pos in 0..seq_len {
        let mpk_row = &mpk_slice[pos * vocab_size..(pos + 1) * vocab_size];
        let bn_row = &bn_slice[pos * vocab_size..(pos + 1) * vocab_size];
        let mpk_argmax = mpk_row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let bn_argmax = bn_row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if mpk_argmax == bn_argmax { top1_matches += 1; }
    }

    // Top-5 match
    let mut top5_matches = 0usize;
    for pos in 0..seq_len {
        let mpk_row = &mpk_slice[pos * vocab_size..(pos + 1) * vocab_size];
        let bn_row = &bn_slice[pos * vocab_size..(pos + 1) * vocab_size];
        let mut mpk_top5: Vec<(usize, f32)> = mpk_row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        mpk_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut bn_top5: Vec<(usize, f32)> = bn_row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        bn_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mpk_set: std::collections::HashSet<usize> = mpk_top5.iter().take(5).map(|(i, _)| *i).collect();
        let bn_set: std::collections::HashSet<usize> = bn_top5.iter().take(5).map(|(i, _)| *i).collect();
        if mpk_set.intersection(&bn_set).count() > 0 { top5_matches += 1; }
    }

    println!("║                                                              ║");
    println!("║  Resultados numéricos:                                       ║");
    println!("║    MSE:              {:>14.8}                           ", mse);
    println!("║    RMSE:             {:>14.8}                           ", rmse);
    println!("║    MAE:              {:>14.8}                           ", mae);
    println!("║    Cosine similarity:{:>14.8}                           ", cosine_sim);
    println!("║    Max abs diff:     {:>14.8}  (pos={})              ", max_abs_diff, max_abs_diff_pos);
    println!("║                                                              ║");
    println!("║  Per-position accuracy ({} tokens):                        ║", seq_len);
    println!("║    Top-1 match:      {:>5}/{} ({:>5.1}%)                    ", top1_matches, seq_len, top1_matches as f64 / seq_len as f64 * 100.0);
    println!("║    Top-5 overlap:    {:>5}/{} ({:>5.1}%)                    ", top5_matches, seq_len, top5_matches as f64 / seq_len as f64 * 100.0);

    // Quality verdict
    let verdict = if cosine_sim > 0.99 {
        "EXCELENTE — paridad casi perfecta"
    } else if cosine_sim > 0.95 {
        "BUENO — menor error de cuantización"
    } else if cosine_sim > 0.80 {
        "REGULAR — pérdida significativa"
    } else if cosine_sim > 0.50 {
        "MALO — divergencia notable"
    } else {
        "CRÍTICO — modelos fundamentalmente diferentes"
    };

    println!("║                                                              ║");
    println!("║  Veredicto: {:<48}║", verdict);
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
