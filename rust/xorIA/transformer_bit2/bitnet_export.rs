use std::io::{self, Read, Write, BufWriter, BufReader};
use std::fs::File;
use std::path::Path;

use burn::tensor::{Tensor, TensorData, backend::Backend};

use xlstm::blocks::bitlinear::kernel::{I2SKernel, KernelKind};
use xlstm::blocks::bitlinear::layer::{BitLinear, BitLinearInferenceState};

use super::{
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
        let rms = inf * 2;  // f16 rms norm weight
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
    use xlstm::blocks::bitlinear::layer::RMSNorm;

    let rms_norm = RMSNorm {
        weight: burn::module::Param::from_tensor(
            Tensor::<B, 1>::from_data(TensorData::new(bl.rms_norm_weight.clone(), [bl.in_features]), device)
        ),
        eps: 1e-5,
    };

    let numel = bl.in_features * bl.out_features;
    let mut w_f32 = vec![0.0f32; numel];

    for (g_idx, &packed) in bl.packed_w.iter().enumerate() {
        let base = g_idx * 16;
        let scale = bl.scales.get(g_idx).copied().unwrap_or(1e-8);
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
