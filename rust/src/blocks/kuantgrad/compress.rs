// ─── KuantGrad: Gradient Compression ──────────────────────────────
//
// Scheme: 8 gradientes por grupo.
//   - 1 × f32  = intensidad del grupo (max abs)
//   - 8 × 5-bit = cuánto de esa intensidad aplica (0-31)
//   - decoding: value = scale * ((q as f32) / 15.5 - 1.0)
//     (q=0→-scale, q=15.5→0, q=31→+scale)
//
// Por grupo de 8: 4 + 5 = 9 bytes (vs 32 bytes sin comprimir) → 3.56×
//
// Las funciones trabajan con &[f32] y producen/consumen Vec<u8>.

const GROUP: usize = 8;

/// Comprime gradientes: f32[] → u8[] empaquetado.
/// Retorna (datos comprimidos, número de grupos).
pub fn compress(grads: &[f32]) -> (Vec<u8>, usize) {
    let n = grads.len();
    let n_groups = (n + GROUP - 1) / GROUP;
    let mut out = Vec::with_capacity(n_groups * 9);

    for g in 0..n_groups {
        let start = g * GROUP;
        let end = (start + GROUP).min(n);
        let slice = &grads[start..end];

        // Escalar = max absolute value
        let scale = slice.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        out.extend_from_slice(&scale.to_le_bytes());

        if scale == 0.0 {
            // sin bits (todo cero)
            out.extend_from_slice(&[0u8; 5]);
            continue;
        }

        // Cuantizar cada valor a 5 bits (0-31), signed mapping
        let mut bits = [0u8; 5]; // 40 bits
        for (j, &val) in slice.iter().enumerate() {
            // normalized = clamp(val / scale, -1, 1) → [0, 31] unsigned
            let norm = (val / scale).clamp(-1.0, 1.0);
            let q = ((norm + 1.0) * 15.5).round() as i32;
            let q = q.clamp(0, 31) as u32;

            // Escribir 5 bits en el array de 5 bytes (40 bits)
            let bit_pos = j * 5;
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            let mask = (q << bit_off) as u16;
            bits[byte_idx] = (bits[byte_idx] as u16 | mask) as u8;
            if byte_idx + 1 < 5 {
                let carry = (q >> (8 - bit_off)) as u8;
                bits[byte_idx + 1] |= carry;
            }
        }
        out.extend_from_slice(&bits);
    }

    (out, n_groups)
}

/// Descomprime: u8[] empaquetado → Vec<f32>.
pub fn decompress(data: &[u8], n_groups: usize, original_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(original_len);
    let mut idx = 0;

    for g in 0..n_groups {
        // Leer scale (4 bytes, little-endian)
        let scale_bytes: [u8; 4] = data[idx..idx + 4].try_into().unwrap();
        let scale = f32::from_le_bytes(scale_bytes);
        idx += 4;

        let n_in_group = if g == n_groups - 1 {
            let rem = original_len % GROUP;
            if rem == 0 { GROUP } else { rem }
        } else {
            GROUP
        };

        if scale == 0.0 {
            for _ in 0..n_in_group {
                out.push(0.0);
            }
            idx += 5; // skip zero bits
            continue;
        }

        // Leer 5 bytes de bits
        let bits_slice: &[u8; 5] = &data[idx..idx + 5].try_into().unwrap();
        let bits = read_40_bits(bits_slice);
        idx += 5;

        for j in 0..n_in_group {
            let bit_pos = j * 5;
            let q = (bits >> bit_pos) & 0x1F; // 5-bit mask
            let norm = (q as f32) / 15.5 - 1.0; // [q=0→-1, q=15.5→0, q=31→+1]
            out.push(norm * scale);
        }
    }

    out
}

fn read_40_bits(buf: &[u8; 5]) -> u64 {
    let mut val: u64 = 0;
    for i in 0..5 {
        val |= (buf[i] as u64) << (i * 8);
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let grads: Vec<f32> = vec![1.0, -2.0, 0.5, -0.1, 0.0, 3.0, -1.5, 0.3];
        let (compressed, ng) = compress(&grads);
        let decompressed = decompress(&compressed, ng, grads.len());

        assert_eq!(decompressed.len(), grads.len());
        // tolerancia ~ 1/15.5 ≈ 6.5%
        for (a, b) in grads.iter().zip(decompressed.iter()) {
            let err = (a - b).abs() / a.abs().max(1e-8);
            assert!(err < 0.07, "error relativo {} para {} vs {}", err, a, b);
        }
        println!("KuantGrad roundtrip OK: {} bytes → {} bytes ({} grupos)",
            grads.len() * 4, compressed.len(), ng);
    }

    #[test]
    fn compression_ratio() {
        let grads: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let (compressed, ng) = compress(&grads);
        let raw_bytes = grads.len() * 4;
        let ratio = raw_bytes as f64 / compressed.len() as f64;
        println!("Raw: {} bytes, KuantGrad: {} bytes, ratio: {:.2}×", raw_bytes, compressed.len(), ratio);
        assert!(ratio > 3.0, "ratio debería ser ~3.56×, fue {:.2}", ratio);
    }
}
