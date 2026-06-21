//! TurboQuant KV-cache compression ported from `bitnet.c`.
//!
//! The implementation follows the C reference in `bitnet.c/src/turboquant.c`:
//! random Hadamard rotation, Lloyd-Max scalar quantization, packed 2/3/4-bit
//! indices, and a QJL sign residual correction for keys.

use half::f16;

const MAX_ELEMS: usize = 8192;
const SQRT_PI_OVER_2: f32 = 1.253_314_1;

const LLOYD_MAX_2BIT: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const LLOYD_MAX_2BIT_BOUNDS: [f32; 3] = [-0.9816, 0.0, 0.9816];

const LLOYD_MAX_3BIT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
const LLOYD_MAX_3BIT_BOUNDS: [f32; 7] =
    [-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480];

const LLOYD_MAX_4BIT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3880, -0.1284,
    0.1284, 0.3880, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
const LLOYD_MAX_4BIT_BOUNDS: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582,
    0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurboQuantError {
    InvalidBits(usize),
    InvalidHeadDim(usize),
    VectorLen { expected: usize, actual: usize },
    PackedLen { expected_at_least: usize, actual: usize },
}

impl std::fmt::Display for TurboQuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBits(bits) => write!(f, "invalid TurboQuant bit width: {bits}"),
            Self::InvalidHeadDim(dim) => write!(f, "invalid TurboQuant head_dim: {dim}"),
            Self::VectorLen { expected, actual } => {
                write!(f, "invalid vector length: expected {expected}, got {actual}")
            }
            Self::PackedLen {
                expected_at_least,
                actual,
            } => write!(
                f,
                "invalid packed buffer length: expected at least {expected_at_least}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for TurboQuantError {}

#[derive(Debug, Clone)]
pub struct TurboQuant {
    pub head_dim: usize,
    pub bits: usize,
    pub n_centroids: usize,
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub rht_signs: Vec<f32>,
    pub qjl_signs: Vec<f32>,
    pub rht_scale: f32,
}

impl TurboQuant {
    pub fn new(head_dim: usize, bits: usize, seed: u64) -> Result<Self, TurboQuantError> {
        if !(2..=4).contains(&bits) {
            return Err(TurboQuantError::InvalidBits(bits));
        }
        if head_dim < 8 || head_dim > MAX_ELEMS || !head_dim.is_power_of_two() {
            return Err(TurboQuantError::InvalidHeadDim(head_dim));
        }

        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let (src_c, src_b): (&[f32], &[f32]) = match bits {
            2 => (&LLOYD_MAX_2BIT, &LLOYD_MAX_2BIT_BOUNDS),
            3 => (&LLOYD_MAX_3BIT, &LLOYD_MAX_3BIT_BOUNDS),
            _ => (&LLOYD_MAX_4BIT, &LLOYD_MAX_4BIT_BOUNDS),
        };

        let mut rng = Xoshiro256StarStar::seed(seed);
        let rht_signs = (0..head_dim)
            .map(|_| if rng.next() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let qjl_signs = (0..head_dim)
            .map(|_| if rng.next() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();

        Ok(Self {
            head_dim,
            bits,
            n_centroids: 1 << bits,
            centroids: src_c.iter().map(|v| v * inv_sqrt_d).collect(),
            boundaries: src_b.iter().map(|v| v * inv_sqrt_d).collect(),
            rht_signs,
            qjl_signs,
            rht_scale: inv_sqrt_d,
        })
    }

    pub fn key_bytes(&self) -> usize {
        self.index_bytes() + self.head_dim / 8 + 4
    }

    pub fn value_bytes(&self) -> usize {
        self.index_bytes() + 2
    }

    pub fn rotate_query(&self, query: &[f32]) -> Result<Vec<f32>, TurboQuantError> {
        self.check_vec(query)?;
        Ok(self.rht_forward(query))
    }

    pub fn quantize_key(&self, key: &[f32]) -> Result<Vec<u8>, TurboQuantError> {
        self.check_vec(key)?;
        let mut out = vec![0u8; self.key_bytes()];
        self.quantize_key_into(key, &mut out)?;
        Ok(out)
    }

    pub fn quantize_key_into(&self, key: &[f32], out: &mut [u8]) -> Result<(), TurboQuantError> {
        self.check_vec(key)?;
        self.check_packed(out, self.key_bytes())?;

        let idx_sz = self.index_bytes();
        let qjl_sz = self.head_dim / 8;
        let vec_norm = l2_norm(key);
        let inv_norm = if vec_norm > 1e-10 { 1.0 / vec_norm } else { 0.0 };
        let normalized: Vec<f32> = key.iter().map(|v| v * inv_norm).collect();
        let rotated = self.rht_forward(&normalized);

        let indices: Vec<usize> = rotated.iter().map(|&v| self.quantize_scalar(v)).collect();
        let mut residual = vec![0.0; self.head_dim];
        for i in 0..self.head_dim {
            residual[i] = rotated[i] - self.centroids[indices[i]];
        }
        let res_norm = l2_norm(&residual);
        let qjl = self.qjl_project_signs(&residual);

        pack_indices(&indices, self.bits, &mut out[..idx_sz]);
        out[idx_sz..idx_sz + qjl_sz].copy_from_slice(&qjl);
        out[idx_sz + qjl_sz..idx_sz + qjl_sz + 2]
            .copy_from_slice(&f16::from_f32(res_norm).to_le_bytes());
        out[idx_sz + qjl_sz + 2..idx_sz + qjl_sz + 4]
            .copy_from_slice(&f16::from_f32(vec_norm).to_le_bytes());
        Ok(())
    }

    pub fn quantize_value(&self, value: &[f32]) -> Result<Vec<u8>, TurboQuantError> {
        self.check_vec(value)?;
        let mut out = vec![0u8; self.value_bytes()];
        self.quantize_value_into(value, &mut out)?;
        Ok(out)
    }

    pub fn quantize_value_into(&self, value: &[f32], out: &mut [u8]) -> Result<(), TurboQuantError> {
        self.check_vec(value)?;
        self.check_packed(out, self.value_bytes())?;

        let idx_sz = self.index_bytes();
        let vec_norm = l2_norm(value);
        let inv_norm = if vec_norm > 1e-10 { 1.0 / vec_norm } else { 0.0 };
        let normalized: Vec<f32> = value.iter().map(|v| v * inv_norm).collect();
        let rotated = self.rht_forward(&normalized);
        let indices: Vec<usize> = rotated.iter().map(|&v| self.quantize_scalar(v)).collect();

        pack_indices(&indices, self.bits, &mut out[..idx_sz]);
        out[idx_sz..idx_sz + 2].copy_from_slice(&f16::from_f32(vec_norm).to_le_bytes());
        Ok(())
    }

    pub fn attention_scores(
        &self,
        rotated_q: &[f32],
        packed_keys: &[u8],
        n_keys: usize,
        key_stride: usize,
    ) -> Result<Vec<f32>, TurboQuantError> {
        self.check_vec(rotated_q)?;
        self.check_packed(packed_keys, n_keys.saturating_mul(key_stride))?;
        let q_signs = self.qjl_precompute(rotated_q)?;
        let mut out = Vec::with_capacity(n_keys);
        for k in 0..n_keys {
            let start = k * key_stride;
            out.push(self.score_key_precomputed(rotated_q, &q_signs, &packed_keys[start..])?);
        }
        Ok(out)
    }

    pub fn attention_combine(
        &self,
        packed_values: &[u8],
        n_values: usize,
        value_stride: usize,
        weights: &[f32],
    ) -> Result<Vec<f32>, TurboQuantError> {
        if weights.len() != n_values {
            return Err(TurboQuantError::VectorLen {
                expected: n_values,
                actual: weights.len(),
            });
        }
        self.check_packed(packed_values, n_values.saturating_mul(value_stride))?;

        let idx_sz = self.index_bytes();
        let mut out = vec![0.0; self.head_dim];
        for (k, &weight) in weights.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let start = k * value_stride;
            let packed = &packed_values[start..start + value_stride];
            let indices = unpack_indices(&packed[..idx_sz], self.head_dim, self.bits);
            let vec_norm = f16::from_le_bytes([packed[idx_sz], packed[idx_sz + 1]]).to_f32();
            let rotated: Vec<f32> = indices.iter().map(|&idx| self.centroids[idx]).collect();
            let dequant = self.rht_inverse(&rotated);
            let scale = weight * vec_norm;
            for i in 0..self.head_dim {
                out[i] += scale * dequant[i];
            }
        }
        Ok(out)
    }

    pub fn qjl_precompute(&self, rotated_q: &[f32]) -> Result<Vec<u8>, TurboQuantError> {
        self.check_vec(rotated_q)?;
        Ok(self.qjl_project_signs(rotated_q))
    }

    pub fn score_key_precomputed(
        &self,
        rotated_q: &[f32],
        q_signs: &[u8],
        packed_key: &[u8],
    ) -> Result<f32, TurboQuantError> {
        self.check_vec(rotated_q)?;
        self.check_packed(q_signs, self.head_dim / 8)?;
        self.check_packed(packed_key, self.key_bytes())?;

        let idx_sz = self.index_bytes();
        let qjl_sz = self.head_dim / 8;
        let indices = unpack_indices(&packed_key[..idx_sz], self.head_dim, self.bits);
        let res_norm = f16::from_le_bytes([packed_key[idx_sz + qjl_sz], packed_key[idx_sz + qjl_sz + 1]])
            .to_f32();
        let vec_norm =
            f16::from_le_bytes([packed_key[idx_sz + qjl_sz + 2], packed_key[idx_sz + qjl_sz + 3]])
                .to_f32();

        let centroid_dot = rotated_q
            .iter()
            .zip(indices.iter())
            .map(|(&q, &idx)| q * self.centroids[idx])
            .sum::<f32>();

        let key_signs = &packed_key[idx_sz..idx_sz + qjl_sz];
        let agree: u32 = q_signs
            .iter()
            .zip(key_signs.iter())
            .map(|(&a, &b)| (!(a ^ b)).count_ones())
            .sum();
        let qjl_correction =
            (2.0 * agree as f32 - self.head_dim as f32) * res_norm * SQRT_PI_OVER_2 / self.head_dim as f32;

        Ok(vec_norm * (centroid_dot + qjl_correction))
    }

    fn check_vec(&self, data: &[f32]) -> Result<(), TurboQuantError> {
        if data.len() == self.head_dim {
            Ok(())
        } else {
            Err(TurboQuantError::VectorLen {
                expected: self.head_dim,
                actual: data.len(),
            })
        }
    }

    fn check_packed(&self, data: &[u8], expected_at_least: usize) -> Result<(), TurboQuantError> {
        if data.len() >= expected_at_least {
            Ok(())
        } else {
            Err(TurboQuantError::PackedLen {
                expected_at_least,
                actual: data.len(),
            })
        }
    }

    fn index_bytes(&self) -> usize {
        match self.bits {
            2 => self.head_dim / 4,
            3 => self.head_dim * 3 / 8,
            _ => self.head_dim / 2,
        }
    }

    fn quantize_scalar(&self, x: f32) -> usize {
        self.boundaries.partition_point(|&boundary| x >= boundary)
    }

    fn rht_forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = input
            .iter()
            .zip(self.rht_signs.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        fwht_inplace(&mut out);
        for v in &mut out {
            *v *= self.rht_scale;
        }
        out
    }

    fn rht_inverse(&self, input: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = input.iter().map(|&v| v * self.rht_scale).collect();
        fwht_inplace(&mut out);
        for (v, &sign) in out.iter_mut().zip(self.rht_signs.iter()) {
            *v *= sign;
        }
        out
    }

    fn qjl_project_signs(&self, input: &[f32]) -> Vec<u8> {
        let mut tmp: Vec<f32> = input
            .iter()
            .zip(self.qjl_signs.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        fwht_inplace(&mut tmp);
        let mut out = vec![0u8; self.head_dim / 8];
        for (i, &v) in tmp.iter().enumerate() {
            if v >= 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }
}

fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut len = 1;
    while len < n {
        let step = len << 1;
        for i in (0..n).step_by(step) {
            for j in i..i + len {
                let a = x[j];
                let b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
        }
        len = step;
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn pack_indices(indices: &[usize], bits: usize, out: &mut [u8]) {
    match bits {
        2 => {
            for i in (0..indices.len()).step_by(4) {
                out[i / 4] = (indices[i] as u8 & 3)
                    | ((indices[i + 1] as u8 & 3) << 2)
                    | ((indices[i + 2] as u8 & 3) << 4)
                    | ((indices[i + 3] as u8 & 3) << 6);
            }
        }
        3 => {
            for i in (0..indices.len()).step_by(8) {
                let mut packed = 0u32;
                for j in 0..8 {
                    packed |= ((indices[i + j] as u32) & 7) << (j * 3);
                }
                let o = (i / 8) * 3;
                out[o] = (packed & 0xff) as u8;
                out[o + 1] = ((packed >> 8) & 0xff) as u8;
                out[o + 2] = ((packed >> 16) & 0xff) as u8;
            }
        }
        _ => {
            for i in (0..indices.len()).step_by(2) {
                out[i / 2] = (indices[i] as u8 & 0x0f) | ((indices[i + 1] as u8 & 0x0f) << 4);
            }
        }
    }
}

fn unpack_indices(packed: &[u8], d: usize, bits: usize) -> Vec<usize> {
    let mut indices = vec![0usize; d];
    match bits {
        2 => {
            for i in (0..d).step_by(4) {
                let b = packed[i / 4];
                indices[i] = (b & 3) as usize;
                indices[i + 1] = ((b >> 2) & 3) as usize;
                indices[i + 2] = ((b >> 4) & 3) as usize;
                indices[i + 3] = ((b >> 6) & 3) as usize;
            }
        }
        3 => {
            for i in (0..d).step_by(8) {
                let o = (i / 8) * 3;
                let v = packed[o] as u32 | ((packed[o + 1] as u32) << 8) | ((packed[o + 2] as u32) << 16);
                for j in 0..8 {
                    indices[i + j] = ((v >> (j * 3)) & 7) as usize;
                }
            }
        }
        _ => {
            for i in (0..d).step_by(2) {
                let b = packed[i / 2];
                indices[i] = (b & 0x0f) as usize;
                indices[i + 1] = ((b >> 4) & 0x0f) as usize;
            }
        }
    }
    indices
}

#[derive(Clone, Debug)]
struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn seed(mut seed: u64) -> Self {
        let mut s = [0u64; 4];
        for slot in &mut s {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let na = l2_norm(a);
        let nb = l2_norm(b);
        if na < 1e-10 || nb < 1e-10 {
            0.0
        } else {
            dot(a, b) / (na * nb)
        }
    }

    #[derive(Clone)]
    struct TestRng(u64);

    impl TestRng {
        fn randn(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (((self.0 >> 11) + 1) as f64 / ((1u64 << 53) as f64)).min(1.0) as f32;
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (((self.0 >> 11) + 1) as f64 / ((1u64 << 53) as f64)).min(1.0) as f32;
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn byte_sizes_match_c_reference() {
        let tq = TurboQuant::new(128, 2, 42).unwrap();
        assert_eq!(tq.key_bytes(), 52);
        assert_eq!(tq.value_bytes(), 34);

        let tq = TurboQuant::new(128, 3, 42).unwrap();
        assert_eq!(tq.key_bytes(), 68);
        assert_eq!(tq.value_bytes(), 50);

        let tq = TurboQuant::new(128, 4, 42).unwrap();
        assert_eq!(tq.key_bytes(), 84);
        assert_eq!(tq.value_bytes(), 66);
    }

    #[test]
    fn deterministic_init() {
        let a = TurboQuant::new(128, 3, 42).unwrap();
        let b = TurboQuant::new(128, 3, 42).unwrap();
        assert_eq!(a.rht_signs, b.rht_signs);
        assert_eq!(a.qjl_signs, b.qjl_signs);
    }

    #[test]
    fn rht_preserves_norms() {
        let tq = TurboQuant::new(128, 3, 42).unwrap();
        let mut rng = TestRng(12345);
        let mut max_err = 0.0f32;
        for _ in 0..50 {
            let x: Vec<f32> = (0..128).map(|_| rng.randn()).collect();
            let y = tq.rotate_query(&x).unwrap();
            let err = (l2_norm(&x) - l2_norm(&y)).abs() / l2_norm(&x).max(1e-10);
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4, "max_err={max_err}");
    }

    #[test]
    fn value_roundtrip_has_reasonable_cosine() {
        let tq = TurboQuant::new(128, 3, 42).unwrap();
        let mut rng = TestRng(12345);
        let value: Vec<f32> = (0..128).map(|_| rng.randn() * 0.1).collect();
        let packed = tq.quantize_value(&value).unwrap();
        let dequant = tq.attention_combine(&packed, 1, tq.value_bytes(), &[1.0]).unwrap();
        let sim = cosine_sim(&value, &dequant);
        assert!(sim > 0.80, "cosine={sim}");
    }

    #[test]
    fn score_accuracy_matches_reference_threshold() {
        let tq = TurboQuant::new(128, 3, 42).unwrap();
        let mut rng = TestRng(12345);
        let query: Vec<f32> = (0..128).map(|_| rng.randn() * 0.1).collect();
        let keys: Vec<Vec<f32>> = (0..64)
            .map(|_| (0..128).map(|_| rng.randn() * 0.1).collect())
            .collect();
        let packed: Vec<u8> = keys
            .iter()
            .flat_map(|k| tq.quantize_key(k).unwrap())
            .collect();
        let exact: Vec<f32> = keys.iter().map(|k| dot(&query, k)).collect();
        let rotated_q = tq.rotate_query(&query).unwrap();
        let scores = tq
            .attention_scores(&rotated_q, &packed, keys.len(), tq.key_bytes())
            .unwrap();
        let sim = cosine_sim(&exact, &scores);
        assert!(sim > 0.85, "score cosine={sim}");
    }

    #[test]
    fn attention_combine_has_reasonable_cosine() {
        let tq = TurboQuant::new(128, 3, 42).unwrap();
        let mut rng = TestRng(12345);
        let values: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..128).map(|_| rng.randn() * 0.1).collect())
            .collect();
        let packed: Vec<u8> = values
            .iter()
            .flat_map(|v| tq.quantize_value(v).unwrap())
            .collect();

        let mut weights: Vec<f32> = (0..32).map(|_| rng.randn().exp()).collect();
        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        let mut exact = vec![0.0; 128];
        for (value, &w) in values.iter().zip(weights.iter()) {
            for i in 0..128 {
                exact[i] += w * value[i];
            }
        }

        let combined = tq
            .attention_combine(&packed, values.len(), tq.value_bytes(), &weights)
            .unwrap();
        let sim = cosine_sim(&exact, &combined);
        assert!(sim > 0.80, "cosine={sim}");
    }
}
