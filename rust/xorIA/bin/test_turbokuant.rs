use std::error::Error;

use xlstm::blocks::turbokuant::TurboQuant;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (((self.0 >> 11) + 1) as f64 / ((1u64 << 53) as f64)).min(1.0) as f32;
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (((self.0 >> 11) + 1) as f64 / ((1u64 << 53) as f64)).min(1.0) as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    test_turbokuant_main()
}

pub fn test_turbokuant_main() -> Result<(), Box<dyn Error>> {
    println!("=== TurboQuant KV-cache compression ===");

    test_byte_sizes()?;
    test_deterministic()?;
    test_norm_preservation()?;
    test_value_roundtrip()?;
    test_score_accuracy()?;
    test_attention_combine()?;
    test_bit_widths()?;

    println!("=== All TurboQuant checks passed ===");
    Ok(())
}

fn test_byte_sizes() -> Result<(), Box<dyn Error>> {
    print!("test_byte_sizes...");

    let tq = TurboQuant::new(128, 2, 42)?;
    assert_eq!(tq.key_bytes(), 52);
    assert_eq!(tq.value_bytes(), 34);

    let tq = TurboQuant::new(128, 3, 42)?;
    assert_eq!(tq.key_bytes(), 68);
    assert_eq!(tq.value_bytes(), 50);

    let tq = TurboQuant::new(128, 4, 42)?;
    assert_eq!(tq.key_bytes(), 84);
    assert_eq!(tq.value_bytes(), 66);

    println!(" PASS");
    Ok(())
}

fn test_deterministic() -> Result<(), Box<dyn Error>> {
    print!("test_deterministic...");

    let a = TurboQuant::new(128, 3, 42)?;
    let b = TurboQuant::new(128, 3, 42)?;
    assert_eq!(a.rht_signs, b.rht_signs);
    assert_eq!(a.qjl_signs, b.qjl_signs);

    println!(" PASS");
    Ok(())
}

fn test_norm_preservation() -> Result<(), Box<dyn Error>> {
    print!("test_norm_preservation...");

    let tq = TurboQuant::new(128, 3, 42)?;
    let mut rng = TestRng(12345);
    let mut max_err = 0.0f32;

    for _ in 0..50 {
        let x: Vec<f32> = (0..128).map(|_| rng.randn()).collect();
        let y = tq.rotate_query(&x)?;
        let err = (l2_norm(&x) - l2_norm(&y)).abs() / l2_norm(&x).max(1e-10);
        max_err = max_err.max(err);
    }

    print!(" max_rel_err={max_err:.6}");
    assert!(max_err < 1e-4);
    println!(" PASS");
    Ok(())
}

fn test_value_roundtrip() -> Result<(), Box<dyn Error>> {
    print!("test_value_roundtrip...");

    let tq = TurboQuant::new(128, 3, 42)?;
    let mut rng = TestRng(12345);
    let value: Vec<f32> = (0..128).map(|_| rng.randn() * 0.1).collect();
    let packed = tq.quantize_value(&value)?;
    let dequant = tq.attention_combine(&packed, 1, tq.value_bytes(), &[1.0])?;
    let sim = cosine_sim(&value, &dequant);

    print!(" cosine_sim={sim:.4}");
    assert!(sim > 0.80);
    println!(" PASS");
    Ok(())
}

fn test_score_accuracy() -> Result<(), Box<dyn Error>> {
    print!("test_score_accuracy...");

    let tq = TurboQuant::new(128, 3, 42)?;
    let mut rng = TestRng(12345);
    let query: Vec<f32> = (0..128).map(|_| rng.randn() * 0.1).collect();
    let keys: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..128).map(|_| rng.randn() * 0.1).collect())
        .collect();
    let packed: Vec<u8> = keys.iter().flat_map(|k| tq.quantize_key(k).unwrap()).collect();
    let exact: Vec<f32> = keys.iter().map(|k| dot(&query, k)).collect();
    let rotated_q = tq.rotate_query(&query)?;
    let scores = tq.attention_scores(&rotated_q, &packed, keys.len(), tq.key_bytes())?;
    let sim = cosine_sim(&exact, &scores);

    print!(" score_cosine={sim:.4}");
    assert!(sim > 0.85);
    println!(" PASS");
    Ok(())
}

fn test_attention_combine() -> Result<(), Box<dyn Error>> {
    print!("test_attention_combine...");

    let tq = TurboQuant::new(128, 3, 42)?;
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

    let combined = tq.attention_combine(&packed, values.len(), tq.value_bytes(), &weights)?;
    let sim = cosine_sim(&exact, &combined);

    print!(" cosine_sim={sim:.4}");
    assert!(sim > 0.80);
    println!(" PASS");
    Ok(())
}

fn test_bit_widths() -> Result<(), Box<dyn Error>> {
    print!("test_bit_widths...");

    let mut rng = TestRng(12345);
    let value: Vec<f32> = (0..128).map(|_| rng.randn() * 0.1).collect();
    let mut prev = 0.0;

    for bits in 2..=4 {
        let tq = TurboQuant::new(128, bits, 42)?;
        let packed = tq.quantize_value(&value)?;
        let dequant = tq.attention_combine(&packed, 1, tq.value_bytes(), &[1.0])?;
        let sim = cosine_sim(&value, &dequant);
        print!(" {bits}b={sim:.3}");
        assert!(sim >= prev - 0.01);
        prev = sim;
    }

    println!(" PASS");
    Ok(())
}
