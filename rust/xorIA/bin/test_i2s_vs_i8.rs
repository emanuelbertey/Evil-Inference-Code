use xlstm::blocks::bitlinear::kernel::{I2SKernel, I2STile16Kernel, GROUP_SIZE};

fn make_test_data(batch: usize, in_features: usize, out_features: usize) -> (Vec<u32>, Vec<f32>, Vec<i8>, Vec<f32>) {
    let mut rng_state: u32 = 12345;
    let mut rand_i32 = || -> i32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        rng_state as i32
    };

    let packed_w = I2SKernel::pack_weights(
        &(0..(out_features * in_features))
            .map(|_| {
                let r = rand_i32().abs() % 3;
                match r { 0 => -1.0, 1 => 1.0, _ => 0.0 }
            })
            .collect::<Vec<_>>(),
    );

    let n_groups = (out_features * in_features + GROUP_SIZE - 1) / GROUP_SIZE;
    let scales: Vec<f32> = (0..n_groups).map(|i| 0.1 + (i as f32) * 0.001).collect();

    let x_i8: Vec<i8> = (0..(batch * in_features))
        .map(|_| ((rand_i32().abs() % 255) as i8).wrapping_sub(127))
        .collect();

    let x_f32: Vec<f32> = x_i8.iter().map(|&v| v as f32).collect();

    (packed_w, scales, x_i8, x_f32)
}

fn assert_eq_f32(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!((*av).to_bits(), (*bv).to_bits(), "{}: mismatch at {}: f32={} i8={}", label, i, av, bv);
    }
    println!("  {} PASS", label);
}

fn main() {
    println!("=== I2S vs I8 Tests ===\n");

    let (pw, sc, xi8, xf32) = make_test_data(1, 128, 128);
    assert_eq_f32(
        &I2SKernel::forward_raw(&xf32, 1, &pw, &sc, 128, 128),
        &I2SKernel::forward_raw_i8(&xi8, 1, &pw, &sc, 128, 128),
        "i2s 128x128",
    );

    let (pw, sc, xi8, xf32) = make_test_data(4, 256, 256);
    assert_eq_f32(
        &I2SKernel::forward_raw(&xf32, 4, &pw, &sc, 256, 256),
        &I2SKernel::forward_raw_i8(&xi8, 4, &pw, &sc, 256, 256),
        "i2s 4x256",
    );

    let (pw, sc, xi8, xf32) = make_test_data(1, 512, 512);
    assert_eq_f32(
        &I2STile16Kernel::forward_raw(&xf32, 1, &pw, &sc, 512, 512),
        &I2STile16Kernel::forward_raw_i8(&xi8, 1, &pw, &sc, 512, 512),
        "tile16 512x512",
    );

    let (pw, sc, xi8, xf32) = make_test_data(4, 512, 512);
    assert_eq_f32(
        &I2STile16Kernel::forward_raw(&xf32, 4, &pw, &sc, 512, 512),
        &I2STile16Kernel::forward_raw_i8(&xi8, 4, &pw, &sc, 512, 512),
        "tile16 4x512",
    );

    let (pw, sc, xi8, xf32) = make_test_data(1, 16000, 512);
    assert_eq_f32(
        &I2SKernel::forward_raw(&xf32, 1, &pw, &sc, 512, 16000),
        &I2SKernel::forward_raw_i8(&xi8, 1, &pw, &sc, 512, 16000),
        "i2s 16000x512",
    );

    println!("\n=== ALL PASS ===");
}
