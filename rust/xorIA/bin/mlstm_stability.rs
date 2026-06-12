use burn::tensor::{Tensor, Distribution};
use burn::backend::ndarray::NdArray;
use xlstm::blocks::mlstm::backends::recurrent_step_stabilized_simple;

type TestBackend = NdArray<f32>;

fn main() {
    println!("=== mLSTM STABILITY EXPLOSION TEST (RUST) ===");
    let device = Default::default();
    
    let batch_size = 1;
    let nh = 1;
    let dh = 4;
    
    // Initialize states
    let mut c_state = Tensor::<TestBackend, 4>::zeros([batch_size, nh, dh, dh], &device);
    let mut n_state = Tensor::<TestBackend, 4>::zeros([batch_size, nh, dh, 1], &device);
    let mut m_state = Tensor::<TestBackend, 4>::zeros([batch_size, nh, 1, 1], &device);
    
    // Simple inputs
    let q = Tensor::<TestBackend, 4>::ones([batch_size, nh, 1, dh], &device);
    let k = Tensor::<TestBackend, 4>::ones([batch_size, nh, 1, dh], &device);
    let v = Tensor::<TestBackend, 4>::ones([batch_size, nh, 1, dh], &device);
    
    let test_values = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0, 50000.0];
    
    for &val in test_values.iter() {
        let igate_preact = Tensor::<TestBackend, 4>::ones([batch_size, nh, 1, 1], &device).mul_scalar(val);
        let fgate_preact = Tensor::<TestBackend, 4>::ones([batch_size, nh, 1, 1], &device).mul_scalar(val / 2.0);
        
        let (h, (next_c, next_n, next_m)) = recurrent_step_stabilized_simple::<TestBackend>(
            c_state,
            n_state,
            m_state,
            q.clone(),
            k.clone(),
            v.clone(),
            igate_preact,
            fgate_preact,
        );
        
        let h_data = h.clone().into_data();
        let h_slice = h_data.as_slice::<f32>().unwrap();
        let m_data = next_m.clone().into_data();
        let m_val = m_data.as_slice::<f32>().unwrap()[0];
        
        let mut is_nan = false;
        let mut is_inf = false;
        let mut sum = 0.0;
        for &x in h_slice {
            if x.is_nan() { is_nan = true; }
            if x.is_infinite() { is_inf = true; }
            sum += x;
        }
        let mean = sum / h_slice.len() as f32;
        
        let status = if !is_nan && !is_inf { "✅ OK" } else { "❌ EXPLODED (NaN/Inf)" };
        println!("Pre-act: {:8.1} | Result Mean: {:10.6} | State m: {:8.1} | {}", val, mean, m_val, status);
        
        if is_nan || is_inf {
            println!("   !!! Instability detected at value {}", val);
        }
        
        // Update states
        c_state = next_c;
        n_state = next_n;
        m_state = next_m;
    }
}
