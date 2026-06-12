use burn::tensor::Tensor;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;

type B = NdArray<f32>;

fn log_sigmoid(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::activation::softplus(x.neg(), 1.0).neg()
}

fn cumsum_matrix(x: Tensor<B, 4>, s: usize, device: &<B as Backend>::Device) -> Tensor<B, 4> {
    let [b, nh, _s, _] = x.dims();
    
    let zeros = Tensor::<B, 4>::zeros([b, nh, 1, 1], device);
    
    let mut outputs = Vec::with_capacity(s + 1);
    outputs.push(zeros.clone());
    
    let mut current = zeros;
    for i in 0..s {
        let step_x = x.clone().narrow(2, i, 1);
        current = current + step_x;
        outputs.push(current.clone());
    }
    
    let log_fgates_cumsum = Tensor::cat(outputs, 2);
    
    let mut reps = Vec::with_capacity(s + 1);
    for _ in 0..(s + 1) {
        reps.push(log_fgates_cumsum.clone());
    }
    let rep_cumsum = Tensor::cat(reps, 3);
    
    let matrix = rep_cumsum.clone() - rep_cumsum.swap_dims(2, 3);
    matrix.narrow(2, 1, s).narrow(3, 1, s)
}

fn main() {
    let device = Default::default();
    // Tensor determinístico [1, 1, 4, 1]
    let vals: Vec<f32> = vec![0.1, -0.5, 1.2, -0.1];
    let fgate_preact = Tensor::<B, 1>::from_floats(vals.as_slice(), &device).reshape([1, 1, 4, 1]);
    
    let log_fgates = log_sigmoid(fgate_preact);
    println!("log_fgates:\n{}", log_fgates.clone());
    
    let log_fg_cumsum = cumsum_matrix(log_fgates.clone(), 4, &device);
    
    // Matriz de decaimiento (antes de máscara infinita, ya que cumsum_matrix retorna la submatriz (S, S))
    // Python hace: ltr mask con -inf
    let mask = Tensor::<B, 2>::ones([4, 4], &device).tril(0).unsqueeze::<3>().unsqueeze::<4>();
    let log_fg_matrix = log_fg_cumsum.mask_fill(mask.equal_elem(0.0), -1e10);
    
    println!("\nFINAL log_fg_matrix (Rust):");
    println!("{}", log_fg_matrix);
}
