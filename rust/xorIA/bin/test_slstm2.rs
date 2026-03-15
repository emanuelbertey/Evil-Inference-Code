use xlstm::{SLstm, SLstmConfig, SLstmState};
use burn::tensor::{Tensor, Distribution, TensorData};
use burn::tensor::backend::Backend;
use burn::tensor::activation;
use rand::Rng;

type TestBackend = burn_ndarray::NdArray<f32>;

// ─── 1. Equivalence ─────────────────────────────────────────────────────────
fn run_equivalence() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let batch_size = 2;
    let seq_len = 5;
    let hidden_size = 32;
    
    let config = SLstmConfig::new(hidden_size).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    
    let input_seq: Tensor<TestBackend, 3> =
        Tensor::random([batch_size, seq_len, hidden_size], Distribution::Default, &device);
    
    // Parallel forward
    let (output_parallel, final_states_parallel): (Tensor<TestBackend, 3>, _) = 
        slstm.forward_with_state(input_seq.clone(), None);
    
    // Recurrent forward init
    let mut state = slstm.empty_state(batch_size, &device);

    let mut outputs: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let input_t = input_seq
            .clone()
            .slice([0..batch_size, t..(t + 1), 0..hidden_size])
            .reshape([batch_size, hidden_size]); // A 2D (B, D) para el .step()
            
        let (h_new, new_state) = slstm.step(input_t, state);
        outputs.push(h_new.unsqueeze_dim(1)); // Lo volvemos a pasar a 3D (B, 1, D)
        state = new_state;
    }
    
    let output_recurrent: Tensor<TestBackend, 3> = Tensor::cat(outputs, 1);
    
    let diff = (output_parallel.clone() - output_recurrent.clone()).abs().mean().into_scalar();
    println!("Equivalence - Output diff: {:.2e}", diff);
    
    // Cell at index 1, Normalizer at index 2
    let cell_diff = (final_states_parallel.0.1 - state.0.1).abs().mean().into_scalar();
    println!("Equivalence - Cell diff: {:.2e}", cell_diff);
    
    let norm_diff = (final_states_parallel.0.2 - state.0.2).abs().mean().into_scalar();
    println!("Equivalence - Normalizer diff: {:.2e}", norm_diff);
}

// ─── 2. Stability ───────────────────────────────────────────────────────────
fn run_stability() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let batch_size = 2;
    let seq_len = 20;
    let hidden_size = 32;
    let config = SLstmConfig::new(hidden_size).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    
    let input = Tensor::<TestBackend, 3>::random([batch_size, seq_len, hidden_size], Distribution::Default, &device) * 10.0;
    let (out, states): (Tensor<TestBackend, 3>, _) = slstm.forward_with_state(input, None);
    
    let out_mean = out.abs().mean().into_scalar();
    let n_mean = states.0.2.abs().mean().into_scalar();
    let c_mean = states.0.1.abs().mean().into_scalar();
    println!("Estabilidad: |h|={:.4}, |c|={:.4}, |n|={:.4}", out_mean, c_mean, n_mean);
}

// ─── 3. Monotonicity ────────────────────────────────────────────────────────
fn run_monotonic() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let seq_len = 20;
    let hidden_size = 8;
    let config = SLstmConfig::new(hidden_size).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    let ones = Tensor::<TestBackend, 3>::ones([1, seq_len, hidden_size], &device);
    let (out, _): (Tensor<TestBackend, 3>, _) = slstm.forward_with_state(ones, None);
    let mut prev = 0.0f32;
    let mut non_decrease = 0usize;
    for t in 0..seq_len {
        let val = out.clone().slice([0..1, t..t+1, 0..hidden_size]).abs().mean().into_scalar();
        if val >= prev { non_decrease += 1; }
        prev = val;
    }
    println!("Monotonicidad: {}/{}", non_decrease, seq_len);
}

// ─── 4. Compare vs LSTM ─────────────────────────────────────────────────────
struct SimpleLstmCell<B: Backend> {
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    b: Tensor<B, 1>,
}

impl<B: Backend> SimpleLstmCell<B> {
    fn new(input: usize, hidden: usize, device: &B::Device) -> Self {
        let w_ih = Tensor::<B, 2>::random([4 * hidden, input], Distribution::Default, device);
        let w_hh = Tensor::<B, 2>::random([4 * hidden, hidden], Distribution::Default, device);
        let b = Tensor::<B, 1>::zeros([4 * hidden], device);
        Self { w_ih, w_hh, b }
    }
    fn step(&self, x: Tensor<B, 2>, h: Tensor<B, 2>, c: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let gates = x.matmul(self.w_ih.clone().transpose()) + h.matmul(self.w_hh.clone().transpose()) + self.b.clone().unsqueeze_dim(0);
        let chunks = gates.chunk(4, 1);
        let i = activation::sigmoid(chunks[0].clone());
        let f = activation::sigmoid(chunks[1].clone());
        let g = chunks[2].clone().tanh();
        let o = activation::sigmoid(chunks[3].clone());
        let c_new = f * c + i * g;
        let h_new = o * c_new.clone().tanh();
        (h_new, c_new)
    }
}

fn run_compare_lstm() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let batch_size = 1;
    let seq_len = 20;
    let hidden_size = 16;
    let x = Tensor::<TestBackend, 3>::random([batch_size, seq_len, hidden_size], Distribution::Default, &device);
    let slstm: SLstm<TestBackend> = SLstmConfig::new(hidden_size).init(&device);
    let (h_seq, _): (Tensor<TestBackend, 3>, _) = slstm.forward_with_state(x.clone(), None);
    
    let lstm_cell = SimpleLstmCell::<TestBackend>::new(hidden_size, hidden_size, &device);
    let mut h = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
    let mut c = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
    let mut hs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = x.clone().slice([0..batch_size, t..t+1, 0..hidden_size]).reshape([batch_size, hidden_size]);
        let (h_new, c_new) = lstm_cell.step(x_t, h, c);
        h = h_new;
        c = c_new;
        hs.push(h.clone().unsqueeze_dim(1));
    }
    let h_seq_lstm = Tensor::<TestBackend, 3>::cat(hs, 1);
    println!("Average |h|: sLSTM={:.4}, LSTM={:.4}", h_seq.abs().mean().into_scalar(), h_seq_lstm.abs().mean().into_scalar());
}

// ─── 5. Gradient Flow ───────────────────────────────────────────────────────
fn run_grad_input() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    type AtDiffBackend = burn::backend::Autodiff<TestBackend>;
    let hidden_size = 16;
    let slstm: SLstm<AtDiffBackend> = SLstmConfig::new(hidden_size).init(&device);
    let x = Tensor::<AtDiffBackend, 3>::random([1, 10, hidden_size], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let (h_seq, _): (Tensor<AtDiffBackend, 3>, _) = slstm.forward_with_state(x.clone(), None);
    let loss = h_seq.slice([0..1, 9..10, 0..hidden_size]).sum();
    let grads = loss.backward();
    let x_grad = x.grad(&grads).expect("Grad exist");
    println!("Grad mean |dL/dx|: {:.6}", x_grad.abs().mean().into_scalar());
}

// ─── 6. Copy Count Task ─────────────────────────────────────────────────────
use burn::module::Module;
#[derive(Module, Debug)]
struct CopyTestModel<B: Backend> {
    slstm: SLstm<B>,
    linear: burn::nn::Linear<B>,
}

impl<B: Backend> CopyTestModel<B> {
    fn new(hidden_size: usize, device: &B::Device) -> Self {
        Self {
            slstm: SLstmConfig::new(hidden_size).init(device),
            linear: burn::nn::LinearConfig::new(hidden_size, 2).init(device),
        }
    }
}

fn run_copy_count() {
    use burn::optim::{AdamConfig, Optimizer};
    type AtDiffBackend = burn::backend::Autodiff<TestBackend>;
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let mut rng = rand::rng(); 
    let mut model = CopyTestModel::<AtDiffBackend>::new(16, &device);
    let mut optim = AdamConfig::new().init();
    let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init(&device);

    for epoch in 0..50 {
        let mut xs = Vec::new(); let mut ys = Vec::new();
        for _ in 0..64 {
            let mut first = 0i64;
            for t in 0..12 {
                let bit = if rng.random_bool(0.5) { 1 } else { 0 };
                if t == 0 { first = bit; }
                xs.push(bit as f32);
            }
            ys.push(first);
        }
        let x = Tensor::<AtDiffBackend, 3>::from_data(TensorData::new(xs, [64, 12, 1]), &device);
        let x = x.repeat(&[1, 1, 16]);
        let y = Tensor::<AtDiffBackend, 1, burn::tensor::Int>::from_data(TensorData::new(ys, [64]), &device);

        let (h_seq, _): (Tensor<AtDiffBackend, 3>, _) = model.slstm.forward_with_state(x, None);
        let last = h_seq.slice([0..64, 11..12, 0..16]).reshape([64, 16]);
        let logits = model.linear.forward(last);
        let loss = loss_fn.forward(logits, y);
        let l_val = loss.clone().into_scalar();
        let grads = loss.backward();
        let grads_p = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(0.01, model, grads_p);

        if epoch % 10 == 0 { println!("copy_count epoch {}: loss={:.4}", epoch+1, l_val); }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("Running all sLSTM v2 tests...");
        run_equivalence(); run_stability(); run_monotonic();
        run_compare_lstm(); run_grad_input(); run_copy_count();
        return;
    }
    match args[1].as_str() {
        "equiv" => run_equivalence(),
        "grad" => run_grad_input(),
        "copy_count" => run_copy_count(),
        "stability" => run_stability(),
        "monotonic" => run_monotonic(),
        "compare_lstm" => run_compare_lstm(),
        _ => eprintln!("Unknown mode"),
    }
}
