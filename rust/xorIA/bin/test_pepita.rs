use burn::module::Module;
use burn::nn;
use burn::tensor::{Tensor, Distribution, activation, backend::Backend};
use burn_flex::Flex;
use burn_autodiff::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use xlstm::blocks::forward_forward::pepita::{PepitaConfig, init_feedback};

type MyBackend = Autodiff<Flex<f32>>;

#[derive(Module, Debug)]
struct BpMlp<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
    l4: nn::Linear<B>,
}

impl<B: Backend> BpMlp<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = activation::relu(self.l1.forward(x));
        let h2 = activation::relu(self.l2.forward(h1));
        let h3 = activation::relu(self.l3.forward(h2));
        self.l4.forward(h3)
    }
}

fn main() {
    let device = Default::default();
    let n = 512;
    let lr = 3e-4f32;
    let epochs = 500;

    // ── Synthetic non-linear data ──
    let x_f = Tensor::<Flex<f32>, 2>::random([n, 2], Distribution::Uniform(-1.5, 1.5), &device);
    let noise = Tensor::<Flex<f32>, 2>::random([n, 1], Distribution::Normal(0.0, 0.1), &device);
    let x1 = x_f.clone().slice([0..n, 0..1]);
    let x2 = x_f.clone().slice([0..n, 1..2]);
    let y_f: Tensor::<Flex<f32>, 2> = x1.clone() * x2.clone() + 0.5 * x1.powf_scalar(2.0) - x2.clone() + noise;

    let x_ad = Tensor::<MyBackend, 2>::random([n, 2], Distribution::Uniform(-1.5, 1.5), &device);
    let noise_ad = Tensor::<MyBackend, 2>::random([n, 1], Distribution::Normal(0.0, 0.1), &device);
    let x1_ad = x_ad.clone().slice([0..n, 0..1]);
    let x2_ad = x_ad.clone().slice([0..n, 1..2]);
    let y_ad: Tensor::<MyBackend, 2> = x1_ad.clone() * x2_ad.clone() + 0.5 * x1_ad.powf_scalar(2.0) - x2_ad.clone() + noise_ad;

    let nt = 128;
    let x_tf = Tensor::<Flex<f32>, 2>::random([nt, 2], Distribution::Uniform(-1.5, 1.5), &device);
    let noise_t = Tensor::<Flex<f32>, 2>::random([nt, 1], Distribution::Normal(0.0, 0.1), &device);
    let x1_tf = x_tf.clone().slice([0..nt, 0..1]);
    let x2_tf = x_tf.clone().slice([0..nt, 1..2]);
    let y_tf: Tensor::<Flex<f32>, 2> = x1_tf.clone() * x2_tf.clone() + 0.5 * x1_tf.powf_scalar(2.0) - x2_tf.clone() + noise_t;

    let x_tad = Tensor::<MyBackend, 2>::random([nt, 2], Distribution::Uniform(-1.5, 1.5), &device);
    let noise_tad = Tensor::<MyBackend, 2>::random([nt, 1], Distribution::Normal(0.0, 0.1), &device);
    let x1_tad = x_tad.clone().slice([0..nt, 0..1]);
    let x2_tad = x_tad.clone().slice([0..nt, 1..2]);
    let y_tad: Tensor::<MyBackend, 2> = x1_tad.clone() * x2_tad.clone() + 0.5 * x1_tad.powf_scalar(2.0) - x2_tad.clone() + noise_tad;

    // ── 4-layer PEPITA MLP ──
    let mut p1 = PepitaConfig::new(2, 16).init(&device);
    let mut p2 = PepitaConfig::new(16, 16).init(&device);
    let mut p3 = PepitaConfig::new(16, 8).init(&device);
    let mut p4 = PepitaConfig::new(8, 1).init(&device);

    // Single feedback matrix F mapping output (1) → input (2), as in the paper
    let fb = init_feedback::<Flex<f32>>(2, 1, &device);

    // ── 4-layer BP MLP ──
    let mut bp = BpMlp {
        l1: nn::LinearConfig::new(2, 16).with_bias(true).init(&device),
        l2: nn::LinearConfig::new(16, 16).with_bias(true).init(&device),
        l3: nn::LinearConfig::new(16, 8).with_bias(true).init(&device),
        l4: nn::LinearConfig::new(8, 1).with_bias(true).init(&device),
    };
    let mut optim = AdamConfig::new().init();

    println!("=== 4-Layer PEPITA vs Backprop ===");
    println!("y = x1*x2 + 0.5*x1² - x2  | Arch: 2→16→16→8→1 (ReLU, linear out)\n");
    println!("Epoch  PepitaLoss  BPLoss");

    for e in 0..epochs {
        // ── PEPITA (paper Algorithm S1) ──
        // 1. Standard forward: save post-activations for hidden diff
        let h1 = activation::relu(p1.forward_standard(&x_f));
        let h2 = activation::relu(p2.forward_standard(&h1));
        let h3 = activation::relu(p3.forward_standard(&h2));
        let y_pred = p4.forward_standard(&h3);
        let err_1 = y_f.clone() - y_pred.clone();  // error from first pass

        // 2. Perturb input with error projected through F, then forward again
        let x_tilde = x_f.clone() + err_1.clone().matmul(fb.clone().transpose());
        let h1_t = activation::relu(p1.forward_standard(&x_tilde));
        let h2_t = activation::relu(p2.forward_standard(&h1_t));
        let h3_t = activation::relu(p3.forward_standard(&h2_t));
        let y_tilde = p4.forward_standard(&h3_t);
        let err_2 = y_f.clone() - y_tilde.clone();  // error from second pass

        // 3. Weight updates (paper formulas, S1)
        // Output layer: ΔW ∝ (t - ỹ) · h̃^T  (error from SECOND pass, with perturbed h)
        p4 = p4.update_output(&err_2, &h3_t, lr);
        // Hidden layers: ΔW ∝ (h̃ - h_std) · h̃_prev^T
        p3 = p3.update_hidden(&h3, &h3_t, &h2_t, lr);
        p2 = p2.update_hidden(&h2, &h2_t, &h1_t, lr);
        p1 = p1.update_hidden(&h1, &h1_t, &x_tilde, lr);

        // ── Backprop ──
        let y_bp = bp.forward(x_ad.clone());
        let loss = ((y_bp - y_ad.clone()).powf_scalar(2.0)).mean();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &bp);
        bp = optim.step(lr as f64, bp, gp);

        if e % 50 == 0 || e == epochs - 1 {
            let h1t = activation::relu(p1.forward_standard(&x_tf));
            let h2t = activation::relu(p2.forward_standard(&h1t));
            let h3t = activation::relu(p3.forward_standard(&h2t));
            let tp = p4.forward_standard(&h3t);
            let pl = ((tp - y_tf.clone()).powf_scalar(2.0)).mean().into_scalar();

            let btp = bp.forward(x_tad.clone());
            let bl = ((btp - y_tad.clone()).powf_scalar(2.0)).mean().into_scalar();

            println!("{:<6} {:.6}  {:.6}", e, pl, bl);
        }
    }
}
