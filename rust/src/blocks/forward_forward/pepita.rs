use burn::prelude::*;
use burn::module::Param;
use burn::nn;
use burn::tensor::{Tensor, backend::Backend, Distribution};

#[derive(Config, Debug)]
pub struct PepitaConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct PepitaLayer<B: Backend> {
    pub linear: nn::Linear<B>,
}

impl PepitaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PepitaLayer<B> {
        PepitaLayer {
            linear: nn::LinearConfig::new(self.input_dim, self.output_dim)
                .with_bias(self.bias)
                .init(device),
        }
    }
}

impl<B: Backend> PepitaLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }

    /// Standard forward: y = Wx + b
    pub fn forward_standard(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x.clone())
    }

    /// Modulated forward: ŷ = W(x + F·e) + b
    pub fn forward_modulated(&self, x: &Tensor<B, 2>, err: &Tensor<B, 2>, fb: &Tensor<B, 2>) -> Tensor<B, 2> {
        let x_tilde = x.clone() + err.clone().matmul(fb.clone());
        self.linear.forward(x_tilde)
    }

    /// PEPITA weight update for output layer: W = W + η · x̃ᵀ · err / n
    /// (err = target - pred; pseudocode uses e = pred - target = -err, so sign flips)
    pub fn update_output(&self, err: &Tensor<B, 2>, x_tilde: &Tensor<B, 2>, lr: f32) -> Self {
        let n = err.dims()[0] as f32;
        let dw = x_tilde.clone().transpose().matmul(err.clone()) / n;
        let w_new = self.linear.weight.val() + dw * lr;
        let bias_new = self.linear.bias.as_ref().map(|b| {
            let d_out = b.val().dims()[0];
            let db = err.clone().mean_dim(0).reshape([d_out]);
            b.val() + db * lr
        });
        let record = nn::LinearRecord {
            weight: Param::from_tensor(w_new),
            bias: bias_new.map(Param::from_tensor),
        };
        PepitaLayer {
            linear: self.linear.clone().load_record(record),
        }
    }

    /// PEPITA weight update for hidden layer: ΔW = h_prev_tildeᵀ · δ / n,  W = W - η·ΔW
    /// δ = z_tilde - z_std (pre-activation difference, before non-linearity)
pub fn update_hidden(&self, h_std: &Tensor<B, 2>, h_tilde: &Tensor<B, 2>, h_prev_tilde: &Tensor<B, 2>, lr: f32) -> Self {
        let n = h_std.dims()[0] as f32;
        let delta = h_tilde.clone() - h_std.clone();
        let dw = h_prev_tilde.clone().transpose().matmul(delta.clone()) / n;
        let w_new = self.linear.weight.val() + dw * lr;
        let bias_new = self.linear.bias.as_ref().map(|b| {
            let d_out = b.val().dims()[0];
            let db = delta.clone().mean_dim(0).reshape([d_out]);
            b.val() + db * lr
        });
        let record = nn::LinearRecord {
            weight: Param::from_tensor(w_new),
            bias: bias_new.map(Param::from_tensor),
        };
        PepitaLayer {
            linear: self.linear.clone().load_record(record),
        }
    }
}

/// Initialize a random feedback matrix F with He-like scaling × 0.05 (paper)
pub fn init_feedback<B: Backend>(input_dim: usize, output_dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let limit = (1.0 / output_dim as f32).sqrt() as f64;
    Tensor::random([input_dim, output_dim], Distribution::Uniform(-limit, limit), device) * 0.05
}
