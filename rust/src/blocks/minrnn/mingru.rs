use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor},
};

fn g_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // g(x) = x + 0.5 if x >= 0 else sigmoid(x)
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = x.clone().add_scalar(0.5);
    let neg = activation::sigmoid(x);
    neg.mask_where(mask, pos)
}

fn log_g_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // log_g(x) = (relu(x) + 0.5).log() if x >= 0 else -softplus(-x)
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = activation::relu(x.clone()).add_scalar(0.5).log();
    let neg = activation::softplus(x.clone().neg(), 1.0).neg();
    neg.mask_where(mask, pos)
}

// torch.logcumsumexp(x, dim=1) — direct translation
fn logcumsumexp<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.exp().cumsum(1).log()
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    // a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    let [b, s, h] = log_coeffs.dims();
    let device = log_coeffs.device();
    let a_star = Tensor::cat(vec![
        Tensor::zeros([b, 1, h], &device),
        log_coeffs.cumsum(1),
    ], 1);
    // log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    let log_h0_plus_b_star = logcumsumexp(log_values - a_star.clone());
    // log_h = a_star + log_h0_plus_b_star
    let log_h = a_star + log_h0_plus_b_star;
    // return torch.exp(log_h)[:, 1:]
    log_h.slice([0..b, 1..(s + 1), 0..h]).exp()
}

#[derive(Clone, Debug)]
pub struct MinGruState<B: Backend> {
    pub hidden: Tensor<B, 3>,
}

impl<B: Backend> MinGruState<B> {
    pub fn new(hidden: Tensor<B, 3>) -> Self { Self { hidden } }
}

#[derive(Config, Debug)]
pub struct MinGruConfig {
    pub input_features: usize,
    #[config(default = "2")]
    pub expansion_factor: usize,
}

impl MinGruConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGru<B> {
        let hidden_size = self.input_features * self.expansion_factor;
        let linear_z = LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let linear_h = LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let output_projection = LinearConfig::new(hidden_size, self.input_features).with_bias(false).init(device);
        MinGru { linear_z, linear_h, output_projection }
    }
}

#[derive(Module, Debug)]
pub struct MinGru<B: Backend> {
    pub linear_z: Linear<B>,
    pub linear_h: Linear<B>,
    pub output_projection: Linear<B>,
}

impl<B: Backend> MinGru<B> {
    pub fn forward(&self, x: Tensor<B, 3>, states: Option<std::vec::Vec<MinGruState<B>>>) -> (Tensor<B, 3>, std::vec::Vec<MinGruState<B>>) {
        let [b, _seq_len, _] = x.dims();
        let device = x.device();
        let hidden_size = self.linear_z.weight.dims()[1];

        let h0 = if let Some(mut state_vec) = states {
            state_vec.pop().unwrap().hidden
        } else {
            Tensor::<B, 3>::zeros([b, 1, hidden_size], &device)
        };

        // k = -F.softplus(update_gate)
        let update_gate = self.linear_z.forward(x.clone());
        let hidden_state = self.linear_h.forward(x.clone());

        let k = activation::softplus(update_gate, 1.0).neg();
        let log_z = activation::softplus(k.clone().neg(), 1.0).neg();
        let log_coeffs = activation::softplus(k, 1.0).neg();

        let log_h_0 = log_g_3d(h0);
        let log_tilde_h = log_g_3d(hidden_state);

        let log_values = Tensor::cat(vec![log_h_0, log_z + log_tilde_h], 1);
        let output = parallel_scan_log(log_coeffs, log_values);

        let [b_out, s_out, h_out] = output.dims();
        let latest_hidden = output.clone().slice([0..b_out, s_out-1..s_out, 0..h_out]);
        let final_output = self.output_projection.forward(output);

        let mut new_states = std::vec::Vec::new();
        new_states.push(MinGruState::new(latest_hidden));
        (final_output, new_states)
    }

    /// sequential_mode: h_t = (1 - z_t) * h_prev + z_t * g(h_tilde)
    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let z_t = activation::sigmoid(self.linear_z.forward(x_t.clone()));
        let h_tilde = self.linear_h.forward(x_t);
        let one_minus_z_t = z_t.clone().neg().add_scalar(1.0);
        let h_t = one_minus_z_t * h_prev + z_t * g_3d(h_tilde);
        (self.output_projection.forward(h_t.clone()), h_t)
    }
}
