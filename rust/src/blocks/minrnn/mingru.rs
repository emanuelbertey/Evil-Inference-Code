use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Distribution, Tensor},
};

fn g_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = x.clone().add_scalar(0.5);
    let neg = activation::sigmoid(x);
    neg.mask_where(mask, pos)
}

fn log_g_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()).add_scalar(0.5)).log();
    let neg = activation::softplus(x.clone().neg(), 1.0).neg();
    neg.mask_where(mask, pos)
}

// Cumsum iterativo (Burn-friendly Autodiff) - Punto 2.18 del README_FIX
fn cumsum_3d_stable<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, h] = x.dims();
    let mut results = std::vec::Vec::with_capacity(s);
    let mut acc = Tensor::<B, 3>::zeros([b, 1, h], &x.device());
    
    for i in 0..s {
        acc = acc + x.clone().slice([0..b, i..i+1, 0..h]);
        results.push(acc.clone());
    }
    Tensor::cat(results, 1)
}

// logcumsumexp iterativo con estabilidad local - Similar a PyTorch
fn logcumsumexp_stable<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, h] = x.dims();
    let device = x.device();
    let mut results = std::vec::Vec::with_capacity(s);
    
    let mut current_lse = x.clone().slice([0..b, 0..1, 0..h]);
    results.push(current_lse.clone());

    for i in 1..s {
        let x_i = x.clone().slice([0..b, i..i+1, 0..h]);
        let m = current_lse.clone().max_dim(1); // GRADIENTE PURO sin detach
        let mask = x_i.clone().greater_equal(m.clone());
        let m_new = m.clone().mask_where(mask, x_i.clone().max_dim(1));// GRADIENTE PURO sin detach
        
        current_lse = ((current_lse - m_new.clone()).exp() + (x_i - m_new.clone()).exp()).log() + m_new;
        results.push(current_lse.clone());
    }
    Tensor::cat(results, 1)
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, h] = log_coeffs.dims();
    let device = log_coeffs.device();
    
    // a_star = pad(cumsum(log_coeffs), (0,0, 1,0))
    let cs = cumsum_3d_stable(log_coeffs);
    let zeros = Tensor::<B, 3>::zeros([b, 1, h], &device);
    let a_star = Tensor::cat(vec![zeros, cs], 1);
    
    // log_h = a_star + logcumsumexp(log_values - a_star)
    let log_h = a_star.clone() + logcumsumexp_stable(log_values - a_star);
    
    let [b_out, s_out, h_out] = log_h.dims();
    log_h.slice([0..b_out, 1..s_out, 0..h_out]).exp()
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
        
        MinGru {
            linear_z,
            linear_h,
            output_projection,
        }
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
        
        // REQUERIR GRADIENTE en h0 (Punto 1.13 del README_FIX)
        let h0 = if let Some(mut state_vec) = states {
            state_vec.pop().unwrap().hidden
        } else {
            Tensor::<B, 3>::zeros([b, 1, hidden_size], &device)
        };
        
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

    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let z_t = activation::sigmoid(self.linear_z.forward(x_t.clone()));
        let h_tilde = self.linear_h.forward(x_t.clone());
        let one_minus_z_t = z_t.clone().neg().add_scalar(1.0);
        let h_t = one_minus_z_t * h_prev + z_t * g_3d(h_tilde);
        (self.output_projection.forward(h_t.clone()), h_t)
    }
}
