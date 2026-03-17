use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::Distribution;
use burn::module::Param;

#[derive(Config, Debug)]
pub struct MinLstmConfig {
    pub input_features: usize,
    #[config(default = 2)]
    pub expansion_factor: usize,
}

#[derive(Module, Debug)]
pub struct MinLstm<B: Backend> {
    pub linear_f: nn::Linear<B>,
    pub linear_i: nn::Linear<B>,
    pub linear_h: nn::Linear<B>,
    pub output_projection: nn::Linear<B>,
}

#[derive(Clone, Debug)]
pub struct MinLstmState<B: Backend> {
    pub hidden: Tensor<B, 3>,
}

impl<B: Backend> MinLstmState<B> {
    pub fn new(hidden: Tensor<B, 3>) -> Self {
        Self { hidden }
    }
}

impl MinLstmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinLstm<B> {
        let hidden_size = self.input_features * self.expansion_factor;
        
        // Arquitectura idéntica a minGRU: Sin bias en ninguna capa para mantener la estructura estable
        let l_f = nn::LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let l_i = nn::LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let l_h = nn::LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let proj = nn::LinearConfig::new(hidden_size, self.input_features).with_bias(false).init(device);
        
        let init_weights = |linear: nn::Linear<B>, in_dim: usize| {
            let k = (1.0 / in_dim as f32).sqrt();
            let out_dim = linear.weight.dims()[1];
            linear.load_record(nn::LinearRecord {
                weight: Param::from_tensor(Tensor::random([in_dim, out_dim], Distribution::Uniform(-k as f64, k as f64), device)),
                bias: None,
            })
        };

        MinLstm { 
            linear_f: init_weights(l_f, self.input_features),
            linear_i: init_weights(l_i, self.input_features),
            linear_h: init_weights(l_h, self.input_features),
            output_projection: init_weights(proj, hidden_size) 
        }
    }
}

// Función auxiliar para emular torch.logcumsumexp de forma estable y paralela en f32
fn log_cumsum_exp<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let max = x.clone().detach().max_dim(1);
    let min = x.clone().detach().neg().max_dim(1).neg(); 
    let m = (max + min) / 2.0;
    
    (x - m.clone()).clamp(-85.0, 85.0).exp().cumsum(1).log() + m
}

fn log_g<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()) + 0.5).log();
    let neg = activation::softplus(x.neg(), 1.0).neg(); 
    neg.mask_where(mask, pos)
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, _s_plus_1, h] = log_values.dims();
    let device = log_values.device();
    
    let a_star = Tensor::cat(vec![
        Tensor::zeros([b, 1, h], &device),
        log_coeffs.cumsum(1)
    ], 1);
    
    let x = log_values - a_star.clone();
    let log_h0_plus_b_star = log_cumsum_exp(x);
    
    let log_h = a_star + log_h0_plus_b_star;
    let dims = log_h.dims();
    
    log_h.exp().slice([0..b, 1..dims[1], 0..h])
}

impl<B: Backend> MinLstm<B> {
    pub fn forward(&self, x: Tensor<B, 3>, states: Option<Vec<MinLstmState<B>>>) -> (Tensor<B, 3>, Vec<MinLstmState<B>>) {
        let [b, s, _] = x.dims();
        let device = x.device();
        let hidden_size = self.linear_f.weight.dims()[1];

        let mut states = states.unwrap_or_default();
        let h_prev = states.pop().map(|s| s.hidden);
        let h_0 = h_prev.unwrap_or_else(|| Tensor::zeros([b, 1, hidden_size], &device));

        // diff = softplus(-W_f(x)) - softplus(-W_i(x))
        let diff = activation::softplus(self.linear_f.forward(x.clone()).neg(), 1.0) 
            - activation::softplus(self.linear_i.forward(x.clone()).neg(), 1.0);
            
        // log_f = -softplus(diff)
        let log_f = activation::softplus(diff.clone(), 1.0).neg();
        // log_i = -softplus(-diff)
        let log_i = activation::softplus(diff.neg(), 1.0).neg();
        
        let log_h_0 = log_g(h_0);
        let log_tilde_h = log_g(self.linear_h.forward(x));
        
        let log_values = Tensor::cat(vec![log_h_0, log_i + log_tilde_h], 1);
        let h = parallel_scan_log(log_f, log_values);

        let last_h = h.clone().slice([0..b, s-1..s, 0..hidden_size]);
        (self.output_projection.forward(h), vec![MinLstmState::new(last_h)])
    }

    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let diff = activation::softplus(self.linear_f.forward(x_t.clone()).neg(), 1.0) 
            - activation::softplus(self.linear_i.forward(x_t.clone()).neg(), 1.0);
            
        let log_f = activation::softplus(diff.clone(), 1.0).neg();
        let log_i = activation::softplus(diff.neg(), 1.0).neg();
        
        let log_tilde_h = log_g(self.linear_h.forward(x_t));
        
        // h_t = exp(log_f) * h_prev + exp(log_i + log_tilde_h)
        let h_t = (log_f.exp() * h_prev) + (log_i + log_tilde_h).exp();

        (self.output_projection.forward(h_t.clone()), h_t)
    }
}
