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
        
        // PAPEL: MinLSTM usa capas lineales CON BIAS. 
        // El bias es crucial para que el gate se estabilice sin "rebotes".
        let l_f = nn::LinearConfig::new(self.input_features, hidden_size).init(device);
        let l_i = nn::LinearConfig::new(self.input_features, hidden_size).init(device);
        let l_h = nn::LinearConfig::new(self.input_features, hidden_size).init(device);
        let proj = nn::LinearConfig::new(hidden_size, self.input_features).init(device);
        
        // Inicialización uniforme estándar
        let init_params = |linear: nn::Linear<B>, in_dim: usize| {
            let k = (1.0 / in_dim as f32).sqrt();
            let [in_d, out_d] = linear.weight.dims();
            linear.load_record(nn::LinearRecord {
                weight: Param::from_tensor(Tensor::random([in_d, out_d], Distribution::Uniform(-k as f64, k as f64), device)),
                bias: Some(Param::from_tensor(Tensor::random([out_d], Distribution::Uniform(-k as f64, k as f64), device))),
            })
        };

        MinLstm { 
            linear_f: init_params(l_f, self.input_features),
            linear_i: init_params(l_i, self.input_features),
            linear_h: init_params(l_h, self.input_features),
            output_projection: init_params(proj, hidden_size) 
        }
    }
}

// Función auxiliar para emular torch.logcumsumexp de forma estable y paralela en f32
fn log_cumsum_exp<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let max = x.clone().detach().max_dim(1);
    let min = x.clone().detach().neg().max_dim(1).neg(); 
    let m = (max + min) / 2.0;
    
    // El clamp evita que los valores se salgan del rango de f32 durante el scan
    (x - m.clone()).clamp(-85.0, 85.0).exp().cumsum(1).log() + m
}

fn log_g<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()) + 0.5).log();
    let neg = activation::softplus(x.neg(), 1.0).neg(); 
    neg.mask_where(mask, pos)
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s_plus_1, h] = log_values.dims();
    let device = log_values.device();
    
    let a_star = Tensor::cat(vec![
        Tensor::zeros([b, 1, h], &device),
        log_coeffs.cumsum(1)
    ], 1);
    
    let x = log_values - a_star.clone();
    let log_h0_plus_b_star = log_cumsum_exp(x);
    
    let log_h = a_star + log_h0_plus_b_star;
    
    log_h.exp().slice([0..b, 1..s_plus_1, 0..h])
}

impl<B: Backend> MinLstm<B> {
    pub fn forward(&self, x: Tensor<B, 3>, states: Option<Vec<MinLstmState<B>>>) -> (Tensor<B, 3>, Vec<MinLstmState<B>>) {
        let [b, s, _] = x.dims();
        let device = x.device();
        let hidden_size = self.linear_f.weight.dims()[1];

        let mut states = states.unwrap_or_default();
        let h_prev = states.pop().map(|s| s.hidden);
        let h_0 = h_prev.unwrap_or_else(|| Tensor::zeros([b, 1, hidden_size], &device));

        // PAPEL: diff = log(sigm(f)) - log(sigm(i)) en terminos de log-space
        // log(sigm(x)) = -softplus(-x)
        let log_f_raw = activation::softplus(self.linear_f.forward(x.clone()).neg(), 1.0).neg();
        let log_i_raw = activation::softplus(self.linear_i.forward(x.clone()).neg(), 1.0).neg();
        
        let diff = log_i_raw - log_f_raw;
            
        // Combinación convexa pura: exp(log_f) + exp(log_i) = 1
        let log_f = activation::softplus(diff.clone(), 1.0).neg();
        let log_i = activation::softplus(diff.neg(), 1.0).neg();
        
        let log_h_0 = log_g(h_0);
        let log_tilde_h = log_g(self.linear_h.forward(x));
        
        let log_values = Tensor::cat(vec![log_h_0, log_i + log_tilde_h], 1);
        let h = parallel_scan_log(log_f, log_values);

        let last_h = h.clone().slice([0..b, s-1..s, 0..hidden_size]);
        (self.output_projection.forward(h), vec![MinLstmState::new(last_h)])
    }

    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let f_t = activation::sigmoid(self.linear_f.forward(x_t.clone()));
        let i_t = activation::sigmoid(self.linear_i.forward(x_t.clone()));
        
        // g(x) = donde x>=0 ? x+0.5 : sigmoid(x)
        let tilde_h_t = {
            let x = self.linear_h.forward(x_t);
            let mask = x.clone().greater_equal_elem(0.0);
            let pos = x.clone() + 0.5;
            let neg = activation::sigmoid(x);
            neg.mask_where(mask, pos)
        };

        // f' = f / (f+i), i' = i / (f+i)
        let sum = f_t.clone() + i_t.clone();
        let f_prime = f_t / sum.clone();
        let i_prime = i_t / sum;

        let h_t = (f_prime * h_prev) + (i_prime * tilde_h_t);
;
        (self.output_projection.forward(h_t.clone()), h_t)
    }
}
