use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::Distribution;
use burn::module::Param;

#[derive(Config, Debug)]
pub struct MinGruConfig {
    pub input_features: usize,
    #[config(default = 2)]
    pub expansion_factor: usize,
}

#[derive(Module, Debug)]
pub struct MinGru<B: Backend> {
    pub linear_z: nn::Linear<B>,
    pub linear_h: nn::Linear<B>,
    pub output_projection: nn::Linear<B>,
}

#[derive(Clone, Debug)]
pub struct MinGruState<B: Backend> {
    pub hidden: Tensor<B, 3>,
}

impl<B: Backend> MinGruState<B> {
    pub fn new(hidden: Tensor<B, 3>) -> Self {
        Self { hidden }
    }
}

impl MinGruConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGru<B> {
        let hidden_size = self.input_features * self.expansion_factor;
        let std_dev = (1.0 / self.input_features as f32).sqrt();
        
        // Inicialización manual para estabilidad
        let l_z = nn::LinearConfig::new(self.input_features, hidden_size).with_bias(true).init(device);
        let l_h = nn::LinearConfig::new(self.input_features, hidden_size).with_bias(false).init(device);
        let proj = nn::LinearConfig::new(hidden_size, self.input_features).with_bias(false).init(device);
        
        MinGru { 
            linear_z: l_z.load_record(nn::LinearRecord {
                weight: Param::from_tensor(Tensor::random([self.input_features, hidden_size], Distribution::Normal(0.0, std_dev as f64), device)),
                // Bias de -3.0 para que (1-z) sea ~0.95, permitiendo flujo de gradiente en secuencias largas
                bias: Some(Param::from_tensor(Tensor::zeros([hidden_size], device).add_scalar(-3.0))),
            }),
            linear_h: l_h,
            output_projection: proj 
        }
    }
}

fn log_g<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = x.clone().greater_equal_elem(0.0);
    let pos = (activation::relu(x.clone()) + 0.5).log();
    // Paper: para x < 0, log(sigmoid(x)) = -softplus(-x)
    let neg = activation::softplus(x.neg(), 1.0).neg(); 
    neg.mask_where(mask, pos)
}

fn parallel_scan_log<B: Backend>(log_coeffs: Tensor<B, 3>, log_values: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, _s, h] = log_coeffs.dims();
    let device = log_coeffs.device();
    let a_star = Tensor::cat(vec![
        Tensor::zeros([b, 1, h], &device),
        log_coeffs.cumsum(1)
    ], 1);
    
    let x = log_values - a_star.clone();
    let x_max = x.clone().max_dim(1);
    let log_h0_plus_b_star = (x - x_max.clone()).exp().cumsum(1).clamp_min(1e-10).log() + x_max;
    
    let log_h = a_star + log_h0_plus_b_star;
    let dims = log_h.dims();
    // Clamp de 20.0 es el "punto dulce" para gradientes estables en f32
    log_h.clamp(-20.0, 20.0).exp().slice([0..b, 1..dims[1], 0..h])
}

impl<B: Backend> MinGru<B> {
    pub fn forward(&self, x: Tensor<B, 3>, states: Option<Vec<MinGruState<B>>>) -> (Tensor<B, 3>, Vec<MinGruState<B>>) {
        let [b, s, _] = x.dims();
        let device = x.device();
        let hidden_size = self.linear_z.weight.dims()[1];

        let mut states = states.unwrap_or_default();
        let h_prev = states.pop().map(|s| s.hidden);
        let h_0 = h_prev.unwrap_or_else(|| Tensor::zeros([b, 1, hidden_size], &device));

        let k = self.linear_z.forward(x.clone());
        // let log_z = activation::softplus(k.clone().neg(), 1.0).neg();
        let log_z = activation::softplus(k.clone().neg(), 1.0).neg().clamp_min(-35.0);
        // let log_coeffs = activation::softplus(k, 1.0).neg();
        let log_coeffs = activation::softplus(k, 1.0).neg().clamp_min(-35.0);
        
        let log_h_0 = log_g(h_0);
        let log_tilde_h = log_g(self.linear_h.forward(x));
        
        let log_values = Tensor::cat(vec![log_h_0, log_z + log_tilde_h], 1);
        let h = parallel_scan_log(log_coeffs, log_values);

        let last_h = h.clone().slice([0..b, s-1..s, 0..hidden_size]);
        (self.output_projection.forward(h), vec![MinGruState::new(last_h)])
    }

    pub fn sequential_mode(&self, x_t: Tensor<B, 3>, h_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let k = self.linear_z.forward(x_t.clone());
        
        // 1. Gating idéntico al espacio logarítmico del forward
        let log_coeffs = activation::softplus(k.clone(), 1.0).neg().clamp_min(-35.0);
        let log_z = activation::softplus(k.neg(), 1.0).neg().clamp_min(-35.0);
        
        // 2. Activación de entrada idéntica
        let log_tilde_h = log_g(self.linear_h.forward(x_t));
        
        // 3. Recurrencia: h_t = (1-z)*h_prev + z*tilde_h
        // Usamos exp() para aplicar los valores que el scan calculó en log-space
        let h_t = (log_coeffs.exp() * h_prev) + (log_z + log_tilde_h).exp();

        (self.output_projection.forward(h_t.clone()), h_t)
    }
}
