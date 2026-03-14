/*!
# sLSTM: Scalar Long Short-Term Memory
Implementation according to: "xLSTM: Extended Long Short-Term Memory" (Beck et al. 2024)

This implementation is aligned with high-performance standards, using 
pre-computed input projections and log-space stabilization for numerical safety.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer},
    tensor::{activation, backend::Backend, Tensor},
};

/// State for sLSTM containing cell, normalizer, hidden, and stabilizer states
#[derive(Clone, Debug)]
pub struct SLstmstate<B: Backend, const D: usize> {
    pub cell: Tensor<B, D>,
    pub normalizer: Tensor<B, D>,
    pub hidden: Tensor<B, D>,
    pub max_gate_log: Tensor<B, D>, // stabilizer (m_t)
}

impl<B: Backend, const D: usize> SLstmstate<B, D> {
    pub fn new(
        cell: Tensor<B, D>,
        hidden: Tensor<B, D>,
        normalizer: Tensor<B, D>,
        max_gate_log: Tensor<B, D>,
    ) -> Self {
        Self {
            cell,
            hidden,
            normalizer,
            max_gate_log,
        }
    }

    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
        }
    }
}

/// Configuration for sLSTM (Aligned with Candle/Paper stability standards)
#[derive(Config, Debug)]
pub struct SLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = "0.02")]
    pub weight_stdev: f64,
    #[config(default = "1.0")] // Forget gate neutral (log(1.0)=0)
    pub forget_bias: f32,
    #[config(default = "0.1")]
    pub input_bias: f32,
    #[config(default = "1e-6")]
    pub epsilon: f32,
    #[config(default = "-30.0")]
    pub exp_clamp_min: f32,
    #[config(default = "30.0")]
    pub exp_clamp_max: f32,
    #[config(default = "0.0")]
    pub stabilizer_init: f32,
    #[config(default = "Initializer::Normal{mean: 0.0, std: 0.02}")]
    pub initializer: Initializer,
}

impl SLstmconfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SLstm<B> {
        let mut layers = alloc::vec::Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            layers.push(SLstmcell::new(input_size, self.d_hidden, self, device));
        }

        SLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
            stabilizer_init: self.stabilizer_init,
        }
    }
}

#[derive(Module, Debug)]
pub struct SLstm<B: Backend> {
    pub layers: alloc::vec::Vec<SLstmcell<B>>,
    pub dropout_layer: Dropout,
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub stabilizer_init: f32,
}

impl<B: Backend> SLstm<B> {
    pub fn forward(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<SLstmstate<B, 2>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<SLstmstate<B, 2>>) {
        self.forward_ext(input_seq, states, false)
    }

    pub fn forward_ext(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<SLstmstate<B, 2>>>,
        frozen: bool,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<SLstmstate<B, 2>>) {
        let device = input_seq.device();
        let [batch_size, seq_length, _] = input_seq.dims();

        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));
        let mut current_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_outputs = alloc::vec::Vec::with_capacity(seq_length);
            let mut state = hidden_states[layer_idx].clone();

            // Precompute input projections: [B, S, 4*H]
            let projected_input = current_input.clone().matmul(
                layer.weight_ih.val().transpose().unsqueeze_dim::<3>(0)
            ) + layer.bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);

            for t in 0..seq_length {
                let input_t_projected = projected_input.clone()
                    .slice([0..batch_size, t..(t + 1), 0..(4 * layer.hidden_size)])
                    .squeeze(1);

                let (h_new, new_state) = layer.forward_step(input_t_projected, state, frozen);
                state = new_state;
                layer_outputs.push(h_new.unsqueeze_dim(1));
            }

            hidden_states[layer_idx] = state;
            let layer_output_seq = Tensor::cat(layer_outputs, 1);

            current_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(layer_output_seq)
            } else {
                layer_output_seq
            };
        }

        (current_input, hidden_states)
    }

    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<SLstmstate<B, 2>> {
        (0..self.num_layers)
            .map(|_| {
                SLstmstate::new(
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device).add_scalar(self.stabilizer_init),
                )
            })
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct SLstmcell<B: Backend> {
    pub weight_ih: Param<Tensor<B, 2>>,
    pub weight_hh: Param<Tensor<B, 2>>,
    pub bias: Param<Tensor<B, 1>>,
    pub hidden_size: usize,
    pub epsilon: f32,
    pub exp_clamp_min: f32,
    pub exp_clamp_max: f32,
}

impl<B: Backend> SLstmcell<B> {
    pub fn new(input_size: usize, hidden_size: usize, config: &SLstmconfig, device: &B::Device) -> Self {
        let weight_ih = config.initializer.init_with(
            [4 * hidden_size, input_size],
            Some(input_size),
            Some(4 * hidden_size),
            device,
        );
        let weight_hh = config.initializer.init_with(
            [4 * hidden_size, hidden_size],
            Some(hidden_size),
            Some(4 * hidden_size),
            device,
        );

        // Biased initialization for Forget and Input gates
        let mut bias_data = alloc::vec![0.0f32; 4 * hidden_size];
        for i in 0..hidden_size {
            bias_data[i] = config.input_bias; // Input gate bias
            bias_data[i + hidden_size] = config.forget_bias; // Forget gate bias
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        Self {
            weight_ih,
            weight_hh,
            bias: Param::from_tensor(bias),
            hidden_size,
            epsilon: config.epsilon,
            exp_clamp_min: config.exp_clamp_min,
            exp_clamp_max: config.exp_clamp_max,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>, state: SLstmstate<B, 2>) -> (Tensor<B, 2>, SLstmstate<B, 2>) {
        let projected = input.matmul(self.weight_ih.val().transpose()) + self.bias.val().unsqueeze_dim(0);
        self.forward_step(projected, state, false)
    }

    pub fn forward_step(
        &self, 
        input_projected: Tensor<B, 2>, 
        state: SLstmstate<B, 2>,
        frozen: bool,
    ) -> (Tensor<B, 2>, SLstmstate<B, 2>) {
        let cell = state.cell.clone();
        let hidden = state.hidden.clone();
        let normalizer = state.normalizer.clone();
        let max_gate_log = state.max_gate_log.clone();

        let gates = input_projected + hidden.matmul(self.weight_hh.val().transpose());
        let chunks = gates.chunk(4, 1);
        
        let i_log = chunks[0].clone();
        let f_log = chunks[1].clone();
        let z = chunks[2].clone().tanh();
        let o = activation::sigmoid(chunks[3].clone());

        let m_prev_plus_f = f_log + max_gate_log.clone();
        let m_new = if frozen { max_gate_log.clone() } else { m_prev_plus_f.clone().max_pair(i_log.clone()) };

        let i_exp = (i_log - m_new.clone()).exp();
        let f_exp = (m_prev_plus_f - m_new.clone()).exp();

        let c_new = if frozen { cell.clone() } else { f_exp.clone() * cell.clone() + i_exp.clone() * z };
        let n_new = if frozen { normalizer.clone() } else { f_exp * normalizer.clone() + i_exp };

        // Output normalization
        let n_stable = n_new.clone().max_pair(n_new.full_like(self.epsilon));
        let h_new = o * (c_new.clone() / n_stable);

        if frozen {
            (h_new, state)
        } else {
            (h_new.clone(), SLstmstate::new(c_new, h_new, n_new, m_new))
        }
    }
}