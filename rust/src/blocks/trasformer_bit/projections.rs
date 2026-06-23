// ─── BitLinear Projections ─────────────────────────────────────────────────

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use crate::blocks::bitlinear::layer::{BitLinear, BitLinearInferenceState};

// ─── BitLinear QKV Projection ──────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearQKVProjection<B: Backend> {
    pub q_proj: BitLinear<B>,
    pub k_proj: BitLinear<B>,
    pub v_proj: BitLinear<B>,
    pub num_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
}

impl<B: Backend> BitLinearQKVProjection<B> {
    pub fn release_weights(&mut self, device: &B::Device) {
        self.q_proj.release_weights(device);
        self.k_proj.release_weights(device);
        self.v_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();
        let q = self.q_proj.forward(x.clone()).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward(x).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        (q, k, v)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, q_state: &BitLinearInferenceState, k_state: &BitLinearInferenceState, v_state: &BitLinearInferenceState) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d] = x.dims();
        let q = self.q_proj.forward_inference(x.clone(), q_state).reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward_inference(x.clone(), k_state).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        let v = self.v_proj.forward_inference(x, v_state).reshape([batch, seq_len, self.num_kv_groups, self.head_dim]);
        (q, k, v)
    }
}

// ─── BitLinear Output Projection ────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearOutputProjection<B: Backend> {
    pub o_proj: BitLinear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> BitLinearOutputProjection<B> {
    pub fn release_weights(&mut self, device: &B::Device) {
        self.o_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        self.o_proj.forward(x.reshape([batch, seq_len, self.num_heads * self.head_dim]))
    }

    pub fn forward_inference(&self, x: Tensor<B, 4>, state: &BitLinearInferenceState) -> Tensor<B, 3> {
        let [batch, seq_len, _nh, _hd] = x.dims();
        self.o_proj.forward_inference(x.reshape([batch, seq_len, self.num_heads * self.head_dim]), state)
    }
}

// ─── BitLinear SwiGLU FeedForward ───────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearSwiGLUFeedForward<B: Backend> {
    pub gate_up_proj: BitLinear<B>,
    pub down_proj: BitLinear<B>,
    pub dropout: burn::nn::Dropout,
    pub intermediate_dim: usize,
}

impl<B: Backend> BitLinearSwiGLUFeedForward<B> {
    pub fn release_weights(&mut self, device: &B::Device) {
        self.gate_up_proj.release_weights(device);
        self.down_proj.release_weights(device);
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward(x);
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        let h = self.dropout.forward(h);
        self.down_proj.forward(h)
    }

    pub fn forward_inference(&self, x: Tensor<B, 3>, gate_up_state: &BitLinearInferenceState, down_state: &BitLinearInferenceState) -> Tensor<B, 3> {
        let gate_up = self.gate_up_proj.forward_inference(x, gate_up_state);
        let chunks = gate_up.chunk(2, 2);
        let gate = chunks[0].clone();
        let up = chunks[1].clone();
        let h = burn::tensor::activation::silu(gate) * up;
        self.down_proj.forward_inference(h, down_state)
    }
}

// ─── BitLinear RMSNorm ─────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct BitLinearRMSNorm<B: Backend> {
    pub weight: burn::module::Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> BitLinearRMSNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: burn::module::Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let denom = (x.clone().powf_scalar(2.0).mean_dim(2) + self.eps as f32).sqrt();
        let normed = x / denom;
        normed * self.weight.val().unsqueeze::<2>().unsqueeze::<3>()
    }
}
