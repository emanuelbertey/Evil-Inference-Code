use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    conv1d, embedding, linear, linear_no_bias, loss::cross_entropy, ops, rms_norm, Conv1d,
    Conv1dConfig, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

fn heinsen_associative_scan_log(log_coeffs: &Tensor, log_values: &Tensor) -> Result<Tensor> {
    let a_star = log_coeffs.cumsum(1)?;
    // logcumsumexp, with clamping for numerical stability
    let log_h0_plus_b_star = log_values
        .sub(&a_star)?
        .exp()?
        .clamp(1e-6, 1e6)?
        .cumsum(1)?
        .log()?
        .clamp(-1e6, 1e6)?;
    let log_h = a_star.add(&log_h0_plus_b_star)?;
    log_h.exp()
}

fn g(x: &Tensor) -> Result<Tensor> {
    x.lt(0_u32)?
        .where_cond(&ops::sigmoid(x)?, &x.affine(0., 0.5)?)
}

fn softplus(x: &Tensor, beta: f64, threshold: f32) -> Result<Tensor> {
    let mask = x.affine(beta, 0.)?.gt(threshold)?;
    let res = x.affine(beta, 0.)?.exp()?.affine(1. / beta, 0.)?;
    mask.where_cond(x, &res)
}

fn log_g(x: &Tensor) -> Result<Tensor> {
    x.lt(0_f32)?.where_cond(
        &softplus(&x.affine(-1., 0.)?, 1.0, 20.0)?.affine(-1., 0.)?,
        &x.relu()?.affine(0., 0.5)?.log()?,
    )
}

fn lerp(start: &Tensor, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
    start.add(&end.sub(start)?.mul(weight)?)
}

/// MinGRU log-space as in appendix B.3.1
struct MinGRU {
    to_hidden_and_gate: Linear,
    to_out: Linear,
    exp_factor: f32,
}

impl MinGRU {
    fn new(dim: usize, exp_factor: f32, vb: VarBuilder) -> Result<Self> {
        let dim_inner = (dim as f32 * exp_factor).round() as usize;
        // first linear layer
        let to_hidden_and_gate = linear(dim, dim_inner * 2, vb.pp("hidden_n_gate"))?;
        // second linear layer
        let to_out = linear_no_bias(dim_inner, dim, vb.pp("to_out"))?;

        Ok(MinGRU {
            to_hidden_and_gate,
            to_out,
            exp_factor,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        prev_hidden: Option<&Tensor>,
        return_next_prev_hidden: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let dims = x.dims();
        let seq_len = dims[1];
        let hid_n_gate = self.to_hidden_and_gate.forward(x)?.chunk(2, D::Minus1)?;

        let mut out: Tensor;

        match seq_len {
            1 => {
                let hidden = g(&hid_n_gate[0])?;
                let gate = ops::sigmoid(&hid_n_gate[1])?;
                match prev_hidden {
                    Some(prev_h) => {
                        out = lerp(prev_h, &hidden, &gate)?;
                    }
                    None => {
                        out = hidden.mul(&gate)?;
                    }
                }
            }
            sq_len => {
                let mut log_coeffs = (-1.0 * softplus(&hid_n_gate[1], 1.0, 20.0)?)?;
                let log_z = (-1.0 * softplus(&(-1.0 * &hid_n_gate[1])?, 1.0, 20.0)?)?;
                let log_tilde_h = log_g(&hid_n_gate[0])?;
                let mut log_values = log_z.add(&log_tilde_h)?;

                if prev_hidden.is_some() {
                    log_values = Tensor::cat(&[prev_hidden.unwrap().log()?, log_values], 1)?;
                    log_coeffs = log_coeffs.pad_with_zeros(D::Minus1, 1, 0)?;
                }
                let hscan_out = heinsen_associative_scan_log(&log_coeffs, &log_values)?;
                let dim1_size = hscan_out.dims()[1];
                out = hscan_out.i((.., dim1_size - sq_len..))?;
            }
        }

        let nxt_prv_hid = out.i((.., out.dims()[1] - 1..))?;

        if self.exp_factor != 1.0 {
            out = self.to_out.forward(&out)?;
        }

        let next_prev_hidden = match return_next_prev_hidden {
            false => None,
            true => Some(nxt_prv_hid),
        };

        Ok((out, next_prev_hidden))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn new(dim: usize, mult: usize, vs: VarBuilder) -> Result<Self> {
        let dim_inner = dim * mult;
        let linear1 = linear(dim, dim_inner, vs.pp("linear1"))?;
        let linear2 = linear(dim_inner, dim, vs.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.linear1.forward(x)?.gelu()?;
        self.linear2.forward(&out)
    }
}

struct CausalDepthWiseConv1d {
    kernel_size: usize,
    conv1: Conv1d,
    conv2: Conv1d,
}

impl CausalDepthWiseConv1d {
    fn new(dim: usize, kernel_size: usize, vs: &mut VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig::default();
        let conv1 = conv1d(dim, dim, kernel_size, conv_cfg, vs.pp("conv1"))?;
        let conv2 = conv1d(dim, dim, 1, conv_cfg, vs.pp("conv2"))?;
        Ok(Self {
            kernel_size,
            conv1,
            conv2,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.t()?.pad_with_zeros(D::Minus1, self.kernel_size - 1, 0)?;
        let out = self.conv1.forward(&out)?;
        self.conv2.forward(&out)?.t()
    }
}

struct MinGRUBlock {
    conv: Option<CausalDepthWiseConv1d>,
    norm1: RmsNorm,
    min_gru: MinGRU,
    norm2: RmsNorm,
    ffn: FeedForward,
}
impl MinGRUBlock {
    fn new(
        dim: usize,
        conv_kernel_size: Option<usize>,
        exp_factor: f32,
        ff_mult: usize,
        rms_eps: f64,
        blk_num: usize,
        vs: &mut VarBuilder,
    ) -> Result<Self> {
        let conv = match conv_kernel_size {
            None => None,
            Some(kern_size) => Some(CausalDepthWiseConv1d::new(dim, kern_size, vs)?),
        };
        let norm1 = rms_norm(dim, rms_eps, vs.pp(format!("block{}.norm1", blk_num)))?;
        let norm2 = rms_norm(dim, rms_eps, vs.pp(format!("block{}.norm2", blk_num)))?;
        let min_gru = MinGRU::new(dim, exp_factor, vs.pp(format!("block{}.mingru", blk_num)))?;
        let ffn = FeedForward::new(dim, ff_mult, vs.pp(format!("block{}.ffn", blk_num)))?;
        Ok(Self {
            conv,
            norm1,
            min_gru,
            norm2,
            ffn,
        })
    }
}

pub struct MinGRUConfig {
    num_tokens: usize,
    dim: usize,
    depth: usize,
    ff_mult: usize,
    exp_factor: f32,
    conv_kernel_size: Option<usize>,
}

impl Default for MinGRUConfig {
    fn default() -> Self {
        Self {
            num_tokens: 256,
            dim: 512,
            depth: 6,
            ff_mult: 4,
            exp_factor: 1.5,
            conv_kernel_size: Some(3),
        }
    }
}

impl MinGRUConfig {
    pub fn set_num_tokens(&mut self, num_tokens: usize) {
        self.num_tokens = num_tokens
    }
}

pub struct MinGRULM {
    token_emb: Embedding,
    layers: Vec<MinGRUBlock>,
    norm: RmsNorm,
    to_logits: Linear,
    can_cache: bool,
    num_tokens: usize,
}

impl MinGRULM {
    pub fn new(config: MinGRUConfig, mut vb: VarBuilder) -> Result<Self> {
        let token_emb = embedding(config.num_tokens, config.dim, vb.clone())?;
        let mut layers: Vec<MinGRUBlock> = Vec::new();
        for i in 0..config.depth {
            layers.push(MinGRUBlock::new(
                config.dim,
                config.conv_kernel_size,
                config.exp_factor,
                config.ff_mult,
                0.001,
                i,
                &mut vb,
            )?);
        }
        let norm = rms_norm(config.dim, 1e-6, vb.pp("norm"))?;
        let to_logits = linear_no_bias(config.dim, config.num_tokens, vb.pp("to_logits"))?;
        let can_cache = config.conv_kernel_size.is_none();

        Ok(Self {
            token_emb,
            layers,
            norm,
            to_logits,
            can_cache,
            num_tokens: config.num_tokens,
        })
    }
    pub fn can_cache(&self) -> bool {
        self.can_cache
    }

    pub fn forward(
        &self,
        x: &Tensor,
        return_loss: bool,
        prev_hiddens: Option<Vec<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let x_dim = x.dims();
        let mut out = x.clone();

        let mut labels = Tensor::zeros((x_dim[0], self.num_tokens), DType::U32, x.device())?;

        if return_loss {
            out = x.i((.., ..x_dim[1] - 1))?;
            labels = x.i((.., 1..))?;
        }

        out = self.token_emb.forward(&out)?;

        if prev_hiddens.is_some() {
            let out_dims = out.dims();
            out = out.i((.., out_dims[1] - 1..))?;
        }

        let prev_hid = prev_hiddens.unwrap_or_default();
        let mut prev_hiddens_iter = prev_hid.iter();
        let mut next_prev_hiddens: Vec<Tensor> = Vec::new();

        for gru_block in self.layers.iter() {
            // conv
            if gru_block.conv.is_some() {
                out = gru_block.conv.as_ref().unwrap().forward(&out)?.add(&out)?;
            }

            // minGRU
            let prev_hidden = prev_hiddens_iter.next();
            let norm_out = gru_block.norm1.forward(&out)?;
            let (min_gru_out, next_prev_hidden) =
                gru_block.min_gru.forward(&norm_out, prev_hidden, true)?;
            out = min_gru_out.add(&out)?;
            next_prev_hiddens.push(next_prev_hidden.unwrap());

            // ffn
            out = gru_block
                .ffn
                .forward(&gru_block.norm2.forward(&out)?)?
                .add(&out)?;
        }

        let embed = self.norm.forward(&out)?;
        let logits = self.to_logits.forward(&embed)?;

        if !return_loss {
            Ok((logits, next_prev_hiddens))
        } else {
            let logits_flat = logits.reshape(((), logits.dim(D::Minus1)?))?;
            let labels_flat = labels.reshape(((),))?;
            let loss = cross_entropy(&logits_flat, &labels_flat)?;
            Ok((loss, next_prev_hiddens))
        }
    }
}
