use crate::data::Dataset;
use crate::model::{MinGRUConfig, MinGRULM};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use candle_transformers::generation::{LogitsProcessor, Sampling};

pub struct TrainConfig {
    num_batches: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    validate_every: usize,
    prime_length: usize,
    generate_every: usize,
    generate_length: usize,
    seq_len: usize,
    temperature: f32,
    threshold: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_batches: 1e5 as usize,
            batch_size: 4,
            grad_accum: 4,
            learning_rate: 1e-4,
            validate_every: 100,
            prime_length: 256,
            generate_every: 500,
            generate_length: 512,
            seq_len: 512,
            temperature: 1.0,
            threshold: Some(0.9),
        }
    }
}

impl TrainConfig {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

fn base_decoding(
    model: &MinGRULM,
    prompt: &Tensor,
    seq_len: usize,
    logits_processor: &mut LogitsProcessor,
) -> anyhow::Result<Vec<u32>> {
    let mut tokens: Vec<u32> = Vec::new();
    let prompt_seq_len = prompt.dim(D::Minus1)?;
    let mut out = prompt.clone();
    let sample_n_times = (seq_len - prompt_seq_len).max(0);
    let mut prev_hiddens: Option<Vec<Tensor>> = None;
    for _ in 0..sample_n_times {
        let res = model.forward(&out, false, prev_hiddens)?;
        let dim1_size = res.0.dim(1)?;
        let logits = res.0.narrow(1, dim1_size - 1, 1)?;
        if model.can_cache() {
            prev_hiddens = Some(res.1);
        } else {
            prev_hiddens = None;
        }
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let next_token = logits_processor.sample(&logits)?;
        let tok_tensor = Tensor::from_slice(&[next_token], (1, 1), prompt.device())?;
        out = Tensor::cat(&[&out, &tok_tensor], 1)?;
        tokens.push(next_token);
    }
    Ok(tokens)
}

fn batch_encoding_to_tensor(encoding: &[Vec<u32>]) -> Result<Tensor> {
    let ids = encoding
        .iter()
        .map(|enc| Tensor::from_slice(enc, (enc.len(),), &Device::Cpu).unwrap())
        .collect::<Vec<_>>();
    Tensor::stack(&ids, 0)
}

pub fn training_loop(m: &mut Dataset, cfg: &TrainConfig) -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let mut model_cfg = MinGRUConfig::default();
    model_cfg.set_num_tokens(m.vocab_size);
    let model = MinGRULM::new(model_cfg, vs.clone())?;

    let mut optim = AdamW::new_lr(varmap.all_vars(), cfg.learning_rate)?;

    let mut logit_proc = LogitsProcessor::from_sampling(
        42,
        Sampling::TopK {
            k: (m.vocab_size as f32 * (1. - cfg.threshold.unwrap_or(0.9))) as usize,
            temperature: cfg.temperature as f64,
        },
    );

    let mut acc_loss = Tensor::zeros((), DType::F32, &dev)?;
    for i in 0..cfg.num_batches {
        let batch = m.get_batch("train", cfg.batch_size)?;
        let batch_input = batch_encoding_to_tensor(&batch)?.to_device(&dev)?;
        let (loss, _) = model.forward(&batch_input, true, None)?;
        acc_loss = acc_loss.add(&loss)?;

        if i % cfg.grad_accum == 0 {
            println!(
                "Training loss: {:?}",
                acc_loss
                    .affine(1. / cfg.grad_accum as f64, 0.)?
                    .to_scalar::<f32>()?
            );
            optim.backward_step(&acc_loss)?;
            // reset loss for accumulate grad batch
            acc_loss = Tensor::zeros((), DType::F32, &dev)?;
        }

        // one validation step
        if i % cfg.validate_every == 0 {
            let val_batch = m.get_batch("val", cfg.batch_size)?;
            let val_input = batch_encoding_to_tensor(&val_batch)?.to_device(&dev)?;
            let (val_loss, _) = model.forward(&val_input, true, None)?;
            println!("Validation loss: {}", val_loss.to_vec0::<f32>()?);
        }

        // generate some random sentence from validation
        if i % cfg.generate_every == 0 {
            let test_batch = m.get_batch("val", 1)?;
            let test_input = batch_encoding_to_tensor(&test_batch)?
                .to_device(&dev)?
                .narrow(D::Minus1, 0, cfg.prime_length)?;
            let txt_input: Vec<u32> = test_input.i((0, ..))?.to_vec1()?;
            let prime = m.tokenizer.decode(&txt_input, false).unwrap();
            println!("INPUT: {}", prime);

            let sampled = base_decoding(&model, &test_input, cfg.generate_length, &mut logit_proc)?;
            let decoded_output = m.tokenizer.decode(&sampled, false).unwrap();
            println!("OUTPUT: {}", decoded_output);
        }
    }

    Ok(())
}
