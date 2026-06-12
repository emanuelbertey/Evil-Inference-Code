use anyhow::Error as E;
use flate2::bufread::GzDecoder;
use hf_hub::api::sync::Api;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use tokenizers::tokenizer::{Encoding, Tokenizer};

fn extract_data(filepath: &str) -> String {
    let file = File::open(filepath).unwrap();
    let bufreader = BufReader::new(file);
    let mut gz = GzDecoder::new(bufreader);
    let mut text = String::new();
    gz.read_to_string(&mut text).unwrap();
    text
}

fn get_tokenizer(tokenizer_file: Option<&str>) -> Result<Tokenizer, E> {
    let api = Api::new()?;
    let tokenizer_filename = match tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => api.model("Xenova/gpt2".to_string()).get("tokenizer.json")?,
    };
    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

pub struct Dataset {
    pub train_data: Encoding,
    pub val_data: Encoding,
    pub tokenizer: Tokenizer,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rng: ThreadRng,
}

impl Dataset {
    pub fn new(filepath: &str, tokenizer_file: Option<&str>, seq_len: usize) -> Self {
        let data = extract_data(filepath);
        let split_point = data.len() as f32 * 0.85;
        let (train_data, val_data) = data.split_at(split_point as usize);
        let tokenizer = get_tokenizer(tokenizer_file).unwrap();
        let vocab_size = tokenizer.get_vocab_size(false);
        let train_tokens = tokenizer.encode_fast(train_data, false).unwrap();
        let val_tokens = tokenizer.encode_fast(val_data, false).unwrap();
        let rng = rand::thread_rng();
        Self {
            train_data: train_tokens,
            val_data: val_tokens,
            tokenizer,
            vocab_size,
            seq_len,
            rng,
        }
    }

    pub fn get_batch(&mut self, mode: &str, batch_size: usize) -> Result<Vec<Vec<u32>>, E> {
        let mut batch: Vec<Vec<u32>> = Vec::new();
        for _ in 0..batch_size {
            let sample = if mode == "train" {
                self.get_train_sample()
            } else {
                self.get_val_sample()
            };
            batch.push(sample);
        }
        Ok(batch)
    }

    fn get_train_sample(&mut self) -> Vec<u32> {
        let max_idx = self.train_data.len() - self.seq_len;
        let rnd_start = self.rng.gen_range(0..max_idx);
        let end = rnd_start + self.seq_len;
        let input = &self.train_data.get_ids()[rnd_start..end];
        input.to_vec()
    }

    fn get_val_sample(&mut self) -> Vec<u32> {
        let max_idx = self.val_data.len() - self.seq_len;
        let rnd_start = self.rng.gen_range(0..max_idx);
        let end = rnd_start + self.seq_len;
        let input = &self.val_data.get_ids()[rnd_start..end];
        input.to_vec()
    }
}
