mod data;
mod model;
mod train;
use crate::train::{training_loop, TrainConfig};
use data::Dataset;

fn main() -> anyhow::Result<()> {
    let train_cfg = TrainConfig::default();
    let mut ds = Dataset::new(
        "data/input.txt.gz",
        Some("data/shakespeare_tokenizer.json"),
        train_cfg.seq_len(),
    );
    training_loop(&mut ds, &train_cfg)?;
    Ok(())
}
