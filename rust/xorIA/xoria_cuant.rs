use burn::{
    module::{AutodiffModule, Module},
    nn::EmbeddingConfig,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use burn_autodiff::Autodiff;
use burn_flex::Flex;
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use xlstm::blocks::bitlinear::kernel::KernelKind;
use xlstm::blocks::bitlinear::layer::BitLinearConfig;
use xlstm::blocks::trasformer_bit::infer_bit;
use xlstm::blocks::trasformer_bit::model::{
    sample_from_logits, BitLinearOutputProjection, BitLinearQKVProjection, BitLinearRMSNorm,
    BitLinearSwiGLUFeedForward, BitLinearTransformerLayer, BitLinearTransformerStack,
    FileFragmentIterator, KuantKVCache, Tokenizer, TransformerBitLinearLM,
    TransformerInferenceState,
};

type MyBackend = Autodiff<Flex<f32>>;

#[derive(serde::Deserialize, Debug)]
struct TrainConfig {
    layers: Option<usize>,
    d_model: Option<usize>,
    num_heads: Option<usize>,
}

fn load_config() -> Option<TrainConfig> {
    let path = "config.toml";
    let content = std::fs::read_to_string(path).ok()?;
    toml::from_str(&content).ok()
}

fn generate_text_kuant_cached<B: Backend>(
    model: &TransformerBitLinearLM<B>,
    inf_state: &TransformerInferenceState,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    caches: Vec<KuantKVCache>,
    mut current_offset: usize,
) -> (String, usize, f32, Vec<KuantKVCache>, usize, usize) {
    let ids = tokenizer.encode(seed_text);
    if ids.is_empty() {
        return (seed_text.to_string(), 0, 0.0, caches, current_offset, 0);
    }

    let device: B::Device = Default::default();
    let start_gen = Instant::now();
    let seed_len = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|&id| id as i64).collect(), [1, seed_len]),
        &device,
    );

    let (logits, mut caches) =
        model.forward_with_kuant_cache_inference(input, current_offset, caches, inf_state);

    let [_, s_len, v_dim] = logits.dims();
    let last_logits = logits
        .slice([0..1, (s_len - 1)..s_len, 0..v_dim])
        .reshape([1, v_dim]);

    let mut history: Vec<usize> = ids.clone();
    let mut generated = Vec::new();
    current_offset += seed_len;
    let mut next_id =
        sample_from_logits(last_logits, temperature, top_k, top_p, repetition_penalty, &history);

    let mut total_model_time = 0.0f32;
    let mut total_other_time = 0.0f32;

    for _ in 0..length {
        if let Some(token) = tokenizer.id_to_token(next_id) {
            if token == "eos" {
                break;
            }
        }

        generated.push(next_id);
        history.push(next_id);
        if history.len() > 64 {
            history.remove(0);
        }

        let token_raw = tokenizer.id_to_token(next_id).unwrap_or_default();
        let clean_str = token_raw.replace('\u{2581}', " ").replace(' ', " ");
        print!("{}", clean_str);
        io::stdout().flush().unwrap();

        let t0 = Instant::now();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_id as i64], [1, 1]), &device);
        let (next_logits, new_caches) =
            model.forward_with_kuant_cache_inference(input, current_offset, caches, inf_state);
        let model_time = t0.elapsed().as_secs_f32();
        total_model_time += model_time;

        let t1 = Instant::now();
        caches = new_caches;
        current_offset += 1;

        let [_, _, v] = next_logits.dims();
        let logits_2d = next_logits.reshape([1, v]);
        next_id = sample_from_logits(
            logits_2d,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            &history,
        );
        total_other_time += t1.elapsed().as_secs_f32();
    }

    let elapsed = start_gen.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated);
    println!();
    if !generated.is_empty() {
        let n = generated.len() as f32;
        println!(
            "[DEBUG] Modelo: {:.3}s ({:.1} ms/tok) | Other: {:.3}s ({:.1} ms/tok) | Kernel+attn: ~{:.1}%",
            total_model_time,
            total_model_time * 1000.0 / n,
            total_other_time,
            total_other_time * 1000.0 / n,
            total_model_time / elapsed.max(0.001) * 100.0
        );
    }
    let cache_bytes = caches.iter().map(KuantKVCache::compressed_bytes).sum();
    (text, generated.len(), elapsed, caches, current_offset, cache_bytes)
}

pub fn xoria_cuant() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     xoria_cuant — BitLinear CPU + TurboQuant KV Cache         ║");
    println!("║     Inferencia BitNet sin trim de cache KV                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let text_file = if args.len() >= 2 {
        args[1].clone()
    } else {
        "xorIA/input.txt".to_string()
    };

    let model_path = "transformer_bit2";
    let model_file = format!("{}.mpk", model_path);
    let bitnet_file = format!("{}.bitnet", model_path);
    let tokenizer_file = format!("{}_tokenizer.json", model_path);
    let model_exists = Path::new(&model_file).exists();
    let bitnet_exists = Path::new(&bitnet_file).exists();

    let target_vocab_size = 16000;
    let tokenizer = if Path::new(&tokenizer_file).exists() {
        println!("Cargando tokenizer BPE desde {}...", tokenizer_file);
        Tokenizer::load(&tokenizer_file)?
    } else {
        println!("Leyendo primeros 50MB para entrenar tokenizer...");
        let mut frag_iter = FileFragmentIterator::new(Path::new(&text_file), 50)?;
        let text = frag_iter.next().unwrap_or_default();
        println!("Entrenando tokenizer BPE (vocab_size={})...", target_vocab_size);
        let tok = Tokenizer::from_text(&text, target_vocab_size)?;
        tok.save(&tokenizer_file)?;
        tok
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size (BPE): {}", vocab_size);

    let mut temperature = 0.8f32;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.95;
    let mut repetition_penalty: f32 = 1.1;
    let mut d_model: usize = 512;
    let mut num_layers: usize = 6;
    let mut num_heads: usize = 8;
    let mut modo_inferencia = false;

    if let Some(cfg) = load_config() {
        d_model = cfg.d_model.unwrap_or(d_model);
        num_layers = cfg.layers.unwrap_or(num_layers);
        num_heads = cfg.num_heads.unwrap_or(num_heads);
        println!("config.toml encontrado — usando parámetros automáticos.");
    }

    if bitnet_exists || model_exists {
        loop {
            println!("\n--- CONFIGURACIÓN ACTUAL ---");
            println!(
                "  (1) d_model: {}  (2) Layers: {}  (3) Heads: {}",
                d_model, num_layers, num_heads
            );
            println!(
                "  (4) Temp: {}  (5) Top-K: {}  (6) Top-P: {}  (7) R-Pen: {}",
                temperature, top_k, top_p, repetition_penalty
            );
            println!("----------------------------");
            print!("¿Inferir con TurboQuant (i) o Ajustar (s)? [i/s]: ");
            io::stdout().flush()?;
            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();
            if choice == "i" {
                modo_inferencia = true;
                break;
            } else if choice == "s" {
                macro_rules! read_param {
                    ($label:expr, $val:expr) => {
                        print!("{} [{}]: ", $label, $val);
                        io::stdout().flush()?;
                        let mut buf = String::new();
                        io::stdin().read_line(&mut buf)?;
                        if let Ok(v) = buf.trim().parse() {
                            $val = v;
                        }
                    };
                }
                read_param!("d_model", d_model);
                read_param!("Layers", num_layers);
                read_param!("Heads", num_heads);
                read_param!("Temp", temperature);
                read_param!("Top-K", top_k);
                read_param!("Top-P", top_p);
                read_param!("R-Pen", repetition_penalty);
            }
        }
    } else {
        return Err(
            "No se encontró `transformer_bit2.bitnet` ni `transformer_bit2.mpk` para inferencia."
                .into(),
        );
    }

    if !modo_inferencia {
        return Ok(());
    }

    let device = Default::default();
    let num_kv_groups = 4;
    let head_dim = d_model / num_heads;
    let ffn_dim = ((4.0 * d_model as f64 * 2.0 / 3.0) as usize / 64 + 1) * 64;

    println!("\n── Configuración ──");
    println!(
        "  d_model={} | layers={} | heads={} | kv_groups={}",
        d_model, num_layers, num_heads, num_kv_groups
    );
    println!(
        "  head_dim={} | ffn_dim={} | TurboQuant KV=3-bit | sin trim\n",
        head_dim, ffn_dim
    );

    let mut model: TransformerBitLinearLM<MyBackend>;
    let mut loaded_state: Option<TransformerInferenceState> = None;

    if bitnet_exists {
        println!(
            "Cargando modelo BitNet ({} MB)...",
            std::fs::metadata(&bitnet_file)
                .map(|m| m.len() as f64 / 1e6)
                .unwrap_or(0.0)
        );
        let (loaded_model, state) =
            infer_bit::load::<MyBackend>(&bitnet_file, &device, KernelKind::Tile16, KernelKind::I2S)?;
        model = loaded_model;
        loaded_state = Some(state);
    } else {
        let layers = (0..num_layers)
            .map(|_| BitLinearTransformerLayer {
                attn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
                qkv: BitLinearQKVProjection {
                    q_proj: BitLinearConfig {
                        in_features: d_model,
                        out_features: num_heads * head_dim,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    k_proj: BitLinearConfig {
                        in_features: d_model,
                        out_features: num_kv_groups * head_dim,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    v_proj: BitLinearConfig {
                        in_features: d_model,
                        out_features: num_kv_groups * head_dim,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    num_heads,
                    num_kv_groups,
                    head_dim,
                },
                o_proj: BitLinearOutputProjection {
                    o_proj: BitLinearConfig {
                        in_features: num_heads * head_dim,
                        out_features: d_model,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    num_heads,
                    head_dim,
                },
                ffn_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
                ffn: BitLinearSwiGLUFeedForward {
                    gate_up_proj: BitLinearConfig {
                        in_features: d_model,
                        out_features: 2 * ffn_dim,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    down_proj: BitLinearConfig {
                        in_features: ffn_dim,
                        out_features: d_model,
                        bias: false,
                        activation_bits: 8,
                        rms_norm_eps: 1e-5,
                    }
                    .init(&device),
                    dropout: burn::nn::DropoutConfig::new(0.1).init(),
                    intermediate_dim: ffn_dim,
                },
                residual_dropout: burn::nn::DropoutConfig::new(0.1).init(),
            })
            .collect();

        model = TransformerBitLinearLM {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(&device),
            transformer: BitLinearTransformerStack {
                final_norm: BitLinearRMSNorm::new(d_model, 1e-5, &device),
                num_layers,
                d_model,
                layers,
            },
            head: BitLinearConfig {
                in_features: d_model,
                out_features: vocab_size,
                bias: false,
                activation_bits: 8,
                rms_norm_eps: 1e-5,
            }
            .init(&device),
            vocab_size,
            d_model,
            num_layers,
        };

        println!("Cargando pesos del modelo MPK...");
        let record = CompactRecorder::new().load(model_file.clone().into(), &device)?;
        model = model.load_record(record);
    }

    println!("Pre-computando kernels ternarios...");
    let inf_start = Instant::now();
    let mut model_v = model.valid();
    let inf_state = if let Some(state) = loaded_state {
        println!("  Inference state ya cargado desde .bitnet.");
        state
    } else {
        model_v.build_inference_state(&device, KernelKind::Tile16, KernelKind::I2S)
    };
    model_v.release_all_weights(&device);
    println!("Kernels listos en {:.2}s\n", inf_start.elapsed().as_secs_f32());

    let kv_seed = 42u64;
    let mut kv_bits = loop {
        print!("TurboQuant bits (2/3/4) [3]: ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() { break 3usize; }
        match line.parse::<usize>() {
            Ok(2 | 3 | 4) => break line.parse().unwrap(),
            _ => println!("  Opciones: 2, 3 o 4"),
        }
    };
    let mut current_len = 50usize;
    let mut session_caches = model_v.build_kuant_caches(kv_bits, kv_seed);
    let mut session_offset = 0usize;

    println!("Comandos: 'len <n>', 'temp <f>', 'topk <n>', 'topp <f>', 'rpen <f>', 'quant <n>', 'reset', 'salir'\n");

    loop {
        print!(
            "Chat [kuant:{}b len:{} t:{} k:{} p:{} rp:{}] > ",
            kv_bits, current_len, temperature, top_k, top_p, repetition_penalty
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.to_lowercase().starts_with("len ") {
            if let Ok(v) = input[4..].trim().parse::<usize>() {
                current_len = v;
                println!("  -> Longitud: {}\n", current_len);
                continue;
            }
        }
        if input.to_lowercase().starts_with("temp ") {
            if let Ok(v) = input[5..].trim().parse::<f32>() {
                temperature = v;
                println!("  -> Temperatura: {}\n", temperature);
                continue;
            }
        }
        if input.to_lowercase().starts_with("topk ") {
            if let Ok(v) = input[5..].trim().parse::<usize>() {
                top_k = v;
                println!("  -> Top-K: {}\n", top_k);
                continue;
            }
        }
        if input.to_lowercase().starts_with("topp ") {
            if let Ok(v) = input[5..].trim().parse::<f32>() {
                top_p = v;
                println!("  -> Top-P: {}\n", top_p);
                continue;
            }
        }
        if input.to_lowercase().starts_with("rpen ") {
            if let Ok(v) = input[5..].trim().parse::<f32>() {
                repetition_penalty = v;
                println!("  -> R-Pen: {}\n", repetition_penalty);
                continue;
            }
        }
        if input.to_lowercase().starts_with("quant ") {
            if let Ok(v) = input[6..].trim().parse::<usize>() {
                match v {
                    2 | 3 | 4 => {
                        kv_bits = v;
                        session_caches = model_v.build_kuant_caches(kv_bits, kv_seed);
                        session_offset = 0;
                        println!("  -> TurboQuant cambiado a {} bits, cache reiniciada.\n", kv_bits);
                    },
                    _ => println!("  Opciones: 2, 3 o 4\n"),
                }
                continue;
            }
        }
        if input.eq_ignore_ascii_case("reset") {
            session_caches = model_v.build_kuant_caches(kv_bits, kv_seed);
            session_offset = 0;
            println!("  -> Cache TurboQuant reiniciada.\n");
            continue;
        }
        if input.is_empty() {
            continue;
        }

        println!("\n--- TEXTO GENERADO (TurboQuant KV) ---");
        let (_, tokens_count, elapsed, updated_caches, updated_offset, cache_bytes) =
            generate_text_kuant_cached(
                &model_v,
                &inf_state,
                &tokenizer,
                input,
                current_len,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                session_caches,
                session_offset,
            );
        session_caches = updated_caches;
        session_offset = updated_offset;
        let tps = tokens_count as f32 / elapsed.max(0.001);
        println!("---");
        println!(
            "Tokens: {} | Tiempo: {:.2}s | {:.2} tok/s | Offset: {} | KV comprimida: {:.2} KB\n",
            tokens_count,
            elapsed,
            tps,
            session_offset,
            cache_bytes as f32 / 1024.0
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn main() {
    if let Err(e) = xoria_cuant() {
        eprintln!("Error: {}", e);
    }
}
