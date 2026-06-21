// ─── auto_train: Launcher Automatizado ────────────────────────────
//
// Lee config.toml, descarga modelo/tokenizer/dataset de HuggingFace,
// lanza un thread que sube .mpk cada N minutos,
// y lanza xoria_bit (CPU) o xoria_bit_cuda (CUDA) como subproceso.
//
// Usage: cargo run --release --bin auto_train
//        (o desde el menú principal, opción 9)
//
// Config TOML example (config.toml):
//   name = "transformer_bit2"
//   url = "https://huggingface.co/usuario/repo"
//   model_mpk = "model.mpk"
//   bpejson = "tokenizer.json"
//   dataset = "dataset.txt"
//   layers = 6
//   d_model = 512
//   num_heads = 8
//   batch = 8
//   lr = 3e-4
//   epochs = 10
//   upload_interval_minutes = 10

use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use xlstm::blocks::trasformer_bit::hf_manager;

#[derive(serde::Deserialize, Debug)]
struct Config {
    name: Option<String>,
    url: Option<String>,
    model_mpk: Option<String>,
    bpejson: Option<String>,
    dataset: Option<String>,
    layers: Option<usize>,
    d_model: Option<usize>,
    num_heads: Option<usize>,
    batch: Option<usize>,
    lr: Option<f64>,
    epochs: Option<usize>,
    upload_interval_minutes: Option<u64>,
}

pub fn auto_train_main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Auto Train — HuggingFace Downloader + Launcher              ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    let config_content = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("No se pudo leer {}: {}", config_path, e))?;
    let config: Config = toml::from_str(&config_content)
        .map_err(|e| format!("Error parseando {}: {}", config_path, e))?;

    println!("Config:\n  name: {:?}\n  url: {:?}\n", config.name, config.url);

    let model_name = config.name.as_deref().unwrap_or("transformer_bit2");
    let model_file = format!("{}.mpk", model_name);
    let tokenizer_file = format!("{}_tokenizer.json", model_name);
    let token = hf_manager::get_hf_token();

    // ─── Download from HF ────────────────────────────────────────────
    if let Some(ref url) = config.url {
        let hf_dir = format!("hf_downloads/{}", hf_manager::parse_repo(url));
        println!("Descargando desde: {}", url);

        if let Some(f) = config.model_mpk.as_deref() {
            let _ = hf_manager::download_file(url, f, &token, &hf_dir)?;
            let src = format!("{}/{}", hf_dir, f);
            if Path::new(&src).exists() && !Path::new(&model_file).exists() {
                std::fs::copy(&src, &model_file)?;
                println!("  → {}", model_file);
            }
        }

        if let Some(f) = config.bpejson.as_deref() {
            let _ = hf_manager::download_file(url, f, &token, &hf_dir)?;
            let src = format!("{}/{}", hf_dir, f);
            if Path::new(&src).exists() && !Path::new(&tokenizer_file).exists() {
                std::fs::copy(&src, &tokenizer_file)?;
                println!("  → {}", tokenizer_file);
            }
        }

        if let Some(f) = config.dataset.as_deref() {
            match hf_manager::download_file(url, f, &token, &hf_dir) {
                Ok(_) => {},
                Err(e) => eprintln!("  [WARN] Dataset '{}' no encontrado en HF ({}), continuando con dataset por defecto de xoria...", f, e),
            }
        }
    } else {
        println!("(sin url, saltando descarga HF)");
    }

    // ─── Background upload thread ────────────────────────────────────
    let upload_handle = if let (Some(ref url), Some(minutes)) = (config.url, config.upload_interval_minutes) {
        if minutes > 0 {
            let running = Arc::new(AtomicBool::new(true));
            let r_for_thread = running.clone();
            let url = url.clone();
            let mf = model_file.clone();
            let interval = Duration::from_secs(minutes * 60);
            println!("  Upload thread cada {} min a {}", minutes, url);
            std::thread::spawn(move || {
                while r_for_thread.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    if !r_for_thread.load(Ordering::Relaxed) { break; }
                    let tmp = format!("{}.upload_tmp", mf);
                    if std::fs::copy(&mf, &tmp).is_ok() {
                        println!("\n[Upload] Subiendo {}...", mf);
                        if let Err(e) = hf_manager::upload_file(&url, &tmp, &mf, "", &format!("Auto-save {}", mf)) {
                            eprintln!("[Upload] Error: {}", e);
                        } else {
                            println!("[Upload] OK");
                        }
                        let _ = std::fs::remove_file(&tmp);
                    } else {
                        eprintln!("[Upload] No se pudo copiar {} (modelo guardándose?)", mf);
                    }
                }
            });
            Some(running)
        } else { None }
    } else { None };

    // ─── Menu ────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Seleccioná qué lanzar                                 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║                                                          ║");
    println!("║   1.  xoria_bit      (CPU)    BitLinear 1.58-bit        ║");
    println!("║   2.  xoria_bit_cuda (CUDA)   BitLinear GPU             ║");
    println!("║                                                          ║");
    println!("║   q.  Salir                                              ║");
    println!("║                                                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    print!("  Opción [1-2/q]: ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;
    match choice.trim().to_lowercase().as_str() {
        "1" => {
            println!("\n  → xoria_bit (CPU)...\n");
            let status = std::process::Command::new(
                std::env::current_exe()
                    .map(|p| p.parent().unwrap().join("xoria_bit.exe"))
                    .unwrap_or_else(|_| "xoria_bit.exe".into())
            )
            .status()?;
            if !status.success() {
                eprintln!("xoria_bit terminó con error: {:?}", status.code());
            }
        }
        "2" => {
            println!("\n  → xoria_bit_cuda (CUDA)...\n");
            let status = std::process::Command::new(
                std::env::current_exe()
                    .map(|p| p.parent().unwrap().join("xoria_bit_cuda.exe"))
                    .unwrap_or_else(|_| "xoria_bit_cuda.exe".into())
            )
            .status()?;
            if !status.success() {
                eprintln!("xoria_bit_cuda terminó con error: {:?}", status.code());
            }
        }
        "q" | "quit" | "salir" | "" => println!("  Saliendo."),
        _ => eprintln!("  Opción inválida."),
    }

    if let Some(running) = upload_handle {
        running.store(false, Ordering::Relaxed);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    auto_train_main()
}
