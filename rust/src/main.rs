// ─── xoria_bit Panel — Selección CPU / CUDA ─────────────────────────────────
//
// Panel interactivo que llama a xoria_cpu() o xoria_cuda() directamente.
// Usage: cargo run --release

use std::io::{self, Write};

#[path = "../xorIA/xoria_bit.rs"]
mod xoria_bit;

#[path = "../xorIA/xoria_bit_cuda.rs"]
mod xoria_bit_cuda;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         xoria_bit — Panel de Selección               ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║                                                      ║");
    println!("║   1.  CPU   (xoria_bit)        Entrenamiento/Infer.  ║");
    println!("║   2.  CUDA  (xoria_bit_cuda)   Entrenamiento GPU     ║");
    println!("║                                                      ║");
    println!("║   q.  Salir                                          ║");
    println!("║                                                      ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    print!("  Seleccioná una opción [1/2/q]: ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim().to_lowercase();

    match choice.as_str() {
        "1" | "cpu" => {
            println!("\n  → Iniciando xoria_bit (CPU)...\n");
            if let Err(e) = xoria_bit::xoria_cpu() {
                eprintln!("Error: {}", e);
            }
        }
        "2" | "cuda" => {
            println!("\n  → Iniciando xoria_bit_cuda (CUDA)...\n");
            if let Err(e) = xoria_bit_cuda::xoria_cuda() {
                eprintln!("Error: {}", e);
            }
        }
        "q" | "quit" | "salir" | "" => {
            println!("  Saliendo.");
        }
        _ => {
            eprintln!("  Opción inválida: '{}'", choice);
        }
    }
}
