use std::io::{self, Write};

#[path = "bin/test_cuda_kernel.rs"]
mod test_cuda_kernel;

pub fn testlist_main() {
    loop {
        println!();
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║               Test Suite Menu                       ║");
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║                                                    ║");
        println!("║     1. test_cuda_kernel    CUDA I2S Kernel         ║");
        println!("║                                                    ║");
        println!("║     b.  Back to main menu                          ║");
        println!("║                                                    ║");
        println!("╚══════════════════════════════════════════════════════╝");
        println!();
        print!("  Seleccioná un test [1/b]: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        let choice = choice.trim().to_lowercase();

        match choice.as_str() {
            "1" => {
                println!("\n  → test_cuda_kernel...\n");
                if let Err(e) = test_cuda_kernel::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "b" | "back" | "" => {
                println!("  Volviendo al menú principal.");
                break;
            }
            _ => {
                eprintln!("  Opción inválida: '{}'", choice);
            }
        }
    }
}
