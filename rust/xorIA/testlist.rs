use std::io::{self, Write};

#[path = "bin/test_cuda_kernel.rs"]
mod test_cuda_kernel;
#[path = "bin/test_ddos.rs"]
mod test_ddos;
#[path = "bin/test_xor_embed.rs"]
mod test_xor_embed;
#[path = "bin/test_kuantgrad.rs"]
mod test_kuantgrad;

pub fn testlist_main() {
    loop {
        println!();
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║               Test Suite Menu                       ║");
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║                                                    ║");
        println!("║     1. test_cuda_kernel    CUDA I2S Kernel         ║");
        println!("║     2. test_ddos           GPU Throughput Stress   ║");
        println!("║     3. test_xor_embed      XOR-Encrypted Kernel    ║");
        println!("║     4. test_kuantgrad      KuantGrad Optimizer     ║");
        println!("║                                                    ║");
        println!("║     b.  Back to main menu                          ║");
        println!("║                                                    ║");
        println!("╚══════════════════════════════════════════════════════╝");
        println!();
        print!("  Seleccioná un test [1-4/b]: ");
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
            "2" => {
                println!("\n  → test_ddos (GPU Throughput Stress)...\n");
                if let Err(e) = test_ddos::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "3" => {
                println!("\n  → test_xor_embed (XOR-Encrypted Kernel)...\n");
                if let Err(e) = test_xor_embed::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "4" => {
                println!("\n  → test_kuantgrad (KuantGrad Optimizer)...\n");
                if let Err(e) = test_kuantgrad::test_kuantgrad_main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "b" | "back" | "salir" | "" => {
                println!("  Volviendo al menú principal.");
                break;
            }
            _ => {
                eprintln!("  Opción inválida: '{}'", choice);
            }
        }
    }
}
