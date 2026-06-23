use std::io::{self, Write};

#[path = "bin/test_cuda_kernel.rs"]
mod test_cuda_kernel;
#[path = "bin/test_ddos.rs"]
mod test_ddos;
#[path = "bin/test_xor_embed.rs"]
mod test_xor_embed;
#[path = "bin/test_kuantgrad.rs"]
mod test_kuantgrad;
#[path = "bin/test_turbokuant.rs"]
mod test_turbokuant;
#[path = "bin/test_cuda_kuantgrad.rs"]
mod test_cuda_kuantgrad;
#[path = "bin/test_rope.rs"]
mod test_rope;
#[path = "bin/test_transformer_vs_bit.rs"]
mod test_transformer_vs_bit;

pub fn testlist_main() {
    loop {
        println!();
        println!("========================================================");
        println!("                  Test Suite Menu");
        println!("========================================================");
        println!("  1. test_cuda_kernel    CUDA I2S Kernel");
        println!("  2. test_ddos           GPU Throughput Stress");
        println!("  3. test_xor_embed      XOR-Encrypted Kernel");
        println!("  4. test_kuantgrad      KuantGrad Optimizer");
        println!("  5. test_turbokuant     TurboQuant KV Cache");
        println!("  6. test_cuda_kuantgrad CUDA KuantGrad AdamW Kernel");
        println!("  7. test_rope           RoPE Mathematical Test");
        println!("  8. test_transformer_vs_bit  Transformer FP32 vs BitLinear");
        println!();
        println!("  b. Back to main menu");
        println!("========================================================");
        println!();
        print!("  Selecciona un test [1-8/b]: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        let choice = choice.trim().to_lowercase();

        match choice.as_str() {
            "1" => {
                println!("\n  -> test_cuda_kernel...\n");
                if let Err(e) = test_cuda_kernel::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "2" => {
                println!("\n  -> test_ddos (GPU Throughput Stress)...\n");
                if let Err(e) = test_ddos::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "3" => {
                println!("\n  -> test_xor_embed (XOR-Encrypted Kernel)...\n");
                if let Err(e) = test_xor_embed::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "4" => {
                println!("\n  -> test_kuantgrad (KuantGrad Optimizer)...\n");
                if let Err(e) = test_kuantgrad::test_kuantgrad_main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "5" => {
                println!("\n  -> test_turbokuant (TurboQuant KV Cache)...\n");
                if let Err(e) = test_turbokuant::test_turbokuant_main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "6" => {
                println!("\n  -> test_cuda_kuantgrad (CUDA KuantGrad AdamW Kernel)...\n");
                if let Err(e) = test_cuda_kuantgrad::main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "7" => {
                println!("\n  -> test_rope (RoPE Mathematical Test)...\n");
                if let Err(e) = test_rope::test_rope_main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "8" => {
                println!("\n  -> test_transformer_vs_bit (FP32 vs BitLinear)...\n");
                if let Err(e) = test_transformer_vs_bit::test_transformer_vs_bit_main() {
                    eprintln!("  Error: {}", e);
                }
            }
            "b" | "back" | "salir" | "" => {
                println!("  Volviendo al menu principal.");
                break;
            }
            _ => {
                eprintln!("  Opcion invalida: '{}'", choice);
            }
        }
    }
}
