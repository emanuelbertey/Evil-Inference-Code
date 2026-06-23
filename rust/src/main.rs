// ─── Panel — Selección de Apps ───────────────────────────────────────────────
//
// Usage: cargo run --release --bin xoria

use std::io::{self, Write};

#[path = "../xorIA/xoria_bit.rs"]
mod xoria_bit;
#[path = "../xorIA/xoria_bit_cuda.rs"]
mod xoria_bit_cuda;
#[path = "../xorIA/xoria_cuant.rs"]
mod xoria_cuant;
#[path = "../xorIA/large_chat.rs"]
mod large_chat;
#[path = "../xorIA/large_chat_cuda.rs"]
mod large_chat_cuda;
#[path = "../xorIA/msltmchat.rs"]
mod msltmchat;
#[path = "../xorIA/msltmchat (cuda).rs"]
mod msltmchat_cuda;
#[path = "../xorIA/transformer_chat.rs"]
mod transformer_chat;
#[path = "../xorIA/transformer_chat_cuda.rs"]
mod transformer_chat_cuda;
#[path = "../xorIA/transformer_quant_kv.rs"]
mod transformer_quant_kv;
#[path = "../xorIA/auto_train.rs"]
mod auto_train;
#[path = "../xorIA/testlist.rs"]
mod testlist;
#[path = "../xorIA/bin/test_turbokuant.rs"]
mod test_turbokuant;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    Panel de Selección                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║                                                          ║");
    println!("║   xoria_bit:                                             ║");
    println!("║     1.  xoria_bit        (CPU)    BitLinear 1.58-bit     ║");
    println!("║     2.  xoria_bit_cuda   (CUDA)   BitLinear GPU          ║");
    println!("║     3.  xoria_cuant      (CPU)    TurboQuant KV          ║");
    println!("║                                                          ║");
    println!("║   xLSTM Large:                                           ║");
    println!("║     4.  large_chat       (CPU)    xLSTM Large Chat       ║");
    println!("║     5.  large_chat_cuda  (CUDA)   xLSTM Large GPU        ║");
    println!("║                                                          ║");
    println!("║   mSLTM:                                                 ║");
    println!("║     6.  msltmchat        (CPU)    mSLTM Chat             ║");
    println!("║     7.  msltmchat_cuda   (CUDA)   mSLTM GPU              ║");
    println!("║                                                          ║");
    println!("║   Transformer:                                           ║");
    println!("║     8.  transformer_chat (CPU)    Transformer Chat       ║");
    println!("║     9.  transformer_chat_cuda (CUDA) Transformer GPU     ║");
    println!("║     13. transformer_quant_kv (CPU) Transformer+TurboQuant║");
    println!("║                                                          ║");
    println!("║   Tools:                                                 ║");
    println!("║     10. auto_train      (auto)   HF + Training           ║");
    println!("║     11. testlist                 Test Suite               ║");
    println!("║     12. test_turbokuant          TurboQuant KV Cache      ║");
    println!("║                                                          ║");
    println!("║     q.  Salir                                            ║");
    println!("║                                                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    print!("  Seleccioná una opción [1-13/q]: ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim().to_lowercase();

    match choice.as_str() {
        "1" => {
            println!("\n  → xoria_bit (CPU)...\n");
            if let Err(e) = xoria_bit::xoria_cpu() { eprintln!("Error: {}", e); }
        }
        "2" => {
            println!("\n  → xoria_bit_cuda (CUDA)...\n");
            if let Err(e) = xoria_bit_cuda::xoria_cuda() { eprintln!("Error: {}", e); }
        }
        "3" => {
            println!("\n  → xoria_cuant (TurboQuant KV)...\n");
            if let Err(e) = xoria_cuant::xoria_cuant() { eprintln!("Error: {}", e); }
        }
        "4" => {
            println!("\n  → large_chat (CPU)...\n");
            if let Err(e) = large_chat::large_chat() { eprintln!("Error: {}", e); }
        }
        "5" => {
            println!("\n  → large_chat_cuda (CUDA)...\n");
            if let Err(e) = large_chat_cuda::large_chat_cuda() { eprintln!("Error: {}", e); }
        }
        "6" => {
            println!("\n  → msltmchat (CPU)...\n");
            if let Err(e) = msltmchat::msltmchat() { eprintln!("Error: {}", e); }
        }
        "7" => {
            println!("\n  → msltmchat_cuda (CUDA)...\n");
            if let Err(e) = msltmchat_cuda::msltmchat_cuda() { eprintln!("Error: {}", e); }
        }
        "8" => {
            println!("\n  → transformer_chat (CPU)...\n");
            if let Err(e) = transformer_chat::transformer_chat() { eprintln!("Error: {}", e); }
        }
        "9" => {
            println!("\n  → transformer_chat_cuda (CUDA)...\n");
            if let Err(e) = transformer_chat_cuda::transformer_chat_cuda() { eprintln!("Error: {}", e); }
        }
        "10" => {
            println!("\n  → auto_train (HF + Launcher)...\n");
            if let Err(e) = auto_train::auto_train_main() { eprintln!("Error: {}", e); }
        }
        "11" => {
            println!("\n  → testlist (Test Suite)...\n");
            testlist::testlist_main();
        }
        "12" => {
            println!("\n  -> test_turbokuant (TurboQuant KV Cache)...\n");
            if let Err(e) = test_turbokuant::test_turbokuant_main() { eprintln!("Error: {}", e); }
        }
        "13" => {
            println!("\n  → transformer_quant_kv (CPU Transformer + TurboQuant KV)...\n");
            if let Err(e) = transformer_quant_kv::transformer_quant_kv() { eprintln!("Error: {}", e); }
        }
        "q" | "quit" | "salir" | "" => {
            println!("  Saliendo.");
        }
        _ => {
            eprintln!("  Opción inválida: '{}'", choice);
        }
    }
}
