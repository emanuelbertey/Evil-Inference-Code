import torch
from config import PrismaConfig
from transformer import PrismaTransformer
from transformers import AutoTokenizer

def test_mini():
    print("1. Configurando Qwen3 MINI (2 capas, dim 128)...")
    config = PrismaConfig(
        dim=128, n_layers=2, n_heads=4, n_heads_kv=2, 
        vocab_size=151936, hidden_dim=256, rope_theta=1000000.0
    )
    
    print("2. Inicializando modelo MINI...")
    model = PrismaTransformer(config)
    
    print("3. Cargando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    
    print("4. Ejecutando Forward Pass...")
    prompt = "Hola!"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    
    with torch.no_grad():
        logits, _ = model(inputs)
    
    print(f"5. ¡Éxito! Logits shape: {logits.shape}")
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"Token: {next_token.item()} -> '{tokenizer.decode(next_token)}'")

if __name__ == "__main__":
    test_mini()
