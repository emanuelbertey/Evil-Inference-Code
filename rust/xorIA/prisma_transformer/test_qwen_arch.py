import torch
from config import PrismaConfig
from transformer import PrismaTransformer
from transformers import AutoTokenizer
import time

def test_inference():
    print(">>> [PREVIA] Iniciando Configuración Qwen3 (Bonsai 1.7B)...", flush=True)
    config = PrismaConfig(
        dim=2048, n_layers=28, n_heads=16, n_heads_kv=8, 
        vocab_size=151936, hidden_dim=6144, rope_theta=1000000.0
    )
    
    print(">>> [PREVIA] Reservando Memoria para el Modelo...", flush=True)
    start_time = time.time()
    
    # Vamos a inicializar el modelo
    model = PrismaTransformer(config)
    
    print(f">>> [PREVIA] Modelo inicializado en {time.time() - start_time:.2f}s", flush=True)
    
    print(">>> [PREVIA] Cargando Tokenizer Qwen2...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    
    print(">>> [PREVIA] Preparando prompt de prueba...", flush=True)
    prompt = "Hola Bonsai, ¿estás listo?"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    
    print(f">>> [PREVIA] Ejecutando Forward Pass (28 capas, dim 2048)...", flush=True)
    with torch.no_grad():
        logits, _ = model(inputs)
    
    print(f">>> [FINAL] Logits calculados: {logits.shape}", flush=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f">>> [FINAL] Predicción: {tokenizer.decode(next_token)}", flush=True)

if __name__ == "__main__":
    test_inference()
