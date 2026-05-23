from transformers import AutoTokenizer

MODEL_ID = "facebook/opt-350m"  
SAVE_PATH = "./local_tokenizer"

print(f"Descargando tokenizador de {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Guardando tokenizador en {SAVE_PATH}...")
tokenizer.save_pretrained(SAVE_PATH)
print("¡Listo! Tokenizador extraído y guardado con éxito.")
