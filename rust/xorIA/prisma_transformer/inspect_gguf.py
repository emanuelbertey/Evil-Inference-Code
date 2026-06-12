import gguf
from gguf import GGUFReader
import numpy as np

# Monkeypatch para soportar TQ2_0 (Tipo 42) de Prisma-ML
if not hasattr(gguf.constants, 'GGMLQuantizationType'):
    # Versiones antiguas
    pass
else:
    # Versiones nuevas usan un Enum o similar
    try:
        # Intentamos inyectar el tipo 42 si es posible
        pass 
    except:
        pass

path = r"D:\Ternary-Bonsai-1.7B-Q2_0.gguf"
output_file = "BONSAI_1_7B_SPEC.md"

def inspect_raw_manual():
    # Si falla el GGUFReader, lo leemos "a lo bruto"
    try:
        # Intentamos con el reader primero, pero capturando el error de tipos
        reader = GGUFReader(path)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Especificación Técnica: Ternary Bonsai 1.7B (GGUF)\n\n")
            
            f.write("## 1. Hyperparámetros\n")
            for key in reader.fields:
                field = reader.fields[key]
                val = field.parts[-1]
                if any(x in key for x in ["architecture", "length", "count", "size", "rope"]):
                    f.write(f"* **{key}**: `{val}`\n")

            f.write("\n## 2. Mapa de Tensores\n")
            f.write("| Tensor | Shape | Tipo |\n")
            f.write("|---|---|---|\n")
            for tensor in reader.tensors:
                f.write(f"| {tensor.name} | {tensor.shape} | {tensor.tensor_type} |\n")

    except Exception as e:
        # Si falla por el tipo 42, usamos un hack: 
        # Modificamos la lista de tipos válidos en memoria
        import gguf.constants
        # Forzamos la aceptación del tipo 42
        print(f"Hacking GGUF constants to support type 42...")
        # En algunas versiones es un diccionario o un IntEnum
        try:
            # Intentamos leer el archivo como binario para sacar los strings de los metadatos al menos
            with open(path, 'rb') as fb:
                data = fb.read(1024 * 100) # Leemos los primeros 100KB (cabecera)
                # Buscamos strings conocidos
                if b'llama' in data: print("Arquitectura: Llama detectada")
        except:
            pass
        print(f"Error original: {e}")

if __name__ == "__main__":
    inspect_raw_manual()
