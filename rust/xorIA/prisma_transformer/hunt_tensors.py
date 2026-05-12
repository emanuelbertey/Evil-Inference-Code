import os
import struct

def hunt_tensors(path):
    print(f"--- CAZADOR DE TENSORES GGUF ---")
    with open(path, 'rb') as f:
        # Buscamos el token_embd.weight (17 letras)
        # En GGUF el nombre viene precedido por su longitud (8 bytes)
        target = b'\x11\x00\x00\x00\x00\x00\x00\x00token_embd.weight'
        data = f.read(10 * 1024 * 1024) # Leemos 10MB
        pos = data.find(target)
        
        if pos != -1:
            print(f"Encontrado 'token_embd.weight' en pos: {pos}")
            # Estructura: [Nombre] [n_dims:u32] [dims:u64*N] [type:u32] [offset:u64]
            ptr = pos + len(target)
            n_dims = struct.unpack('<I', data[ptr:ptr+4])[0]
            ptr += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack('<Q', data[ptr:ptr+8])[0])
                ptr += 8
            ttype = struct.unpack('<I', data[ptr:ptr+4])[0]
            offset = struct.unpack('<Q', data[ptr:ptr+8])[0]
            print(f"RESULTADO: Dims={dims} | Tipo GGUF={ttype} | Offset={offset}")
        else:
            print("No se encontró el patrón de texto.")

if __name__ == "__main__":
    hunt_tensors(r"D:\Ternary-Bonsai-1.7B-Q2_0.gguf")
