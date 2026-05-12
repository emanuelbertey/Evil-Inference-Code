import struct
import os

def final_diagnose(path):
    print(f"--- DIAGNOSTICO FINAL BONSÁI ---")
    with open(path, 'rb') as f:
        f.seek(8)
        t_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Saltamos metadatos buscando la tabla de tensores
        # El primer tensor suele ser token_embd.weight
        anchor = b"token_embd.weight"
        data = f.read(10 * 1024 * 1024)
        pos = data.find(anchor)
        
        if pos != -1:
            # Retrocedemos al inicio de la entrada (longitud del nombre u64)
            curr = pos - 8
            print(f"Tabla de tensores encontrada en: {curr}")
            for i in range(10): # Vemos los primeros 10
                name_len = struct.unpack('<Q', data[curr:curr+8])[0]
                name = data[curr+8 : curr+8+name_len].decode('utf-8')
                ptr = curr + 8 + name_len
                n_dims = struct.unpack('<I', data[ptr:ptr+4])[0]
                ptr += 4
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack('<Q', data[ptr:ptr+8])[0]); ptr += 8
                ttype = struct.unpack('<I', data[ptr:ptr+4])[0]
                off = struct.unpack('<Q', data[ptr:ptr+8])[0]
                
                # Buscamos el SIGUIENTE para saber el tamaño real
                next_ptr = ptr + 12
                # (Simplificado para el primer vistazo)
                print(f"Tensor: {name:30} | Tipo: {ttype:2} | Dims: {dims}")
                curr = next_ptr
        else:
            print("No se encontró la tabla de tensores.")

if __name__ == "__main__":
    final_diagnose(r"D:\Ternary-Bonsai-1.7B-Q2_0.gguf")
