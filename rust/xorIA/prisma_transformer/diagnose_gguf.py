import struct

def diagnose_gguf(path):
    print(f"--- DIAGNOSTICO DE EMERGENCIA GGUF ---")
    with open(path, 'rb') as f:
        f.seek(8)
        t_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensores totales: {t_count}")
        
        # Saltamos metadatos rapido
        for _ in range(kv_count):
            l = struct.unpack('<Q', f.read(8))[0]; f.read(l)
            vt = struct.unpack('<I', f.read(4))[0]
            if vt == 8: l2 = struct.unpack('<Q', f.read(8))[0]; f.read(l2)
            elif vt == 9: f.seek(4, 1); al = struct.unpack('<Q', f.read(8))[0]; f.read(al * 4) # simplificado
            else: f.seek(4, 1) if vt < 10 else f.seek(8, 1)

        print("\n--- INFO DE TENSORES ---")
        for _ in range(t_count):
            l = struct.unpack('<Q', f.read(8))[0]; name = f.read(l).decode('utf-8')
            dc = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(dc)]
            tt = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            if "token_embd" in name or "blk.0.attn_q" in name:
                print(f"Tensor: {name:30} | Shape: {dims} | Tipo GGUF: {tt}")

if __name__ == "__main__":
    diagnose_gguf(r"D:\Ternary-Bonsai-1.7B-Q2_0.gguf")
