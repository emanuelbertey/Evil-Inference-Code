import torch

def pack_q1_0(weights_int8):
    """
    Empaqueta tensores de -1, 0, 1 en bloques de bits (Simulando GGML Q1_0 / Q2_0).
    En Prisma/Bonsai, 4 o 5 pesos ternarios caben en un solo byte (8 bits).
    Aca simulamos meter 4 pesos ternarios en 1 byte usando 2 bits por peso:
    -1 -> 00 (0)
     0 -> 01 (1)
    +1 -> 10 (2)
    """
    N, M = weights_int8.shape
    assert M % 4 == 0, "M debe ser multiplo de 4 para bloques de 1 byte"
    
    # Mapeo: -1 -> 0, 0 -> 1, 1 -> 2
    mapped = weights_int8 + 1 
    
    packed = torch.zeros((N, M // 4), dtype=torch.uint8)
    for i in range(4):
        # Desplazamos 2 bits por cada elemento: elemento 0 en bits 0-1, elemento 1 en bits 2-3...
        packed |= (mapped[:, i::4].to(torch.uint8) & 0b11) << (i * 2)
        
    return packed

def matmul_bitwise_ternary(x_int8, packed_w):
    """
    Multiplicacion estricta iterando a nivel de BITS en CPU.
    ESTO ES EXACTAMENTE LO QUE HACE EL CPU EN LLAMA.CPP.
    ¡Cero multiplicaciones! Solo mascaras de bits, sumas y restas.
    """
    B, S, M = x_int8.shape
    N, M_packed = packed_w.shape
    assert M == M_packed * 4
    
    out = torch.zeros((B, S, N), dtype=torch.int32)
    
    for b in range(B):
        for s in range(S):
            for n in range(N):
                acc = 0
                for m_p in range(M_packed):
                    byte_val = packed_w[n, m_p].item()
                    
                    # Extraer los 4 pesos del byte usando mascaras bit a bit
                    w0 = (byte_val & 0b00000011) - 1
                    w1 = ((byte_val & 0b00001100) >> 2) - 1
                    w2 = ((byte_val & 0b00110000) >> 4) - 1
                    w3 = ((byte_val & 0b11000000) >> 6) - 1
                    
                    # Obtener las 4 activaciones correspondientes
                    x_idx = m_p * 4
                    x0 = x_int8[b, s, x_idx].item()
                    x1 = x_int8[b, s, x_idx+1].item()
                    x2 = x_int8[b, s, x_idx+2].item()
                    x3 = x_int8[b, s, x_idx+3].item()
                    
                    # Multiplicacion simulada como llama.cpp (if w == 1 add, if w == -1 sub)
                    # El hardware hace esto nativamente o con instucciones popcount.
                    if w0 == 1: acc += x0
                    elif w0 == -1: acc -= x0
                    
                    if w1 == 1: acc += x1
                    elif w1 == -1: acc -= x1
                    
                    if w2 == 1: acc += x2
                    elif w2 == -1: acc -= x2
                    
                    if w3 == 1: acc += x3
                    elif w3 == -1: acc -= x3
                    
                out[b, s, n] = acc
                
    return out

if __name__ == "__main__":
    print("--- DEMOSTRACION DE TERNARIO A NIVEL DE BITS (COMO LLAMA.CPP) ---")
    # Generar pesos ternarios aleatorios de prueba
    M, N = 8, 2
    w_raw = torch.randint(-1, 2, (N, M)).to(torch.int8)
    print(f"Pesos originales (int8):\n{w_raw}")
    
    # 1. EMPAQUETADO (Simulando crear el archivo GGUF Q2_0)
    w_packed = pack_q1_0(w_raw)
    print(f"\nPesos Empaquetados (uint8 - 4 pesos por byte):\n{w_packed}")
    
    # 2. ACTIVACIONES
    x = torch.randint(-127, 127, (1, 1, M)).to(torch.int8)
    print(f"\nActivaciones de entrada (int8):\n{x}")
    
    # 3. MULTIPLICACION A NIVEL DE BITS
    out_bitwise = matmul_bitwise_ternary(x, w_packed)
    print(f"\nResultado de la multiplicacion BITWISE pura:\n{out_bitwise}")
    
    # Verificacion con Pytorch normal para demostrar que el empaquetado es perfecto
    out_normal = torch.matmul(x.to(torch.int32), w_raw.to(torch.int32).t())
    print(f"\nResultado usando torch.matmul normal (Para comparar):\n{out_normal}")
    assert torch.equal(out_bitwise, out_normal)
    print("\n¡Son exactamente iguales! Así es como el CPU de llama.cpp ejecuta los bloques de Prisma/Khosravipasha.")
