# El Flujo Real de Prisma/Ternarios (Q1_0 / Q2_0) en llama.cpp

Este documento detalla estrictamente cómo la arquitectura de C++ (`ggml` y Prisma de Khosravipasha) maneja los tres pilares del Transformer bajo el paradigma de 1.58 bits (Ternario).

---

## 1. Embeddings (El Mapeo Inicial)
**¿Se usan ternarios aquí? NO.**
*   **La Realidad de llama.cpp:** El bloque de embedding inicial (`token_embd`) **no es una multiplicación de matrices**, es un simple *Lookup* de memoria. Se implementa usando la operación `ggml_get_rows`.
*   **¿Por qué no es ternario?** Destruir la primera representación de un token reduciéndola a `-1, 0, 1` aniquila la semántica del modelo. Incluso en modelos Prisma/BitNet extremos, la tabla de embeddings suele mantenerse en `FP16` (Float de 16 bits) o cuantizaciones tradicionales (`Q4_0`, `Q8_0`).
*   **El proceso:** El ID del token se usa como índice para "copiar y pegar" el vector exacto de la RAM hacia la memoria de trabajo. 

---

## 2. BitLinear (Multiplicaciones Q, K, V y Red FeedForward)
**¿Se usan ternarios aquí? SÍ. ES EL CORAZÓN DEL SISTEMA.**
*   **Estructura de Memoria (`struct block_q2_0` o `block_q1_0`):** Los pesos estáticos del modelo NO son tensores de flotantes. Están guardados en discos y subidos a RAM en bloques que empaquetan entre 4 y 8 pesos ternarios por cada mísero byte (`uint8_t`), además de guardar una pequeña escala en flotante por bloque para corregir la magnitud después.
*   **Ejecución (`ggml_mul_mat`):** Cuando entra la activación (ej. lo que salió del Embedding), llama.cpp la convierte temporalmente a enteros de 8 bits (`int8`). 
*   **El Truco del Hardware:** Los kernels de CPU, CUDA o Vulkan (los que Khosravipasha agregó) iteran sobre estos bloques. En vez de usar las lentas unidades FPU (Floating Point Unit) para multiplicar, usan operaciones lógicas a nivel de bits y **conteo de bits (popcount)** o sumadores SIMD. Como el peso es 1 o -1, el hardware solo suma o resta el valor de activación. Cero multiplicaciones reales.

---

## 3. Cache VK (Vulkan Key-Value Cache)
**¿Se usan ternarios aquí? EL ENFOQUE ES COMPRESIÓN DE HISTORIAL.**
*   **El Problema:** La caché KV guarda todo el pasado para no recalcularlo. En modelos de contexto largo, esto agota la VRAM de la tarjeta de video, sin importar si los pesos del modelo son de 1 bit.
*   **La Solución de llama.cpp (Vulkan):** Cuando procesas un token y sacas su "Key" y su "Value", la función `cpy_k` y `cpy_v` de `ggml` inyecta esos vectores en el *Ring Buffer* de la memoria gráfica.
*   **La Integración Prisma / Vulkan:** Khosravipasha introdujo los encabezados SPIR-V (los *shaders* precompilados de Vulkan) para que la gráfica soporte realizar la atención iterativa leyendo matrices extremadamente empaquetadas. Esto permite que la KV Cache también se guarde en formatos altamente cuantizados (como Int8 o menores) y que los shaders de Vulkan la desempaqueten y calculen el `Q * K^T` a velocidades extremas usando la misma lógica de suma de bits descrita en BitLinear.

---

## 4. Archivos Clave de Implementación Ternaria en CPU (Proyecto Original)
Si vas al repositorio original `prism-ml-llama.cpp`, estos son los archivos específicos donde ocurre la manipulación física de los bits ternarios para el procesador central.

**IMPORTANTE: En el repositorio existen DOS familias de cuantización ternaria independientes:**
*   **`Q1_0` / `Q1_0_g128`** (Khosravipasha — los commits de Prisma): Cuantización binaria pura de 1 bit. Cada peso es `+1` o `-1` (sin cero). Empaqueta **1 bit por peso**.
*   **`TQ1_0` / `TQ2_0`** (ggerganov — los ternarios originales de BitNet/TriLM): Cuantización ternaria real de 1.58 bits. Cada peso es `-1`, `0` o `+1`. Empaqueta usando base-3 (trits).

### 4.1 Definición de los Structs de Bloques (ggml-common.h, líneas 180-288)

#### `block_q1_0` (Prisma — 1 bit puro, Khosravipasha)
```c
#define QK1_0 32   // 32 pesos por bloque
typedef struct {
    ggml_half d;           // escala (2 bytes, float16)
    uint8_t qs[QK1_0 / 8]; // 32 bits / 8 = 4 bytes de bits empaquetados
} block_q1_0;
// Total: 6 bytes por cada 32 pesos → 1.5 bpw
```
Cada bit en `qs` representa: `1 = +1`, `0 = -1`. **No hay valor cero.**

#### `block_q1_0_g128` (Prisma — grupo de 128)
```c
#define QK1_0_g128 128
typedef struct {
    ggml_half d;               // escala (2 bytes)
    uint8_t qs[QK1_0_g128 / 8]; // 128/8 = 16 bytes
} block_q1_0_g128;
// Total: 18 bytes por 128 pesos → ~1.125 bpw
```

#### `block_tq1_0` (Ternario original — 1.6875 bpw)
```c
typedef struct {
    uint8_t qs[(QK_K - 4*QK_K/64) / 5]; // 5 trits por byte (3^5=243 < 256)
    uint8_t qh[QK_K/64];                // 4 trits por byte
    ggml_half d;
} block_tq1_0;
```

#### `block_tq2_0` (Ternario original — 2.0625 bpw)
```c
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits por elemento (256/4 = 64 bytes)
    ggml_half d;
} block_tq2_0;
```
Cada par de bits: `00=-1`, `01=0`, `10=+1`, `11=reservado`.

### 4.2 El Dot Product Real en CPU (ggml-cpu/quants.c)

#### `ggml_vec_dot_q1_0_q8_0_generic` (Prisma Q1_0 — líneas 127-166)
```c
// Itera por bloques de 32 pesos
for (int i = 0; i < nb; i++) {
    const float d0 = GGML_FP16_TO_FP32(x[i].d);  // escala del peso
    const float d1 = GGML_FP16_TO_FP32(y[i].d);  // escala de la activación
    int sumi = 0;
    for (int j = 0; j < QK1_0; j++) {
        const int byte_index = j / 8;
        const int bit_offset = j % 8;
        // Extraer 1 bit: si es 1 → +1, si es 0 → -1
        const int xi = ((x[i].qs[byte_index] >> bit_offset) & 1) ? 1 : -1;
        const int yi = y[i].qs[j];  // activación int8
        sumi += xi * yi;  // suma/resta entera pura
    }
    sumf += d0 * d1 * sumi;  // escalar solo al final
}
```

#### `ggml_vec_dot_tq2_0_q8_K_generic` (TQ2_0 ternario — líneas 524-554)
```c
// Itera bloques de 256 pesos (QK_K)
for (int i = 0; i < nb; ++i) {
    int32_t sumi = 0;
    for (size_t j = 0; j < sizeof(x->qs); j += 32) {
        for (size_t l = 0; l < 4; ++l) {           // 4 pares de bits por byte
            for (size_t k = 0; k < 32; ++k) {
                // Extraer 2 bits, restar 1 para obtener {-1, 0, +1}
                sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
            }
        }
    }
    const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    sumf += (float) sumi * d;
}
```

### 4.3 Archivos del Proyecto Original

| Archivo | Función |
|---|---|
| `ggml/src/ggml-common.h` (líneas 180-288) | Structs `block_q1_0`, `block_tq1_0`, `block_tq2_0` |
| `ggml/src/ggml-cpu/quants.h` (líneas 15-16, 34-35, 40-42, 58-59, 75-76, 86-87) | Declaraciones de `quantize_row_q1_0`, `quantize_row_tq1_0/tq2_0`, `ggml_vec_dot_q1_0_q8_0`, `ggml_vec_dot_tq1_0_q8_K`, `ggml_vec_dot_tq2_0_q8_K` |
| `ggml/src/ggml-cpu/quants.c` (líneas 25-31, 105-117, 127-213, 472-554) | Implementación genérica de cuantización y dot products |
| `ggml/src/ggml-cpu/ggml-cpu.c` y `ops.cpp` | Backend maestro / enrutador de operaciones |
| `ggml/src/ggml-cpu/arch/arm/quants.c` (220KB) | Rutinas NEON optimizadas para ARM/Apple Silicon |
| `ggml/src/ggml-cpu/arch/x86/quants.c` (187KB) | Rutinas AVX2/SSE optimizadas para Intel/AMD |
| `ggml/src/ggml-cuda/vecdotq.cuh`, `mmq.cu` | Kernels CUDA para GPU Nvidia |
| `ggml/src/ggml-metal/ggml-metal.metal` | Shaders Metal para Apple GPU |

### 4.4 Diferencia Clave: Mi Implementación Python vs. el Código Real

| Aspecto | Mi `layers.py` | Código real C++ |
|---|---|---|
| Tipo de bloque | Simulé `TQ2_0` (2 bits/peso, con cero) | Existen `Q1_0` (1 bit, sin cero), `TQ1_0` (trits base-3) y `TQ2_0` (2 bits) |
| Tamaño de bloque | 32 pesos | `Q1_0`=32, `Q1_0_g128`=128, `TQ1_0/TQ2_0`=256 (QK_K) |
| Empaquetado | 4 pesos por byte (2 bits c/u) | `Q1_0`: 8 pesos por byte (1 bit c/u). `TQ1_0`: 5 trits por byte |
| Dot product | Bucle Python con if/else | Bucle C con operaciones de bits + versiones SIMD por arquitectura |
