# Notas de layout GGUF: Bonsai 1.7B Prisma

Este archivo documenta una inconsistencia real encontrada al cargar:

```text
D:\Ternary-Bonsai-1.7B-Q2_0.gguf
```

La causa del problema de carga infinita/saturacion de disco fue que el loader Python estaba tratando el archivo como si fuera `TQ2_0` GGUF estandar, pero el GGUF de Prisma usa un layout custom.

## Datos observados

Cabecera GGUF:

```text
version=3
tensors=310
metadata=35
data_offset=5945280
```

Tensores clave:

```text
output_norm.weight      shape=(2048,)        type=0   offset=0
token_embd.weight       shape=(2048,151669)  type=42  offset=8192
blk.0.attn_k.weight     shape=(2048,1024)    type=42  offset=82516128
blk.0.attn_norm.weight  shape=(2048,)        type=0   offset=83073696
blk.0.attn_q.weight     shape=(2048,2048)    type=42  offset=84196000
```

Aunque en `prism-ml-llama.cpp/ggml/include/ggml.h` el tipo `42` aparece como `GGML_TYPE_Q1_0`, el tamano entre offsets no coincide con Q1_0.

## Por que no es Q1_0

Para `token_embd.weight`:

```text
shape = (2048, 151669)
```

Si fuera Q1_0 estandar:

```text
block size = 32 valores
block bytes = 2 bytes escala + 4 bytes bits = 6
row bytes = 2048 / 32 * 6 = 384
total aproximado = 384 * 151669 = 58,240,896 bytes
```

Pero el siguiente tensor empieza en:

```text
82516128 - 8192 = 82,507,936 bytes despues
```

Ese tamano coincide con TQ2_0 custom con padding de fila:

```text
TQ2 block bytes = 66
blocks per row = 2048 / 256 = 8
raw row bytes = 8 * 66 = 528
padded row bytes = align32(528) = 544
total = 544 * 151669 = 82,507,936 bytes
```

Conclusion: `type=42` en este archivo debe tratarse como Prisma TQ2_0 con padding por fila, no como Q1_0.

## Layout real del bloque

El `TQ2_0` estandar de llama.cpp declara:

```c
typedef struct {
    uint8_t qs[64];
    ggml_half d;
} block_tq2_0;
```

Pero el archivo Prisma observado guarda los bloques asi:

```text
ggml_half d
uint8_t qs[64]
```

Es decir:

```text
bytes 0..1   = escala fp16
bytes 2..65  = pesos ternarios empaquetados
```

Esto fue confirmado leyendo los primeros bytes de `token_embd.weight`: interpretar los bytes `0..1` como fp16 da una escala razonable (`0.0271`), mientras que interpretar los bytes `64..65` como escala da valores basura (`162.25`, negativos, etc.).

## Layout por fila

Cada fila esta alineada a 32 bytes:

```python
blocks_per_row = shape[0] // 256
row_bytes = blocks_per_row * 66
stride = align32(row_bytes)
```

Para `shape[0] = 2048`:

```text
blocks_per_row = 8
row_bytes = 528
stride = 544
padding = 16 bytes por fila
```

El loader debe leer `stride` bytes por fila, pero copiar solo `row_bytes`.

## Regla para el loader Python

Para este GGUF:

```text
tensor_type == 42
```

se debe interpretar como:

```text
Prisma TQ2_0 custom:
  QK = 256
  block_bytes = 66
  block layout = d(fp16) + qs[64]
  row stride = align32(blocks_per_row * 66)
```

No se debe interpretar como:

```text
Q1_0:
  QK = 32
  block_bytes = 6
```

ni como:

```text
TQ2_0 estandar:
  block layout = qs[64] + d(fp16)
  sin asumir padding custom por fila
```

## Sintomas si se interpreta mal

Cuando el loader usa el layout equivocado:

- las escalas `d` salen negativas, gigantes o casi aleatorias;
- los offsets se desalinean despues del primer tensor grande;
- la carga parece avanzar indefinidamente o lee datos incorrectos;
- el disco queda saturado por lecturas enormes y repetidas;
- el modelo carga "algo", pero no infiere de manera util.

## Estado esperado

El loader correcto debe:

- parsear GGUF secuencialmente, no buscar nombres con `header.find`;
- usar `data_offset = align32(fin_tabla_tensores)`;
- usar offsets relativos a `data_offset`;
- autodetectar `vocab_size` desde `token_embd.weight.shape[1]`;
- tratar tipo `42` como Prisma TQ2_0 custom para este archivo;
- leer por filas con padding;
- copiar `d` desde bytes `0..1`;
- copiar `qs` desde bytes `2..65`.
