# Notas de layout GGUF: Bonsai 1.7B Prisma

Este archivo documenta la inconsistencia encontrada al cargar:

```text
D:\Ternary-Bonsai-1.7B-Q2_0.gguf
```

La causa del problema fue que el loader Python estaba interpretando el tensor `type=42` como otro formato. La pagina de Hugging Face de PrismaML identifica este archivo como:

```text
Ternary-Bonsai-1.7B-Q2_0.gguf = Q2_0 (g128)
```

Ese dato es clave: no es Q1_0 y tampoco es el TQ2_0 de 256 valores por bloque que se habia asumido.

## Datos observados

Cabecera GGUF:

```text
version=3
tensors=310
metadata=35
data_offset=5945280
architecture=qwen3
```

Tensores clave:

```text
output_norm.weight      shape=(2048,)        type=0   offset=0
token_embd.weight       shape=(2048,151669)  type=42  offset=8192
blk.0.attn_k.weight     shape=(2048,1024)    type=42  offset=82516128
blk.0.attn_norm.weight  shape=(2048,)        type=0   offset=83073696
blk.0.attn_q.weight     shape=(2048,2048)    type=42  offset=84196000
```

Aunque en algunas fuentes del codigo Prisma el id `42` tambien aparece asociado a otros tipos experimentales, en este archivo debe interpretarse como Bonsai/Prisma `Q2_0 (g128)`.

## Layout correcto

Para este GGUF:

```text
QK = 128 valores por bloque
block_stride = 34 bytes
```

Cada bloque se guarda asi:

```text
bytes 0..1   = escala fp16
bytes 2..33  = 32 bytes con codigos de 2 bits
```

Los 32 bytes contienen 128 valores porque cada byte guarda cuatro codigos de 2 bits.

## Comprobacion por offsets

Para `token_embd.weight`:

```text
shape = (2048, 151669)
blocks_per_row = 2048 / 128 = 16
row_bytes = 16 * 34 = 544
total = 544 * 151669 = 82,507,936 bytes
```

Ese total coincide exactamente con la distancia entre offsets:

```text
82516128 - 8192 = 82,507,936
```

Por eso la lectura correcta es:

```python
blocks_per_row = shape[0] // 128
block_stride = 34
d  = block[0:2]    # fp16
qs = block[2:34]   # 32 bytes
```

## Lo que no se debe hacer

No interpretar `type=42` en este archivo como Q1_0:

```text
QK = 32
block_bytes = 6
```

No interpretarlo como bloques de 256 valores y 68 bytes:

```text
QK = 256
block_stride = 68
```

Esa lectura de 68 bytes parecia tener sentido porque 68 = 2 bloques reales de 34 bytes. Pero agrupar dos bloques `g128` como si fueran un solo bloque `g256` mezcla escalas y codigos, asi que el modelo puede cargar tensores sin fallar y aun asi inferir texto basura.

## Sintomas si se interpreta mal

- las escalas `d` salen desplazadas o mezcladas;
- los pesos parecen cargar, pero no corresponden al layout real;
- la salida contiene tokens de idiomas aleatorios o texto sin coherencia;
- puede saturarse el disco si el parser GGUF tambien calcula mal offsets o tamanos;
- faltan `attn_sub_norm` y `ffn_sub_norm`, pero eso es esperado para este Qwen3 y debe tratarse como opcional.

## Estado esperado del loader

El loader correcto debe:

- parsear GGUF secuencialmente, no buscar nombres con `header.find`;
- usar `data_offset = align32(fin_tabla_tensores)`;
- usar offsets relativos a `data_offset`;
- detectar `vocab_size` desde `token_embd.weight.shape[1]`;
- tratar `type=42` como Prisma/Bonsai `Q2_0 (g128)` para este archivo;
- leer bloques de 34 bytes;
- copiar `d` desde bytes `0..1`;
- copiar `qs` desde bytes `2..33`;
- cargar tokenizer Qwen3, no Qwen2.
