# Fix: Soporte de Embedding Ternario y Bloques QK=256

Se han realizado modificaciones críticas en la arquitectura para permitir la carga bit-exacta del modelo **Bonsai 1.7B (TQ2_0)** desde archivos GGUF.

## 1. Mismatch de Bloques (QK=256)
*   **Problema**: El script esperaba bloques de 32 pesos (8 bytes de bits), pero Bonsai usa bloques de 256 pesos (64 bytes de bits).
*   **Solución**: Se actualizó el parámetro `QK` a **256** en todas las capas BitLinear.
*   **Resultado**: Alineación perfecta con el buffer del GGUF, eliminando el error `size mismatch (8 vs 64)`.

## 2. Embedding Ternario (Quantized Input)
*   **Problema**: Se descubrió que Bonsai cuantiza incluso la capa de entrada (`token_embd.weight`) a 1.58 bits (TQ2_0). PyTorch estándar no puede cargar esto en un `nn.Embedding`.
*   **Solución**: Se implementó la clase `TernaryEmbedding` que utiliza la misma lógica de bloques que las capas lineales.
*   **Ahorro de RAM**: El embedding pasó de ocupar **1.2 GB** (en FP32) a solo **~80 MB** (en bits empaquetados).

## 3. Resolución de Vocabulario (55,827 vs 151,669)
*   **Problema**: Los metadatos de Qwen reportan un vocabulario de 151k, pero el tensor real en el GGUF solo tiene 55k filas.
*   **Solución**: El cargador ahora auto-detecta el tamaño real del tensor y ajusta la arquitectura dinámicamente.

## 4. Estado de RAM Final
Gracias a estos cambios, el modelo de 1.7B ahora es cargado íntegramente en formato ternario:
- **PyTorch Overhead**: ~1 GB
- **Modelo (Pesos + Embedding)**: ~463 MB
- **Uso Total Estimado**: **1.4 - 1.5 GB** (Muy cerca del objetivo de 1 GB).

---
**Nota**: Para llegar a <1 GB, la única vía restante es la migración a Rust (Burn/Candle), eliminando el overhead del intérprete de Python y PyTorch.

## 5. Descubrimiento Final: El Vocabulario de 151,051
Tras analizar el binario del GGUF, se ha confirmado que:
- Aunque los metadatos dicen **151,669**, el tensor real tiene bytes para exactamente **151,051 tokens**.
- El cargador ahora es **Agnóstico al Formato**: mide los huecos entre offsets en el archivo para determinar el tamaño real.
- El modelo se auto-configura para usar esos 151,051 tokens, evitando errores de desbordamiento de memoria.

mierda
