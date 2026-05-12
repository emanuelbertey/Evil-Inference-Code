# Especificación Técnica Real: Ternary Bonsai 1.7B (Qwen3 Architecture)

Este modelo es una implementación ternaria (1.58-bit) basada en la arquitectura **Qwen**, optimizada para contexto largo y alta eficiencia.

## 1. Configuración de Arquitectura (Qwen3)
*   **Capas (`block_count`)**: 28
*   **Dimensión (`embedding_length`)**: 2048
*   **FFN Dim (`feed_forward_length`)**: 6144
*   **Cabezas de Atención (Query)**: 16
*   **Cabezas KV (GQA)**: 8 (Ratio 2:1)
*   **Dimensión de Cabeza (`head_dim`)**: 128
*   **Longitud de Contexto**: 32,768 tokens
*   **RMSNorm Epsilon**: 1e-6

## 2. Parámetros de RoPE (Contexto Largo)
*   **Base de Frecuencia**: 1,000,000
*   **Escalado**: Yarn (Factor 4)
*   **Contexto Original**: 8,192

## 3. Tokenizer (Qwen2/GPT2)
*   **Modelo**: GPT2 / Tiktoken
*   **EOS Token ID**: 151645
*   **Padding Token ID**: 151643
*   **Vocabulario**: ~151k tokens (Basado en Qwen2)

## 4. Cuantización Ternaria
*   **Formato**: GGUF TQ2_0 (Tipo 41/42)
*   **Bit-logic**: BitNet b1.58 ({-1, 0, +1})
*   **Sub-norms**: Activadas para estabilidad en aritmética ternaria.
