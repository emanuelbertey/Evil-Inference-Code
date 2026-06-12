# xLSTM Rust Benchmarks & Performance Comparison

Este documento detalla las diferencias de rendimiento y eficiencia entre la implementación estándar de **mLSTM** (primer paper) y la nueva arquitectura **XLSTMLarge** (basada en el último paper oficial).

## Comparativa de Rendimiento

| Métrica | mLSTM Estándar (Original) | XLSTMLarge (Optimizado) | Mejora |
| :--- | :---: | :---: | :---: |
| **Velocidad de Generación** | 80 - 90 tok/s | 230 - 240 tok/s | **~3x más rápido** |
| **Uso de Memoria (VRAM/RAM)** | 3080 MB | 1100 MB | **~65% menos memoria** |

### Análisis de Resultados

1.  **Velocidad (Inferencia/Entrenamiento)**: 
    *   La arquitectura **XLSTMLarge** implementa las últimas optimizaciones en la gestión de estados y proyecciones. 
    *   Mientras que el mLSTM básico procesa tokens a una tasa modesta, el modo **Large** alcanza velocidades significativamente superiores, permitiendo una experiencia de chat mucho más fluida y tiempos de entrenamiento reducidos.

2.  **Eficiencia de Memoria**:
    *   La diferencia de **1980 MB** en el consumo de memoria es vital para ejecutar modelos más grandes en hardware con recursos limitados.
    *   **XLSTMLarge** logra una huella de memoria mucho más ligera (1.1 GB frente a 3.1 GB), lo que permite aumentar el tamaño del vocabulario o la cantidad de bloques sin saturar el sistema.

### Conclusión

La adopción del modo **XLSTMLarge** no solo triplica la velocidad de procesamiento, sino que reduce drásticamente el costo computacional, validando las optimizaciones presentadas en la última investigación oficial sobre la arquitectura xLSTM.

---
*Mediciones realizadas en hardware idéntico para asegurar la validez de la comparativa.*
