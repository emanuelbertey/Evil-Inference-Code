# Blaurel — Gradient-Free LSTM Training

Three training paradigms for LSTM networks implemented in PyTorch, designed for easy porting to [Burn](https://burn.dev/) (Rust).

## Training Methods

| Característica | Backpropagation | Forward-Forward (FF) | PEPITA |
|---|---|---|---|
| Flujo | Forward + Backward | Forward (Positivo) + Forward (Negativo) | Forward (Base) + Forward (Modulado) |
| Uso de Memoria | Alto (guarda activaciones) | Bajo (local a la capa) | Medio |
| Cálculo de Gradiente | Cadena global (Chain rule) | No requiere (usa "Bondad") | Basado en error de entrada |
| Hardware ideal | GPUs modernas (RTX) | Hardware neuromórfico / GPUs antiguas | Hardware especializado |

## Project Structure

```
blaurel/
├── README.md
├── requirements.txt
├── train_backprop.py          # Ejemplo: entrenamiento clásico (backprop)
├── train_ff.py                # Ejemplo: entrenamiento Forward-Forward
├── train_pepita.py            # Ejemplo: entrenamiento PEPITA
├── common/
│   ├── __init__.py
│   ├── data.py                # Dataset y utilidades de datos
│   ├── metrics.py             # Métricas compartidas
│   └── lstm_cell.py           # Celda LSTM base (compartida)
├── ff/
│   ├── __init__.py
│   ├── ff_lstm_layer.py       # Capa LSTM con entrenamiento FF
│   └── ff_model.py            # Modelo completo FF-LSTM
└── pepita/
    ├── __init__.py
    ├── pepita_lstm_layer.py   # Capa LSTM con entrenamiento PEPITA
    └── pepita_model.py        # Modelo completo PEPITA-LSTM
```

## Quick Start

```bash
pip install -r requirements.txt

# Entrenar con Backpropagation (baseline)
python train_backprop.py

# Entrenar con Forward-Forward
python train_ff.py

# Entrenar con PEPITA
python train_pepita.py
```

## Design Principles

1. **Código limpio**: Cada módulo es independiente y autocontenido.
2. **Portable a Rust/Burn**: Se evitan abstracciones complejas de PyTorch; las operaciones son explícitas a nivel de tensor.
3. **Sin magia**: No se usa `autograd` en FF ni PEPITA — los pesos se actualizan manualmente.

## References

- Hinton, G. (2022). *The Forward-Forward Algorithm: Some Preliminary Investigations*.
- Dellaferrera, G. & Bhatt, S. (2022). *PEPITA: Perceptive Propagation by Iterative Adaptation*.
