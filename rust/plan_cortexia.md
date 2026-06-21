# Plan Cortexia

## Concepto

Modelo MoE modular con backbone compartido y expertos especializados por dominio, todo en BitLinear (2 bits/peso). Cada experto es a su vez un MoE interno con sub-expertos de tamaño adaptativo. El grafo de conocimiento se usa para estructurar el entrenamiento (definir dominios, construir datasets, guiar la especialización). El ruteo durante inferencia lo hace un **router entrenado** (gating network) como en MoE clásico. MTP (Multi-Token Prediction) compartido para planificación.

## Arquitectura General

```
[Input]
   │
   ▼
┌──────────────────────────────────────┐
│   Shared Backbone ("Vértebra")       │
│   - Embedding (f16)                  │
│   - N capas BitLineares generales    │
│   - RMS norms                        │
│   - No tiene head                    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│   Router Entrenado (Gating)          │
│   - MLP ligero sobre hidden state    │
│   - Top-K expertos por token         │
│   - Entrenado con gradiente          │
│   - Inicializado con prior del grafo │
└────────┬─────────────────────────────┘
         │
    ┌────┴──────────┬──────────┐
    ▼                ▼          ▼
┌────────┐     ┌────────┐  ┌────────┐
│Exp A   │     │Exp B   │  │Exp C   │ ...
│(medic.)│     │(legal) │  │(técnico)│
│ MoE    │     │ MoE    │  │ MoE    │
│ 3 sub  │     │ 2 sub  │  │ 4 sub  │
└───┬────┘     └───┬────┘  └───┬────┘
    │              │           │
    ▼              ▼           ▼
┌──────────────────────────────────────┐
│   MTP Shared Head                    │
│   - N heads de predicción futura     │
│   - Planificación multi-token        │
│   - Entrenado con backbone + experto │
└────────┬─────────────────────────────┘
         │
         ▼
      [Output]
```

## Componentes Detallados

### 1. Shared Backbone (Vértebra)

- Embedding f16 compartido por todos los expertos
- RMS norms por capa (attn_norm, ffn_norm, final_norm)
- N capas BitLineares base (ej: 8 de 24 capas totales)
- Procesa el input hasta una representación intermedia
- Se entrena con datos de todos los dominios
- Congelado durante fine-tuning de expertos
- Tamaño fijo: ~30-40% del modelo total

### 2. Grafo de Conocimiento para Entrenamiento

El grafo de conocimiento se usa para estructurar y guiar el entrenamiento de los expertos, no para ruteo en inferencia. El ruteo lo hace un **gating network entrenado** (router clásico de MoE).

#### Construcción del Grafo (Pre-entrenamiento)
- Cada dominio = un nodo en el grafo
- Aristas ponderadas por co-ocurrencia de términos, citation overlap, o similaridad semántica
- El grafo se construye del corpus antes de entrenar
- Define los límites y relaciones entre dominios

#### Uso del Grafo en Entrenamiento
- **Segmentación del corpus**: cada documento se etiqueta con su nodo más cercano en el grafo
- **Inicialización del router**: los pesos del gating network se inicializan con la estructura del grafo (prior topológico)
- **Balanceo de carga**: el grafo informa qué expertos necesitan más capacidad (tamaño adaptativo)
- **Curriculum learning**: primero se entrena el backbone, luego expertos individuales, luego el router
- **Nuevos dominios**: se añade el nodo al grafo, se etiquetan datos nuevos, se entrena un nuevo experto sin tocar los existentes

#### Router Entrenado (Gating Network)
- MLP ligero (2-3 capas) sobre el hidden state del backbone
- Entrenado por gradiente (softmax + load balancing loss)
- Top-K expertos por token (como MoE estándar)
- Soporta activar múltiples expertos por consulta

#### Multi-expertos por Consulta
- El router puede activar múltiples expertos (ej: "medicina legal" → med + legal)
- Los expertos activos procesan en paralelo
- Las salidas se combinan ponderadas por los scores del router
- Fusión por: promedio ponderado, atención, o stacking lineal

#### Beneficios del Enfoque Híbrido (Grafo + Router Entrenado)
| Aspecto | Solo Grafo | Grafo + Router Entrenado |
|---|---|---|
| Precisión de ruteo | limitada a similitud coseno | aprende con gradiente |
| Especialización | estática | dinámica por entrenamiento |
| Nuevos dominios | añadir nodo y listo | añadir nodo + fine-tune router |
| Interpretabilidad | grafo visible | grafo inicial + pesos entrenados |
| Adaptación a consultas ambiguas | no | sí, el router aprende matices |

### 3. Expertos MoE con Tamaño Adaptativo

#### Estructura de cada Experto
Cada experto es un modelo TransformerBitLinear completo pero:
- Comparte backbone (vértebra) — no lo replica
- Tiene capas BitLineares extra propias del dominio
- Puede ser MoE internamente con sub-expertos
- El tamaño del experto = f(tamaño del dominio en el corpus)

#### Tamaño Adaptativo
```
Dominio pequeño (500K docs):
  ┌──────────────────┐
  │ Backbone (shared)│
  │ + 2 capas extra  │ ← experto chico
  │ Sin sub-expertos │
  └──────────────────┘

Dominio grande (5M docs):
  ┌────────────────────┐
  │ Backbone (shared)  │
  │ + 6 capas extra    │
  │ + 4 sub-expertos   │ ← MoE interno
  │ por capa           │
  └────────────────────┘
```

#### Sub-expertos Internos
- Cada capa extra del experto puede tener sub-expertos (MoE interno)
- Número de sub-expertos variable según la capa
- Gating local por capa (no global)
- Permite escalar un experto sin aumentar FLOPs por token

#### Fórmula de Tamaño
```
params_experto = params_backbone + Σ(capas_extra_i × params_por_capa × sub_expertos_i)
FLOPs_por_token = FLOPs_backbone + Σ(capas_extra_i × FLOPs_por_capa × activos_i)
```

Donde `activos_i` = top-2 sub-expertos por capa (MoE estándar).

### 4. MTP (Multi-Token Prediction) Compartido

#### Funcionamiento
- Head que predice los próximos N tokens desde la representación actual
- Compartido entre todos los expertos
- Se entrena junto al backbone, fine-tune por experto
- Arquitectura: N cabezas lineales (o BitLineares) sobre el hidden state

#### Beneficios
- Planificación a largo plazo durante generación
- Coherencia contextual entre tokens futuros
- Permite verificación interna (si N=2, verifica el token-2 antes de emitir token-1)
- Reduce el "drifting" en generaciones largas

#### Integración con Expertos
- El MTP head recibe la salida del experto activo
- Durante fine-tuning de experto, solo se afinan las capas del experto + MTP
- El backbone y MTP base se mantienen

### 5. Sistema de Inferencia Rápida

#### 1. Early Exit por Experto
- Cada capa del experto puede decidir si continuar o emitir temprano
- Threshold de confianza por dominio
- Reduce latencia en consultas simples

#### 2. Cache de Ruteo
- El resultado del router se cachea por sesión (mismo prompt → mismo experto)
- Si el usuario sigue el mismo tema, no re-rutea

#### 3. Fusión de Caches KV por Experto
- Cada experto mantiene su KV cache persistente
- Al cambiar de experto, se preserva el cache del anterior
- Vuelta rápida si el usuario retoma el tema anterior

#### 4. Paralelismo de Expertos
- Los top-K expertos activos procesan en paralelo
- Kernel I2S optimizado para batch de mini-expertos
- Escalable a más núcleos (cada experto en su thread)

#### 5. Quantización Dinámica de Activaciones
- Las activaciones entre capas se cuantizan a i8 (como ya hace BitLinear)
- Para expertos grandes, se puede bajar a i4 en capas profundas

### 6. Aprendizaje Continuo

- Nuevo dominio → nuevo nodo en el grafo
- Se entrena solo el nuevo experto (backbone congelado)
- Se añaden aristas al grafo automáticamente por similaridad
- No requiere retraining del modelo completo

## Ejemplo de Flujo Completo

```
Usuario: "Dolor de cabeza con fiebre >38°"

1. Backbone: procesa el prompt → embedding contextual
2. Router entrenado (gating):
   - Score "medicina": 0.85
   - Score "farmacología": 0.45
   - Score "general": 0.10
   - Top-2: medicina (peso 0.65) + farmacología (peso 0.35)
3. Experto medicina (MoE, 3 sub, top-2 activos):
   - Capas extra 1-4: procesan con sub-expertos
   - Salida específica del dominio
4. Experto farmacología (MoE, 2 sub, top-1 activo):
   - Capas extra 1-2 (más chico)
   - Salida específica
5. Fusión: 0.7 × med + 0.3 × farma
6. MTP: predice próximo token + verifica token+1
7. Output: "Podría ser una infección bacteriana..."
```

## Ventajas con BitNet

| Aspecto | Modelo tradicional | Con BitNet (2 bits) |
|---|---|---|
| 170M params | 680 MB f32 | ~48 MB |
| 10 expertos × 50M | 2 GB | ~140 MB |
| KV cache 32K | ~1.5 GB | ~1.5 GB (no cambia) |
| **Total residente** | **~4 GB** | **~700 MB** |

Todos los expertos caben en RAM → latencia cero de swapping.

Con tamaños adaptativos, los expertos chicos ocupan ~10 MB (2 bits). Un sistema con 1 backbone + 20 expertos (10 chicos + 6 medianos + 4 grandes) cabe en ~500 MB.

## Plan de Implementación

### Fase 1: Backbone + Graph Router
1. Backbone: TransformerBitLinear existente, separar en shared + expert layers
2. Graph Router: construir grafo desde corpus etiquetado
3. Export del backbone a .bitnet (inferencia rápida)

### Fase 2: Expertos MoE
1. Fork del backbone con capas extra por dominio
2. Entrenamiento separado por dominio
3. Export a .bitnet individual

### Fase 3: MTP
1. Head de múltiples tokens sobre backbone
2. Entrenar con backbone y fine-tune por experto

### Fase 4: Inferencia Unificada
1. Pipeline: backbone → router → experto(s) → MTP → token
2. Fusión de salidas multi-experto
3. Sistema de caches KV separadas por experto

## Optimizaciones Futuras

- Compartición dinámica de capas entre expertos (cercanía en grafo)
- Auto-expansión de nodos cuando un dominio crece
- Fine-tuning continuo por nodo sin afectar al resto
- Kernel I2S optimizado para Batch × Expertos paralelos
- Compresión del grafo para sub-dominios (jerarquía de 2 niveles)
- Pre-carga de experto predictivo (cargar el experto más probable antes de que termine el backbone)

