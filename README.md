# BDA-Lite

Biblioteca pequeña para construir modelos causales PyTorch con patrones configurables.
Incluye dos rutas explícitamente diferentes: BDA-Lite rápida y BUEORM-DX fiel a la
especificación de este proyecto.

## Rendimiento y alcance

`bda` es un mezclador delta causal vectorizado: proyecciones, conv1d depthwise causal
y gate. No tiene un loop Python por token. Es distinto al núcleo recurrente BUEORM de
referencia; se diseñó como una base que pueda preentrenarse en una GPU normal.

`attention` utiliza `torch.nn.functional.scaled_dot_product_attention`; PyTorch escoge
FlashAttention o el kernel memory-efficient CUDA cuando el entorno lo soporta.

## BUEORM-DX completo

`BUEORMDX` implementa el diseño de los documentos `BUEORM_DX_v1*.md`:

- núcleo asociativo recurrente con lectura previa, error delta y erase/write desacoplados;
- utilidad × sorpresa, precondicionamiento por energía y valores de norma fija;
- inicialización chrono de la compuerta de decaimiento;
- anchors episódicos de escritura/direccionamiento suave y lectura por atención;
- termostato espectral con power iteration persistente;
- atención local causal de ventana deslizante + RoPE parcial;
- Global Latent Relay: compresión de chunks, autoatención causal de resúmenes,
  inyección sólo de chunks completos anteriores, retrieval top-k opcional y buffer
  parcial real entre llamadas de streaming.

Las pruebas cubren forward/backward finito, causalidad y equivalencia entre procesar
una secuencia completa o por segmentos para núcleo, relay y modelo completo.

```python
from bda_lite import (
    BUEORMDX, BUEORMDXConfig, BUEORMCoreConfig,
    LocalAttentionConfig, GlobalRelayConfig,
)

cfg = BUEORMDXConfig(
    vocab_size=8192, d_model=256, n_macrocycles=1,
    core=BUEORMCoreConfig(256, n_heads=8, half_life_max=2048),
    local=LocalAttentionConfig(256, n_heads=8, window=128),
    relay=GlobalRelayConfig(256, chunk_size=64, latent=128),
)
model = BUEORMDX(cfg)
logits, streaming_state = model(token_ids)
```

La ruta BUEORM-DX conserva una recurrencia matemática por token; `compile_bueorm(model)`
puede reducir overhead de PyTorch, pero no equivale a un kernel paralelo WY/Triton. La
ruta BDA-Lite está disponible cuando la prioridad es un primer preentrenamiento rápido.

## Instalar desde GitHub

```bash
pip install "git+https://github.com/bueormnew/bda-lite.git[train]"
```

## Crear un modelo

```python
from bda_lite import BDAConfig, BDALanguageModel

cfg = BDAConfig(vocab_size=8192, d_model=256, n_heads=8,
                context_length=512, pattern=("bda", "bda", "attention"))
model = BDALanguageModel(cfg)
print(model.parameter_count)
```

El script [examples/kaggle_train.py](examples/kaggle_train.py) usa BPE subword, transmite
el corpus sin cargarlo entero en RAM, muestra progreso en tokens y exporta
`config.json`, `model.pt` y `tokenizer.json`.

```python
from bda_lite import BDALanguageModel
from tokenizers import Tokenizer

model = BDALanguageModel.from_pretrained("final")
tokenizer = Tokenizer.from_file("tokenizer.json")
```

## Pruebas

```bash
pip install -e . pytest
pytest -q
```
