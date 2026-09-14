# BDA-Lite

Biblioteca pequeña para construir modelos causales PyTorch con patrones configurables.
El patrón por defecto, `("bda", "bda", "attention")`, usa dos mezcladores BDA-Lite
baratos y una atención global causal para recuperar información de los 512 tokens.

## Rendimiento y alcance

`bda` es un mezclador delta causal vectorizado: proyecciones, conv1d depthwise causal
y gate. No tiene un loop Python por token. Es distinto al núcleo recurrente BUEORM de
referencia; se diseñó como una base que pueda preentrenarse en una GPU normal.

`attention` utiliza `torch.nn.functional.scaled_dot_product_attention`; PyTorch escoge
FlashAttention o el kernel memory-efficient CUDA cuando el entorno lo soporta.

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
