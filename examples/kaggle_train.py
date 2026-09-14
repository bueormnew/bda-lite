"""Kaggle: pip install 'git+https://github.com/bueormnew/bda-lite.git[train]'"""
import itertools
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from bda_lite import BDAConfig, BDALanguageModel, StreamingTokenBlocks, train_bpe, train_one_epoch

OUT = Path("/kaggle/working/bda_es"); OUT.mkdir(parents=True, exist_ok=True)
CONTEXT, VOCAB = 512, 8192
DATASET, TEXT_COLUMN = "SINAI/ALIA-cultural-heritage", "text"

def stream_text(skip=0):
    for row in load_dataset(DATASET, split="train", streaming=True).skip(skip):
        yield row[TEXT_COLUMN]

tokenizer = train_bpe((text[:2048] for text in itertools.islice(stream_text(), 4000)), OUT / "tokenizer.json", VOCAB)
config = BDAConfig(vocab_size=tokenizer.get_vocab_size(), d_model=256, n_heads=8, pattern=("bda", "bda", "attention"), context_length=CONTEXT)
model = BDALanguageModel(config)
print(f"parameters: {model.parameter_count:,}")

gpus = torch.cuda.device_count(); assert gpus, "Enable GPU in Kaggle"
device = torch.device("cuda:0")
if gpus > 1: model = torch.nn.DataParallel(model)
model = model.to(device)
loader = DataLoader(StreamingTokenBlocks(lambda: stream_text(skip=256), tokenizer, CONTEXT), batch_size=max(1, gpus), num_workers=0, pin_memory=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

def checkpoint(step, tokens):
    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(OUT / "latest")
    print(f"checkpoint: step={step}, tokens={tokens:,}")

metrics = train_one_epoch(model, loader, optimizer, device, checkpoint_fn=checkpoint)
raw = model.module if hasattr(model, "module") else model
raw.save_pretrained(OUT / "final")
print(metrics, "saved to", OUT / "final")
