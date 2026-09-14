from __future__ import annotations

from collections.abc import Callable, Iterable
import time

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm


class StreamingTokenBlocks(IterableDataset):
    """Constant-RAM next-token blocks from a factory that yields text."""
    def __init__(self, text_factory: Callable[[], Iterable[str]], tokenizer, context_length: int, max_chars: int = 8192):
        self.text_factory, self.tokenizer = text_factory, tokenizer
        self.context_length, self.max_chars = context_length, max_chars
        self.bos, self.eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")

    def __iter__(self):
        pending: list[int] = []
        for text in self.text_factory():
            for start in range(0, len(text), self.max_chars):
                pending.extend([self.bos, *self.tokenizer.encode(text[start:start + self.max_chars]).ids, self.eos])
                while len(pending) >= self.context_length + 1:
                    yield torch.tensor(pending[:self.context_length + 1], dtype=torch.long)
                    del pending[:self.context_length + 1]


def train_one_epoch(model, loader, optimizer, device, grad_accum: int = 4, checkpoint_every: int = 250, checkpoint_fn=None):
    """Memory-bounded causal training with an honest token-count progress bar."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    optimizer.zero_grad(set_to_none=True)
    bar, steps, tokens, started = tqdm(unit="tokens", desc="Training"), 0, 0, time.time()
    vocab = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        scaler.scale(loss / grad_accum).backward()
        steps, tokens = steps + 1, tokens + x.numel()
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if checkpoint_fn and steps % checkpoint_every == 0: checkpoint_fn(steps, tokens)
        bar.update(x.numel())
        bar.set_postfix(loss=f"{loss.item():.3f}", tok_s=f"{tokens / max(time.time() - started, 1e-6):.0f}")
    if steps % grad_accum:
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
    bar.close()
    return {"steps": steps, "tokens": tokens, "tokens_per_second": tokens / max(time.time() - started, 1e-6)}
