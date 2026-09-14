from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class BDAConfig:
    """Configurable causal model; blocks are `bda` or `attention`."""
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    pattern: tuple[str, ...] = ("bda", "bda", "attention")
    context_length: int = 512
    bda_kernel_size: int = 7
    mlp_ratio: float = 8 / 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 8 or self.d_model < 16 or self.context_length < 2:
            raise ValueError("vocab_size, d_model and context_length are too small")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not self.pattern or any(item not in {"bda", "attention"} for item in self.pattern):
            raise ValueError("pattern can contain only 'bda' and 'attention'")
        if self.bda_kernel_size < 1 or self.bda_kernel_size % 2 == 0:
            raise ValueError("bda_kernel_size must be a positive odd integer")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BDAConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["pattern"] = tuple(raw["pattern"])
        return cls(**raw)
