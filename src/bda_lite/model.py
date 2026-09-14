from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BDAConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(width)); self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, ratio: float):
        super().__init__()
        hidden = int(width * ratio)
        self.gate, self.value, self.out = nn.Linear(width, hidden, bias=False), nn.Linear(width, hidden, bias=False), nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class FastBDAMixer(nn.Module):
    """Fully vectorized BDA-Lite mixer, with no Python loop over tokens.

    It is deliberately distinct from the sequential BUEORM reference core: the
    delta operation is a causal depthwise filter and a learned gate. This makes it
    a robust, fast building block for initial GPU pretraining.
    """
    def __init__(self, config: BDAConfig):
        super().__init__()
        d = config.d_model
        self.content, self.gate = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.delta = nn.Conv1d(d, d, config.bda_kernel_size, groups=d, bias=True)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = self.content(x)
        delta = self.delta(F.pad(content.transpose(1, 2), (self.delta.kernel_size[0] - 1, 0))).transpose(1, 2)
        return self.out(torch.sigmoid(self.gate(x)) * (content + F.silu(delta)))


class CausalAttention(nn.Module):
    """Causal attention using PyTorch SDPA (Flash/memory-efficient CUDA when available)."""
    def __init__(self, config: BDAConfig):
        super().__init__()
        self.heads, self.head_dim, self.dropout = config.n_heads, config.d_model // config.n_heads, config.dropout
        self.qkv, self.out = nn.Linear(config.d_model, 3 * config.d_model, bias=False), nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(z): return z.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: BDAConfig, kind: str):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        self.mixer = FastBDAMixer(config) if kind == "bda" else CausalAttention(config)
        self.mlp = SwiGLU(config.d_model, config.mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BDALanguageModel(nn.Module):
    def __init__(self, config: BDAConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config, kind) for kind in config.pattern])
        self.norm, self.lm_head = RMSNorm(config.d_model), nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None: nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.context_length:
            raise ValueError(f"token_ids must be (batch, <= {self.config.context_length})")
        x = self.embed(token_ids)
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm(x))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        torch.save({"model": self.state_dict()}, directory / "model.pt")

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str | torch.device = "cpu") -> "BDALanguageModel":
        directory = Path(directory)
        model = cls(BDAConfig.load(directory / "config.json"))
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True)["model"])
        return model.to(device).eval()
