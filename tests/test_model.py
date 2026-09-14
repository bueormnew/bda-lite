import torch
import torch.nn.functional as F

from bda_lite import BDAConfig, BDALanguageModel


def test_forward_backward_and_causality():
    model = BDALanguageModel(BDAConfig(vocab_size=128, d_model=64, n_heads=4, context_length=64))
    tokens = torch.randint(0, 128, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, 128)
    F.cross_entropy(logits[:, :-1].reshape(-1, 128), tokens[:, 1:].reshape(-1)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    changed = tokens.clone(); changed[0, 16] = (changed[0, 16] + 1) % 128
    assert (model(tokens)[:, :16] - model(changed)[:, :16]).abs().max().item() == 0.0
