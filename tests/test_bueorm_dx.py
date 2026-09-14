import torch
import torch.nn.functional as F

from bda_lite import BUEORMCore, BUEORMCoreConfig, BUEORMDX, BUEORMDXConfig, GlobalLatentRelay, GlobalRelayConfig, LocalAttentionConfig


def tiny_config():
    return BUEORMDXConfig(
        vocab_size=67, d_model=32, n_macrocycles=1,
        core=BUEORMCoreConfig(32, n_heads=2, n_anchors=4, thermostat_every=4, half_life_max=128),
        local=LocalAttentionConfig(32, n_heads=2, window=8),
        relay=GlobalRelayConfig(32, chunk_size=4, latent=16, n_queries=2, summary_layers=1, summary_heads=2),
    )


def test_bueorm_core_state_and_backward_are_finite():
    core = BUEORMCore(BUEORMCoreConfig(32, n_heads=2, n_anchors=4, thermostat_every=4))
    x = torch.randn(2, 13, 32, requires_grad=True)
    y, state = core(x)
    (y.square().mean() + state["M"].square().mean()).backward()
    assert y.shape == x.shape
    assert state["M"].shape == (2, 2, 16, 16)
    assert torch.linalg.matrix_norm(state["M"], ord=2).max() <= 4.01
    assert torch.isfinite(x.grad).all()


def test_bueorm_core_streaming_matches_one_call():
    torch.manual_seed(9)
    core = BUEORMCore(BUEORMCoreConfig(16, n_heads=2, n_anchors=4, thermostat_every=1000))
    core.eval(); x = torch.randn(1, 16, 16)
    whole, _ = core(x)
    left, state = core(x[:, :7])
    right, _ = core(x[:, 7:], state)
    assert torch.allclose(torch.cat((left, right), 1), whole, atol=3e-6, rtol=3e-5)


def test_relay_streaming_matches_single_call_and_is_causal():
    torch.manual_seed(3)
    relay = GlobalLatentRelay(GlobalRelayConfig(16, chunk_size=4, latent=8, n_queries=2, summary_layers=1, summary_heads=2))
    relay.eval()
    x = torch.randn(1, 12, 16)
    full, _ = relay(x)
    first, state = relay(x[:, :5])
    second, _ = relay(x[:, 5:], state)
    assert torch.allclose(torch.cat((first, second), 1), full, atol=2e-6, rtol=2e-5)
    altered = x.clone(); altered[0, 6] += 4
    changed, _ = relay(altered)
    assert (full[:, :6] - changed[:, :6]).abs().max().item() == 0.0


def test_full_bueorm_dx_forward_backward_and_causality():
    torch.manual_seed(5)
    model = BUEORMDX(tiny_config())
    tokens = torch.randint(0, 67, (2, 16))
    logits, _ = model(tokens)
    F.cross_entropy(logits[:, :-1].reshape(-1, 67), tokens[:, 1:].reshape(-1)).backward()
    assert logits.shape == (2, 16, 67)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval(); changed = tokens.clone(); changed[0, 9] = (changed[0, 9] + 1) % 67
    a, _ = model(tokens); b, _ = model(changed)
    assert (a[:, :9] - b[:, :9]).abs().max().item() == 0.0


def test_full_model_streaming_matches_one_call():
    torch.manual_seed(15)
    model = BUEORMDX(tiny_config()).eval()
    tokens = torch.randint(0, 67, (1, 16))
    full, _ = model(tokens)
    left, state = model(tokens[:, :7])
    right, _ = model(tokens[:, 7:], state)
    assert torch.allclose(torch.cat((left, right), 1), full, atol=5e-6, rtol=5e-5)
