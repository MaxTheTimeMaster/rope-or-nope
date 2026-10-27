#### CLAUDE TESTS
import dataclasses

import pytest
import torch
from model import ARCHITECTURES, RMSNorm, TinyGPT, alibi_bias, apply_rope, sinusoidal_table
from tasks import VOCAB

ALL_CONFIGS = [
    pytest.param(dataclasses.replace(cfg, norm=norm), id=f"{name}-{norm}")
    for name, cfg in ARCHITECTURES.items()
    for norm in ("pre_ln", "rmsnorm")
]


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_forward_shape(cfg):
    model = TinyGPT(cfg)
    ids = torch.randint(len(VOCAB), (2, 10))
    assert model(ids).shape == (2, 10, len(VOCAB))


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_future_tokens_do_not_leak(cfg):
    model = TinyGPT(cfg).eval()
    ids = torch.randint(len(VOCAB), (2, 12))
    changed = ids.clone()
    changed[:, 6:] = torch.randint(len(VOCAB), (2, 6))
    with torch.no_grad():
        original, perturbed = model(ids), model(changed)
    torch.testing.assert_close(original[:, :6], perturbed[:, :6])


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_runs_on_longest_eval_sequence(name):
    model = TinyGPT(ARCHITECTURES[name])
    # <bos> + 64 digits + <sep> + 64 digits; the final <eos> is only ever a target
    ids = torch.randint(len(VOCAB), (1, 130))
    assert model(ids).shape == (1, 130, len(VOCAB))


def test_rejects_sequences_longer_than_max_len():
    cfg = ARCHITECTURES["learned"]
    with pytest.raises(ValueError):
        TinyGPT(cfg)(torch.zeros(1, cfg.max_len + 1, dtype=torch.long))


def test_sinusoidal_table():
    table = sinusoidal_table(50, 16)
    assert table.shape == (50, 16)
    torch.testing.assert_close(table[0, 0::2], torch.zeros(8))
    torch.testing.assert_close(table[0, 1::2], torch.ones(8))
    assert table.abs().max() <= 1.0


def test_rope_keeps_shape_and_norm():
    x = torch.randn(2, 4, 10, 16)
    y = apply_rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_rope_leaves_position_zero_alone():
    x = torch.randn(2, 4, 10, 16)
    torch.testing.assert_close(apply_rope(x)[:, :, 0], x[:, :, 0])


def test_rope_scores_depend_only_on_distance():
    # Same q and k at every position: after RoPE, q_m . k_n must depend on m - n only
    T = 12
    q = torch.randn(16).repeat(1, 1, T, 1)
    k = torch.randn(16).repeat(1, 1, T, 1)
    scores = (apply_rope(q) @ apply_rope(k).transpose(-1, -2))[0, 0]
    torch.testing.assert_close(scores[1:, 1:], scores[:-1, :-1])


def test_alibi_bias():
    n_heads, seq_len = 4, 6
    bias = alibi_bias(n_heads, seq_len)
    assert bias.shape == (n_heads, seq_len, seq_len)
    for h in range(n_heads):
        slope = 2 ** (-8 * (h + 1) / n_heads)
        for i in range(seq_len):
            for j in range(i + 1):
                assert bias[h, i, j].item() == pytest.approx(-slope * (i - j))


def test_rmsnorm_output_has_unit_rms():
    x = torch.randn(3, 5, 64) * 7 + 2
    y = RMSNorm(64)(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4, rtol=1e-4)
