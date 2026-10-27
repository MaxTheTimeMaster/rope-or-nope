from dataclasses import dataclass

from sympy.geometry.line import t
import torch
from torch import nn

from tasks import VOCAB

@dataclass(frozen=True)
class ModelConfig:
    positions: str  # learned | sinusoidal | rope | alibi | nope
    description: str
    norm: str = "pre_ln"  # pre_ln | rmsnorm
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 256  # longest eval sequence: BOS + 64 + SEP + 64 + EOS = 131


ARCHITECTURES = {
    "learned": ModelConfig("learned", "Learned absolute position embeddings, GPT-2 style"),
    "sinusoidal": ModelConfig("sinusoidal", "Fixed sin/cos embeddings from 'Attention Is All You Need'"),
    "rope": ModelConfig("rope", "Rotary embeddings: rotate queries and keys by position, LLaMA style"),
    "alibi": ModelConfig("alibi", "No embeddings, attention penalty grows with distance"),
    "nope": ModelConfig("nope", "No positional information at all, only the causal mask"),
}


def sinusoidal_table(max_len: int, d_model: int, base: float=10000.0) -> torch.Tensor:
    """Fixed position table, shape (max_len, d_model).

    table[pos, 2i]   = sin(pos / base ** (2i / d_model))
    table[pos, 2i+1] = cos(pos / base ** (2i / d_model))
    """
    table = torch.zeros((max_len, d_model)) # m x d
    z = torch.arange(max_len) # m
    z_full = z.repeat(d_model, 1).T # (d x m).T -> m x d
    i = torch.arange(d_model) # d
    i = i // 2 # d
    theta = torch.pow(base, -2.0 * i / d_model) # d
    theta_full = theta.repeat(max_len, 1) # m x d
    table[:, 0::2] = torch.sin(z_full * theta_full)[:, 0::2] # m x d
    table[:, 1::2] = torch.cos(z_full * theta_full)[:, 1::2] # m x d

    return table


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding for queries or keys.

    x: (B, H, T, Dh), Dh even. The vector at position t is split into Dh / 2 pairs;
    pair i is rotated by the angle t * base ** (-2i / Dh). Same shape out.
    How you pair coordinates (neighbours or two halves) is up to you.
    """
    B, H, T, Dh = x.shape
    ts = torch.arange(T, device=x.device) # T
    x = x.reshape(B, H, T, Dh // 2, 2)
    angles = ts.float().reshape(-1, 1) @ torch.pow(base, -2.0 * torch.arange(Dh // 2) / Dh).reshape(1, -1) # T x Dh // 2
    rotation_matrix = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1).reshape(T, Dh // 2, 2, 2)
    x = x.unsqueeze(-2) @ rotation_matrix
    x = x.reshape(B, H, T, Dh)

    return x


def alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """ALiBi bias added to attention scores before softmax, shape (H, T, T).

    bias[h, i, j] = -slope_h * (i - j) for j <= i, slope_h = 2 ** (-8 * (h + 1) / n_heads).
    Values above the diagonal don't matter: the causal mask removes them.
    """
    slope_h = 2 ** (-8 * (torch.arange(n_heads) + 1) / n_heads) # (H,)
    m = torch.tril(torch.ones(seq_len, seq_len)) * (torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]) # (T, T)
    bias = -slope_h[:, None, None] * m[None, :, :] # (H, T, T)

    return bias



class RMSNorm(nn.Module):
    """x / sqrt(mean(x ** 2) + eps) * weight over the last dim; weight starts as ones."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, cfg.n_heads heads of size d_model / n_heads.

    Applies apply_rope to queries and keys when cfg.positions == "rope".
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads

        self.rope = self.cfg.positions == "rope"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)



    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, D) -> (B, T, D). Position t sees only positions <= t.

        bias: optional (H, T, T), added to the scores before softmax (ALiBi).
        """

        B, T, D = x.shape

        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        if bias is None:
            bias = torch.zeros((self.n_heads, T, T), device=x.device)  # (H, T, T)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)

        if self.rope:
            q = apply_rope(q)
            k = apply_rope(k)

        mask = torch.where(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'), 0) # (T, T)


        logits = q @ k.transpose(-2, -1)  # (B, H, T, Dh) @ (B, H, Dh, T) -> (B, H, T, T)
        attention_scores = torch.softmax(logits / (self.head_dim ** 0.5) + bias + mask, dim=-1)  # (B, H, T, T)
        score = (attention_scores @ v).transpose(1, 2)  # (B, H, T, T) @ (B, H, T, Dh) -> (B, T, H, Dh)
        score = score.reshape(B, T, self.n_heads * self.head_dim)  # (B, T, D)

        return self.out(score)

class Block(nn.Module):
    """Pre-norm block: x = x + attn(norm1(x)); x = x + mlp(norm2(x)).

    norm: nn.LayerNorm for cfg.norm == "pre_ln", RMSNorm for "rmsnorm".
    mlp: Linear(D, 4D) -> GELU -> Linear(4D, D).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

        if cfg.norm == "pre_ln":
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
        elif cfg.norm == "rmsnorm":
            self.norm1 = RMSNorm(cfg.d_model)
            self.norm2 = RMSNorm(cfg.d_model)
        else:
            raise ValueError(f"Unknown norm: {cfg.norm}")

        self.attn = CausalSelfAttention(cfg)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), bias)  # Attention
        x = x + self.mlp(self.norm2(x)) # Token thinks about himself
        return x


class TinyGPT(nn.Module):
    """Decoder-only transformer: ids (B, T) -> logits (B, T, vocab_size).

    Where position information enters, by cfg.positions:
        learned     nn.Embedding(cfg.max_len, D), added to token embeddings
        sinusoidal  sinusoidal_table, added to token embeddings; a buffer, not trained
        rope        inside attention, see apply_rope
        alibi       alibi_bias, passed to every block
        nope        nowhere
    After the blocks: final norm (same kind as in blocks) -> Linear(D, vocab_size).
    Raises ValueError if T > cfg.max_len.
    """

    def __init__(self, cfg: ModelConfig, vocab_size: int = len(VOCAB)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model)

        if cfg.positions == "sinusoidal":
            self.register_buffer("sinusoidal_table", sinusoidal_table(cfg.max_len, cfg.d_model))

        if cfg.positions == "learned":
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.Blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        self.cfg = cfg
        self.fc = nn.Linear(cfg.d_model, vocab_size)

        if self.cfg.norm == "pre_ln":
            self.last_ln = nn.LayerNorm(cfg.d_model)
        elif self.cfg.norm == "rmsnorm":
            self.last_ln = RMSNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:

        if ids.shape[1] > self.cfg.max_len:
            raise ValueError(f"Sequence length {ids.shape[1]} exceeds max length {self.cfg.max_len}")

        x = self.emb(ids)

        _, T, D = x.shape
        assert D == self.cfg.d_model

        if self.cfg.positions == "sinusoidal":
            x = x + self.sinusoidal_table[:T, :]
        elif self.cfg.positions == "learned":
            x = x + self.pos_emb(torch.arange(T, device=ids.device))

        for i in range(self.cfg.n_layers):
            if self.cfg.positions == "alibi":
                bias = alibi_bias(self.cfg.n_heads, T)
            else:
                bias = None

            x = self.Blocks[i](x, bias=bias)

        x = self.fc(self.last_ln(x))
        return x
