from dataclasses import dataclass


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
