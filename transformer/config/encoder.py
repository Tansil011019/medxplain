from dataclasses import dataclass

@dataclass
class EncoderConfig:
    embed_dim: int
    num_head: int
    hidden_dim: int
    dropout: float