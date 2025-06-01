from dataclasses import dataclass

@dataclass
class PositionalEncodingConfig:
    embed_dim: int
    max_length: int
    dropout: int