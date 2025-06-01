from dataclasses import dataclass

@dataclass
class DecoderLayerConfig:
    embed_dim: int
    num_heads: int
    hidden_dim: int
    dropout: float

@dataclass
class DecoderConfig:
    pass 