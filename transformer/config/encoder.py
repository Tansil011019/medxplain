from dataclasses import dataclass

@dataclass
class EncoderLayerConfig:
    embed_dim: int
    num_head: int
    hidden_dim: int
    dropout: float

@dataclass
class EncoderConfig:
    num_layers: int
    embed_dim: int
    num_heads: int
    hidden_dim: int
    vocab_size: int
    max_seq_length: int
    dropout: float