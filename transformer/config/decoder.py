from dataclasses import dataclass

@dataclass
class DecoderLayerConfig:
    embed_dim: int
    num_heads: int
    hidden_dim: int
    dropout: float

@dataclass
class DecoderConfig:
    num_layers: int
    embed_dim: int
    num_heads: int
    hidden_dim: int
    vocab_size: int
    max_seq_length: int
    dropout: float