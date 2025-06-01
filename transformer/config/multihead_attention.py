from dataclasses import dataclass

@dataclass
class MultiHeadAttentionConfig:
    embed_dim: int
    num_head: int
    dropout: float