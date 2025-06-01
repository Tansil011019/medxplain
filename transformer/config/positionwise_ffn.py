from dataclasses import dataclass

@dataclass
class PositionWiseFeedForwardConfig:
    embed_dim: int
    hidden_dim: int
    dropout: float