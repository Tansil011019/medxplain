from dataclasses import dataclass

@dataclass
class ViTEncoderConfig:
    height: int
    width: int
    patch_size: int
    in_channel: int
    embed_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    dropout_p: float