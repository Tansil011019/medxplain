from dataclasses import dataclass

@dataclass
class PatchEmbeddingConfig:
    height: int
    width: int
    patch_size: int
    in_channels: int
    embed_dim: int