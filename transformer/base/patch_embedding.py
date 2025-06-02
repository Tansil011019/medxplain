import torch
import torch.nn as nn
from torch.nn import Module

from einops.layers.torch import Rearrange

from transformer.config.patch_embedding import PatchEmbeddingConfig

class PatchEmbedding(Module):
    def __init__(
        self,
        config: PatchEmbeddingConfig
    ):
        super().__init__()
        self.height = config.height
        self.width = config.width
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim

        self.num_patches_h = self.height // self.patch_size
        self.num_patches_w = self.width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        assert self.height % self.patch_size == 0, "Image height should be divisible by patch size"
        assert self.width % self.patch_size == 0, "Image width should be divisible by patch size"

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size),
            nn.Linear((self.patch_size ** 2) * self.in_channels, self.embed_dim)
        )
    
    def forward(self, x):
        x = self.projection(x)
        return x