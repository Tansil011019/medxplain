import torch
import torch.nn as nn
from torch.nn import Module

from transformer.base.patch_embedding import PatchEmbedding
from transformer.base.encoder_layer import EncoderLayer

from transformer.config.vit_encoder import ViTEncoderConfig
from transformer.config.patch_embedding import PatchEmbeddingConfig
from transformer.config.encoder import EncoderLayerConfig

class ViTEncoder(Module):
    def __init__(
        self,
        config: ViTEncoderConfig
    ):
        super().__init__()
        self.height = config.height
        self.width = config.width
        self.patch_size = config.patch_size
        self.in_channel = config.in_channel
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.dropout_p = config.dropout_p

        patch_emb_config = PatchEmbeddingConfig(
            height=self.height,
            width=self.width,
            patch_size=self.patch_size,
            in_channels=self.in_channel,
            embed_dim=self.embed_dim
        )
        self.patch_embedding = PatchEmbedding(patch_emb_config)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )

        num_positions = self.patch_embedding.num_patches + 1
        self.positional_embedding = nn.Parameter(torch.randn(1, num_positions, self.embed_dim))

        self.dropout = nn.Dropout(self.dropout_p)

        enc_layer_config = EncoderLayerConfig(
            embed_dim=self.embed_dim,
            num_head=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_p
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_layer_config)]
            for _ in range (self.num_layers)
        )

        self.final_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x_img, mask=None):
        batch_size = x_img.shape[0]
        patch_emb = self.patch_embedding(x_img)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, patch_emb), dim=1)

        x = x + self.positional_embedding

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        return x
