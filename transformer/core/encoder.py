import torch
import torch.nn as nn
from torch.nn import Module

from transformer.base.positional_encoding import PositionalEncoding
from transformer.base.encoder_layer import EncoderLayer

from transformer.config.encoder import (
    EncoderConfig,
    EncoderLayerConfig
)
from transformer.config.positional_encoding import PositionalEncodingConfig

class Encoder(Module):
    def __init__(
        self,
        config: EncoderConfig
    ):
        super().__init__()
        self.num_layers = config.num_layers
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length
        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        pos_enc_config = PositionalEncodingConfig(
            embed_dim=self.embed_dim,
            max_length=self.max_seq_length,
            dropout=self.dropout
        )
        self.pos_enc = PositionalEncoding(pos_enc_config)

        layer_config = EncoderLayerConfig(
            embed_dim=self.embed_dim,
            num_head=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(layer_config)
                for _ in range (self.num_layers)
            ]
        )

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        # Embedding
        x = self.embedding(x)
        # Positional Encoding
        x = self.pos_enc(x)
        # Dropout
        x = self.dropout(x)
        # Encoder Layer
        for layer in self.layers:
            x = layer(x, mask)

        return x