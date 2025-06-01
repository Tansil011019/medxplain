import torch
import torch.nn as nn
from torch.nn import Module

from transformer.base.positional_encoding import PositionalEncoding
from transformer.base.decoder_layer import DecoderLayer

from transformer.config.decoder import (
    DecoderLayerConfig,
    DecoderConfig
)
from transformer.config.positional_encoding import PositionalEncodingConfig

class Decoder(Module):
    def __init__(
        self,
        config: DecoderConfig
    ):
        super().__init__()
        self.num_layers = config.num_layers
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.hidden_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length
        self.dropout_p = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        pos_enc_config = PositionalEncodingConfig(
            embed_dim=self.embed_dim,
            max_length=self.max_seq_length,
            dropout=self.dropout_p
        )
        self.pos_enc = PositionalEncoding(pos_enc_config)

        layer_config = DecoderLayerConfig(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_p
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(layer_config)
                for _ in range (self.num_layers)
            ]
        )

        self.dropout = nn.Dropout(self.dropout_p) 

        self.out = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # Embedding
        x = self.embedding(x)
        # Positional Encoding
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)

        out_logits = self.out(x)
        return out_logits
